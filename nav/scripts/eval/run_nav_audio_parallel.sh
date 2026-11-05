#!/bin/bash
# =============================================================================
# OmniEchoBench-Nav **spatial-audio-guided navigation** smoke eval.
#
# Backend: an external /infer + /health HTTP service per GPU (one per shard).
#   * Smoke set: randomly choose 10 scenes from case3_vln and take the first episode in each -> case3_smoke10.
#   * **Both stop conditions are run** (the stop logic lives on the agent side, independent of the service):
#       audio        : oracle_stop_dist=1.0 -- distance-based (STOP once known within <=1m of the source);
#       audio_nostop : oracle_stop_dist=0.0 -- disable distance-based oracle stopping;
#                      empty/unusable or stationary trajectories, request failures,
#                      stuck detection, replan limit, and step limit can still end an episode.
#
# Topology (each GPU i = 0..N-1 is an independent chain; N = number of GPUs = number of shards, default 2 for smoke):
#   /infer server (:8210+i) -- worker(shard case3_smoke10_g{i})
#
# Usage:
#   bash scripts/eval/run_nav_audio_parallel.sh [N_GPU] [N_SCENES]
#     N_GPU    : number of GPUs = number of shards (default 2)
#     N_SCENES : number of smoke scenes (default 10)
#
# Environment switches:
#   SERVER_BASE     base port of the pre-launched /infer servers (default 8210; server i listens on SERVER_BASE+i)
#   READY_TIMEOUT   seconds to wait for server readiness (default 2400)
#   MODES           stop modes to run (default "audio audio_nostop")
#   SMOKE_SEED      random seed for scene sampling (default 0, for reproducibility)
#   FULL_SPLIT      **full mode**: set to an existing split name (e.g. case3_vln, 900 ep/30 scene)
#                   -> skip the "sample 1 case from each of N_SCENES scenes" sub-sampling, split that
#                   split evenly by scene into N shards and run all episodes. N_SCENES is ignored here.
#     Example (8-GPU full case3_vln, nostop only):
#       MODES=audio_nostop FULL_SPLIT=case3_vln \
#         bash scripts/eval/run_nav_audio_parallel.sh 8
# =============================================================================
set -uo pipefail
# ROOT = the InternNav checkout root (must contain internnav/ +
# these overlay files merged in). When this overlay is copied into an InternNav tree under
# scripts/eval/, ../../ resolves to the InternNav root exactly as upstream. Otherwise set
# INTERNNAV_ROOT explicitly.
ROOT=${INTERNNAV_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
cd "$ROOT"

# ---- interpreters / external scripts (all overridable via env) ----
PY39=${PY39:-python}                                              # split / merge helpers
TORCHRUN=${TORCHRUN:-torchrun}                                    # worker driver
SPLITTER="$ROOT/scripts/eval/spatial_audio/split_nway.py"
MERGER="$ROOT/scripts/eval/spatial_audio/merge_audio_parallel.py"
export OMNIECHO_BENCH_ROOT="${OMNIECHO_BENCH_ROOT:-$ROOT/data/OmniEchoBench}"
export OMNIECHO_NAV_SCENES_DIR="${OMNIECHO_NAV_SCENES_DIR:-$OMNIECHO_BENCH_ROOT/OmniEchoBench-NAV/scene_data/case3_ce}"
export OMNIECHO_NAV_EP_DIR="${OMNIECHO_NAV_EP_DIR:-$OMNIECHO_NAV_SCENES_DIR/episodes}"
export OMNIECHO_NAV_AUDIO_ROOT="${OMNIECHO_NAV_AUDIO_ROOT:-$OMNIECHO_BENCH_ROOT/OmniEchoBench-NAV}"
EP_DIR="$OMNIECHO_NAV_EP_DIR"

# ---- parameters ----
N_GPU=${1:-2}
N_SCENES=${2:-10}
[[ "$N_GPU" =~ ^[1-9][0-9]*$ && "$N_SCENES" =~ ^[1-9][0-9]*$ ]] || { echo "GPU and scene counts must be positive" >&2; exit 2; }
SRC_SPLIT=${SRC_SPLIT:-case3_vln}
FULL_SPLIT=${FULL_SPLIT:-}
# Full mode: SMOKE_SPLIT reuses FULL_SPLIT directly, skipping sub-sampling
if [ -n "$FULL_SPLIT" ]; then
  SMOKE_SPLIT="$FULL_SPLIT"
else
  SMOKE_SPLIT=${SMOKE_SPLIT:-case3_smoke10}
fi
MODES=${MODES:-"audio audio_nostop"}
for mode in $MODES; do
  [[ "$mode" == audio || "$mode" == audio_nostop ]] || { echo "Unsupported stop mode: $mode" >&2; exit 2; }
done
SMOKE_SEED=${SMOKE_SEED:-0}

SERVER_BASE=${SERVER_BASE:-8210}
MASTER_BASE=29610
READY_TIMEOUT=${READY_TIMEOUT:-2400}
OUT_ROOT=${OMNIECHO_NAV_OUT_ROOT:-logs/habitat_omniecho_nav}

LOGDIR="$OUT_ROOT/_run/$SMOKE_SPLIT"
mkdir -p "$LOGDIR"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOGDIR/run.status"; }

WORKER_PIDS=()

say "START audio smoke | N_GPU=$N_GPU N_SCENES=$N_SCENES modes='$MODES'"

# ============ 1) Build the N_SCENES-scene smoke set (sample 1 case per scene from SRC_SPLIT) ============
if [ -n "$FULL_SPLIT" ]; then
  say "---- 1) Full mode FULL_SPLIT=$FULL_SPLIT -> skip sub-sampling, use existing split directly ----"
  if [ ! -f "$EP_DIR/$SMOKE_SPLIT.json.gz" ]; then
    say "Full split not found: $EP_DIR/$SMOKE_SPLIT.json.gz, exiting"; exit 1
  fi
else
say "---- 1) Build $SMOKE_SPLIT ($N_SCENES scenes, 1 case each, sampled from $SRC_SPLIT, seed=$SMOKE_SEED) ----"
$PY39 - "$EP_DIR" "$SRC_SPLIT" "$SMOKE_SPLIT" "$N_SCENES" "$SMOKE_SEED" <<'PY' 2>&1 | tee -a "$LOGDIR/make_smoke.log"
import gzip, json, os, random, sys
ep_dir, src, dst, n, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
def scene_key(sid):
    for part in sid.replace('.glb','').split('/'):
        if part.startswith('scene_'): return part
    return sid
d = json.load(gzip.open(os.path.join(ep_dir, f'{src}.json.gz'), 'rt'))
eps = d['episodes']; vocab = d.get('instruction_vocab')
# take the first episode from each scene
by_scene = {}
for e in eps:
    by_scene.setdefault(scene_key(e['scene_id']), []).append(e)
scenes = list(by_scene.keys())
random.Random(seed).shuffle(scenes)
pick = sorted(scenes[:n])
sub = [by_scene[s][0] for s in pick]
obj = {'episodes': sub}
if vocab is not None: obj['instruction_vocab'] = vocab
with gzip.open(os.path.join(ep_dir, f'{dst}.json.gz'), 'wt') as f: json.dump(obj, f)
# align the audio sidecar
side = {}
sc_path = os.path.join(ep_dir, f'{src}_audio_sidecar.jsonl')
if os.path.exists(sc_path):
    with open(sc_path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                r = json.loads(ln); side[(scene_key(str(r['scene_id'])), int(r['episode_id']))] = ln
n_side = 0
with open(os.path.join(ep_dir, f'{dst}_audio_sidecar.jsonl'), 'w') as f:
    for e in sub:
        k = (scene_key(e['scene_id']), int(e['episode_id']))
        if k in side: f.write(side[k] + '\n'); n_side += 1
print(f'{dst}: {len(sub)} eps over {len(pick)} scenes, {n_side} sidecar lines')
print('scenes:', pick)
PY
[ ${PIPESTATUS[0]:-1} -ne 0 ] && { say "Building the smoke set failed, exiting"; exit 1; }
fi

# ============ 2) Split evenly by scene into N shards ============
say "---- 2) Split $SMOKE_SPLIT into $N_GPU shards ----"
$PY39 "$SPLITTER" --split "$SMOKE_SPLIT" --n "$N_GPU" 2>&1 | tee -a "$LOGDIR/split.log"
[ ${PIPESTATUS[0]:-1} -ne 0 ] && { say "Split failed, exiting"; exit 1; }

# ============ 3) Wait for the pre-launched /infer servers to be ready (/health) ============
say "---- 3) Wait for servers to be ready (up to ${READY_TIMEOUT}s) ----"
for i in $(seq 0 $((N_GPU-1))); do
  sport=$((SERVER_BASE+i)); ok=0
  for t in $(seq 1 $((READY_TIMEOUT/10))); do
    if curl -s -m 8 "http://127.0.0.1:$sport/health" | grep -q '"status": "ok"'; then ok=1; break; fi
    sleep 10
  done
  [ "$ok" = "1" ] && say "  server g$i READY (:$sport)" || { say "  server g$i not reachable (:$sport), exiting"; exit 1; }
done

# ============ 4) Run workers per stop mode + merge ============
rc_all=0
for MODE in $MODES; do
  say "===== MODE=$MODE (${MODES}) ====="
  WORKER_PIDS=()
  for i in $(seq 0 $((N_GPU-1))); do
    sport=$((SERVER_BASE+i)); mport=$((MASTER_BASE+i))
    CUDA_VISIBLE_DEVICES=$i \
    OMNIECHO_NAV_SPLIT="${SMOKE_SPLIT}_g$i" \
    OMNIECHO_NAV_SERVER_URL="http://127.0.0.1:$sport" \
    OMNIECHO_NAV_OUT_ROOT="$OUT_ROOT" \
    MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet TOKENIZERS_PARALLELISM=false \
      nohup "$TORCHRUN" --nproc_per_node=1 --master_port="$mport" \
        scripts/eval/run_nav_eval.py --model "$MODE" \
        > "$LOGDIR/worker_${MODE}_g$i.log" 2>&1 &
    WORKER_PIDS+=($!)
    say "  worker[$MODE] g$i: GPU$i split=${SMOKE_SPLIT}_g$i -> server :$sport pid=$!"
  done
  for i in $(seq 0 $((N_GPU-1))); do
    if wait "${WORKER_PIDS[$i]}"; then say "  worker[$MODE] g$i done rc=0"
    else rc=$?; rc_all=$rc; say "  worker[$MODE] g$i exited rc=$rc"; fi
  done
  [[ "$rc_all" -eq 0 ]] || { say "Worker failure; refusing to merge stale or partial results"; exit "$rc_all"; }
  say "---- Merge MODE=$MODE ----"
  $PY39 "$MERGER" --split "$SMOKE_SPLIT" --n "$N_GPU" --out_root "$OUT_ROOT" --model "$MODE" \
    2>&1 | tee "$LOGDIR/merged_${MODE}.txt"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || exit 1
done

say "ALL DONE (worker rc_all=$rc_all)"
say "Results: audio (distance stop) -> $LOGDIR/merged_audio.txt ; audio_nostop (policy <stop>) -> $LOGDIR/merged_audio_nostop.txt"

exit "$rc_all"
