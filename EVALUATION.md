# OmniEcho evaluation guide

This document follows the shipped implementation. Examples use Bash on Linux.
See [environments/README.md](environments/README.md) for dependencies and
[nav/README.md](nav/README.md) for InternNav overlay installation.

## Data prerequisites

The package contains 2,972 QA annotations and 900 navigation episodes across
30 scenes. Navigation sidecars reference 12,390 receiver recordings, with 489
sound-event and 411 speech tasks. Checkpoints, media, and InternNav are external.

Published runtime layout:

~~~text
$OMNIECHO_BENCH_ROOT/
  OmniEchoBench-QA/
    OmniEchoBench-QA.jsonl
    videos/
    audio/
    images/
  OmniEchoBench-NAV/
    audio/case3/<scene>/<sample>/rec_pos_*.wav
    scene_data/case3_ce/
      case3/<scene>/<scene>.glb
      ... meshes, navmeshes, stage configurations, metadata ...
      episodes/
        case3_vln.json.gz
        case3_vln_audio_sidecar.jsonl
        case3_smoke.json.gz
        case3_smoke_audio_sidecar.jsonl
~~~

The published full split is `case3_vln`; the provided `case3_smoke` has 30
episodes. QA has no intermediate `media/` directory.

**Bundled `vln.*` files are older snapshots.** Their mesh paths start with
`scene_data/OmniEchoBench-Nav_ce/OmniEchoBench-Nav/`; audio paths start with
`audio/OmniEchoBench-Nav/`. The runtime does not translate those directory names
to `case3/` and `audio/case3/`. Renaming the manifests alone is insufficient.
Use published manifests/media, or explicitly prepare matching legacy assets or
converted manifests outside the source tree. Legacy absolute paths containing
`/audio/case3/` are supported by rebasing that suffix to the local NAV root.

Validate a complete mirror with:

~~~bash
export OMNIECHO_BENCH_ROOT=/path/to/OmniEchoBench
python scripts/check_benchmark_data.py \
  --root "$OMNIECHO_BENCH_ROOT" \
  --report /path/to/results/benchmark-audit.json
~~~

The checker also accepts `--oss-list` and `--oss-prefix`. See
[docs/benchmark_data.md](docs/benchmark_data.md).

## QA inference

### Flow and inputs

Entry point: `evaluate/infer_qa.py`, in the Python 3.11 / Torch 2.9.1 /
Transformers 5.2.0 inference environment.

1. Load model/processor, repair supported transposed expert tensors, disable
   model audio output.
2. Read nonempty JSONL records; zero-based record positions become `index`.
3. Send the first user turn to the model; save the first assistant turn as the
   output reference. Resolve relative media paths.
4. Process audio/vision. Video defaults: 2 FPS, at most 16 frames, at most
   448 x 448 pixels per frame. Explicit media-part values take precedence.
5. If FOA is enabled and usable, encode spatial features and inject an FOA token
   span during prefill, alongside ordinary audio/vision features.
6. Append predictions and diagnostic fields. Errors also write indexed records.

FOA channel order is [W,Y,Z,X]. Features use 16 kHz audio, FFT size 400, hop 160,
128 mel bands, and five channels: W log-mel energy, normalized X/Y/Z active
intensity, and diffuseness. A convolutional/Transformer encoder plus projector
maps them into language-model embeddings. FOA span length matches audio tokens.

Raw scaling is preserved: there is no SN3D-to-N3D conversion. Benchmark/agent
descriptions mention SN3D while feature documentation assumes N3D. Checkpoint
normalization compatibility must be verified with the intended trained assets.

### CLI options

| Option | Default | Behavior |
| --- | --- | --- |
| `--model-path` | Required | Local checkpoint; FOA requires trained thinker.foa_encoder/projector tensors |
| `--test-jsonl` | OMNIECHO_QA_GOLD or published QA path | Input annotations |
| `--media-root` | JSONL directory | Relative-media root |
| `--out-jsonl` | Required | Append-only predictions and resume ledger |
| `--foa` | Off | Additional spatial branch |
| `--no-foa` | Default | Disable FOA; semantic audio remains |
| `--mute-audio-encoder` | Off | Implicit FOA; replace semantic audio embeddings when FOA injection succeeds |
| `--no-audio-in-video` | Off | Disable muxed video audio; separate audio parts remain |
| `--limit` | 0 | Positive N limits records before sharding |
| `--num-shards` | 1 | Modulo sharding |
| `--shard-id` | 0 | Keep index % num_shards == shard_id |
| `--only-indices` | Unset | Whitespace-separated original indices, applied after limit/sharding |
| `--max-new-tokens` | 256 | Thinker output-token cap |
| `--temperature` | 0.0 | Positive enables sampling; zero leaves checkpoint generation defaults unchanged |
| `--top-p` | 0.9 | Used only with positive temperature |
| `--attn` | flash_attention_2 | Use sdpa to avoid FlashAttention dependency |
| `--dtype` | bfloat16 | Torch dtype |
| `--reason` | Off | Append cognitive-map reasoning / Answer: X instruction to every selected sample |

Important behavior:

- `--foa` can still yield `has_foa=0` if probing/decoding/features/placeholders
  fail. Inspect `has_foa` and `n_foa_pad`.
- Muting applies only with successful FOA injection and does not skip the audio
  tower computation itself.
- `--no-audio-in-video` is not a full vision-only switch when separate audio
  exists. For video-only audio it may also prevent FOA insertion.
- `--reason` does not filter question types. Select cognitive-map indices first.
- Temperature zero does not explicitly force `thinker_do_sample=False`.
- Resume treats any existing index, including errors, as done. Retry failed
  indices into a new output, then merge by original index.
- Use separate outputs per GPU. Each process handles samples one at a time.
- Inference removes blank lines before indexing; scoring enumerates physical
  gold lines. Keep benchmark JSONL free of blank lines.

### Environment variables

| Variable | Default / purpose |
| --- | --- |
| OMNIECHO_BENCH_ROOT | `<repo>/data/OmniEchoBench`; set the external complete mirror |
| OMNIECHO_QA_GOLD | Override default annotation path |
| OMNIECHO_VIDEO_FPS | 2.0 |
| OMNIECHO_VIDEO_MAX_FRAMES | 16 |
| OMNIECHO_VIDEO_MAX_PIXELS | 200704 |
| OMNIECHO_FOA_BOS_ID / OMNIECHO_FOA_PAD_ID / OMNIECHO_FOA_EOS_ID | 151676 / 151677 / 151678 |
| OMNIECHO_FOA_ENC_DIM / OMNIECHO_FOA_PROJ_HIDDEN | 384 / 2048; checkpoint contract |
| HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE | Set to 1 unless already defined |
| CUDA_VISIBLE_DEVICES | GPU visibility |

## QA scoring

Entry point: `qa_eval/eval_qa_metrics.py`.

### Extraction and rules

Gold is loaded independently and matched by `index`. Gold metadata and the final
assistant reference determine the answer; prediction-file reference/meta are not
trusted. Missing category means cognitive-map multiple choice. Visibility is
excluded.

Complete think-tag spans are stripped. Bare letters and recognized “Answer: B”
phrases take a local path. Other answers go to a text extractor. The actual
request does not send gold, despite older system-prompt wording. The online
verdict uses rules; undecidable cases become wrong, not a semantic LLM grade.

| Task/subtype | Default strict rule |
| --- | --- |
| Cognitive map | Exact A/B/C/D |
| Yes/no, including motion change/camera consistency | Exact yes/no |
| Approximate direction / motion heading | Exact full horizontal direction set |
| Source angle / camera rotation angle | Numeric error <=45 degrees, correct turn direction without its opposite |
| 3D localization | Distance error <=2 m and exact horizontal direction set |

Gold “front left” versus prediction “left” is wrong. Gold “45 degrees left”
versus “90 degrees left” passes the inclusive default tolerance. Missing numeric
answers fail.

3D localization does not compute XYZ Euclidean error or grade vertical elevation.
Angles use direct numeric difference, not circular wraparound distance.

`--dir_mode bucket` checks whether the single gold bucket is present; extra or
even opposite directions can pass in the active rules. Report this changed
protocol explicitly.

### CLI options

| Option | Default | Purpose |
| --- | --- | --- |
| `--gold` | OMNIECHO_QA_GOLD or published path | Gold annotations |
| `--predictions` | Required | Indexed predictions; required even for rejudge |
| `--output-dir` | OMNIECHO_QA_OUTPUT or qa_eval/output | Judgments and summary |
| `--files` | OmniEcho | Only this key is registered |
| `--model` | OPENAI_MODEL or OmniEcho | Text extractor, not evaluated checkpoint |
| `--api_key` | OPENAI_API_KEY | Extraction credential |
| `--base_url` | OPENAI_BASE_URL or https://api.example.com/v1 | Default URL is a placeholder |
| `--workers` | 32 | Concurrent workers |
| `--rpm` | 55 | Calls/minute, including outer retries; 0 disables limiter |
| `--angle_tol` | 45 | Degrees |
| `--dist_tol` | 2.0 | Meters |
| `--dir_mode` | strict | strict or bucket |
| `--timeout` | 60 | Seconds per request |
| `--retries` | 8 | Outer attempts per record, backoff plus jitter |
| `--limit` | 0 | Positive N limits eligible online records |
| `--overwrite` | Off | Clear judgments before online judging |
| `--rejudge` | Off | Reuse saved extractions, no API call |
| `--rewrite` | Off | Persist detailed rejudged verdicts |

Online mode requires a key even when individual items can all use local letter
scoring. SDK retries may occur inside the script's outer attempts.
“The answer is B” does not match the current local phrase regex and can require
LLM extraction.

### Metrics, coverage, and cache

Outputs are `eval_OmniEcho_judgments.jsonl` and `eval_qa_summary.json`.
Micro = total correct / total judged. Macro = average accuracy of categories
present. JSON values are fractions. Full coverage should be 2,972, including
600 cognitive-map questions.

Current caveats:

1. Missing/empty predictions and extraction ERROR records are excluded from the
   final online denominator. Check coverage alongside accuracy.
2. Duplicate prediction indices keep the last valid record.
3. Resume invalidates judgments only when the prediction's last line changes.
   It does not compare full answers, gold, model, or scoring configuration.
4. Old judgments outside a newly limited subset may still enter the summary.
   Use fresh output directories for independent experiments.
5. Changed tolerances do not invalidate cached online judgments. Rejudge or
   start fresh.
6. Offline rejudge ignores limit, processes saved lines without online-style
   deduplication, and does not re-extract updated predictions.
7. Offline undecidable rules fall back to saved llm_correct; online undecidable
   rules force wrong. Older judgment files can expose this difference.
8. Without rewrite, rejudge leaves detailed records unchanged but still writes
   a recomputed summary.

### Example

~~~bash
export OMNIECHO_BENCH_ROOT=/path/to/OmniEchoBench
CUDA_VISIBLE_DEVICES=0 python evaluate/infer_qa.py \
  --model-path /path/to/OmniEcho \
  --test-jsonl "$OMNIECHO_BENCH_ROOT/OmniEchoBench-QA/OmniEchoBench-QA.jsonl" \
  --media-root "$OMNIECHO_BENCH_ROOT/OmniEchoBench-QA" \
  --out-jsonl /path/to/results/qa-foa/predictions.jsonl \
  --foa --attn sdpa

export OPENAI_BASE_URL=https://your-extractor.example/v1
export OPENAI_API_KEY=YOUR_API_KEY
export OPENAI_MODEL=YOUR_TEXT_EXTRACTION_MODEL
python qa_eval/eval_qa_metrics.py \
  --gold "$OMNIECHO_BENCH_ROOT/OmniEchoBench-QA/OmniEchoBench-QA.jsonl" \
  --predictions /path/to/results/qa-foa/predictions.jsonl \
  --output-dir /path/to/results/qa-foa/scoring \
  --dir_mode strict --angle_tol 45 --dist_tol 2 \
  --workers 8 --rpm 55
~~~

Replace service placeholders. Online extraction can incur API charges. For a
quick check, add `--limit 2` to both commands and use a separate output folder.
For multi-GPU inference, use `--num-shards N --shard-id I` per GPU with separate
outputs, then merge by original index.

Offline rejudge:

~~~bash
python qa_eval/eval_qa_metrics.py \
  --gold "$OMNIECHO_BENCH_ROOT/OmniEchoBench-QA/OmniEchoBench-QA.jsonl" \
  --predictions /path/to/results/qa-foa/predictions.jsonl \
  --output-dir /path/to/results/qa-foa/scoring \
  --rejudge --angle_tol 30 --dist_tol 2 --dir_mode strict
~~~

## Navigation evaluation

### Service and replan loop

The model service runs in the inference environment; the simulator runs in
external InternNav/Habitat. They exchange image/audio paths, not uploaded media.
Both must see the same absolute paths.

GET /health reports readiness. POST /infer takes content_parts, max_new_tokens,
and optional assistant_prefix, returning trajectory, raw, has_foa, n_foa_pad.
HTTP is threaded but inference is serialized. Navigation always initializes FOA;
there are no no-FOA or mute CLI switches.

At episode reset, match sidecar by scene/episode. Missing sidecars cause STOP.
At each replan, when the action queue is empty:

1. Check replan cap and oracle shortcut.
2. Read position/orientation, select nearest recording by horizontal GLB distance.
3. Transform horizontal FOA channels with `M = B @ inverse(A)` using robot
   forward/left basis B. W and vertical Z stay unchanged.
4. Use the first 10 seconds by default. No receiver interpolation, distance
   loudness correction, or global recording playback clock is used.
5. Select up to eight uniformly sampled historical replan RGB frames plus the
   current frame. These are not necessarily the latest simulator frames.
6. Send RGB, FOA, and sidecar task text. No target coordinates are sent.
   Depth is configured but not included in this agent's model request.
7. Request reasoning, textual meta-actions, and eight [x,y,yaw] points:
   forward meters, left meters, and relative heading radians.
8. Convert numeric points to discrete actions and execute the queue before the
   next model call.

A replan is one model call; a simulator action is one forward/turn/stop step.

### Parsing and actions

The parser strips think tags and extracts numeric triples, preferring the last
region beginning with [[. It does not enforce exactly eight points or valid
complete JSON. Stop-only and malformed output both produce an empty trajectory.
Mixed points/stop tokens retain numeric points. Meta-actions are not executed.
The current prompt does not explicitly teach the stop-token convention.

Between adjacent waypoints:

- Displacement >0.125 m: rounded 0.25 m forward steps.
- Otherwise yaw change >7.5 degrees: rounded 15-degree turns.
- Translation takes precedence over yaw; tiny segments produce no action.

This is not a general waypoint follower. [0,0.25,pi/2] becomes forward movement,
not “turn left then move.” Expected model output must fit this discrete format.

### Stopping

| Condition | audio | audio_nostop |
| --- | --- | --- |
| Horizontal source distance <=1 m at a replan boundary | STOP | Disabled |
| Empty or non-executable trajectory | STOP | STOP |
| Predicted max displacement <0.3 m and max yaw <15 degrees | STOP | STOP |
| 50 replans | STOP | STOP |
| Net movement <0.25 m across 16 simulator actions | Forced STOP | Forced STOP |
| Outer action cap | Exit loop | Exit loop |

The core mode difference is oracle_stop_dist=1.0 versus 0.0.

- Oracle distance is horizontal straight-line distance, not Habitat geodesic
  distance; checked only between queues, so the agent may pass near the goal
  while executing a queue.
- HTTP errors are converted to empty trajectories, hence STOP, without retry.
- The agent does not validate response has_foa.
- After a turn-only replan, the next assistant response is prefixed with
  “I hear the sound coming from in front of me.”
- Two consecutive turn-only replans cause a forced forward step.
- Stuck detection uses net displacement, including spinning/collisions/oscillation.
- The <= loop condition allows 501 actions at max_steps_per_episode=500.
  Exiting the bound does not automatically issue a final STOP.
- YAML has max_episode_steps=5000; the outer bound normally comes first.

audio_nostop disables the oracle shortcut; it does not mean that only an explicit
model stop token can finish the episode.

### Metrics

| Metric | Meaning |
| --- | --- |
| success / SR | Habitat stop-and-distance success, configured distance 1 m |
| spl / SPL | Success weighted by path efficiency |
| oracle_success / OS | Ever entering the success region, per external measure |
| distance_to_goal / NE | Final Habitat distance, normally geodesic |
| trajectory_length / TL | Sum of planar movement between observations |

SPL conventionally averages success * shortest_path / max(actual_path, shortest_path).
Exact measurements are supplied by compatible external Habitat/InternNav.
Agent-logged horizontal distance is not NE.

Shard means are weighted by episode length. Missing shards/nonpositive counts/
missing metric keys fail merging, but total coverage, stale values, and all
nonfinite cases are not fully validated. Check length=900 for a full run.

The default habitat_vlnce_stuck evaluator overrides the base loop. Base-only
OMNIECHO_ONLY_EPISODES filtering, episodes_rank*.jsonl output, and its exception
handling do not apply. Select episodes with a prepared subset. Resume/result
persistence is provided by external InternNav.

## Navigation options

### Model service

| Direct Python option | Default |
| --- | --- |
| --model-path | Required |
| --host | 127.0.0.1 |
| --port | 8210 |
| --dtype | bfloat16 |
| --attn | flash_attention_2 |
| --no-audio-in-video | Off; navigation currently sends separate audio/images |

Wrapper variables: OMNIECHO_MODEL_PATH (required), PYTHON (python),
HOST (127.0.0.1), PORT (8210), ATTENTION (sdpa), CUDA_VISIBLE_DEVICES.
Additional arguments pass through. The wrapper defaults to SDPA; direct Python
defaults to FlashAttention. There is no temperature/top-p service CLI; generation
inherits checkpoint defaults. Per-request max_new_tokens defaults to 512.

### Parallel launcher

~~~bash
bash scripts/eval/run_nav_audio_parallel.sh [N_GPU] [N_SCENES]
~~~

| Option / variable | Default | Meaning |
| --- | --- | --- |
| N_GPU | 2 | Independent shards/workers/prestarted servers |
| N_SCENES | 10 | Smoke scenes; ignored with FULL_SPLIT |
| SRC_SPLIT | case3_vln | Smoke source |
| SMOKE_SPLIT | case3_smoke10 | Generated subset name |
| SMOKE_SEED | 0 | Scene shuffle seed |
| FULL_SPLIT | Unset | Existing split; no smoke subsampling |
| MODES | audio audio_nostop | Both sequentially or one |
| SERVER_BASE | 8210 | Worker i uses base+i |
| READY_TIMEOUT | 2400 | Approximate per-server readiness seconds |
| INTERNNAV_ROOT | Derived | External checkout |
| PY39 | python | Split/merge interpreter |
| TORCHRUN | torchrun | Worker launcher |
| OMNIECHO_NAV_OUT_ROOT | logs/habitat_omniecho_nav | Output root |

Smoke selects random scenes, then the first episode per selected scene.
It does not randomly choose episodes inside scenes. Requesting too many scenes
uses all available; too many shards is rejected.

Whole scenes are distributed round-robin. Workers use independent
torchrun --nproc_per_node=1. CUDA device i and distributed port base 29610 are
hard-coded by the launcher. Services/environments are not started automatically.
FULL_SPLIT=case3_smoke uses the provided 30-episode smoke split.

### Runtime variables and worker

| Variable | Purpose |
| --- | --- |
| OMNIECHO_BENCH_ROOT | Default `<InternNav>/data/OmniEchoBench` in launcher |
| OMNIECHO_NAV_SCENES_DIR | NAV/scene_data/case3_ce |
| OMNIECHO_NAV_EP_DIR | Scenes root /episodes |
| OMNIECHO_NAV_AUDIO_ROOT | NAV dataset root, not /audio child |
| OMNIECHO_NAV_SPLIT | Worker split; launcher sets `split_g<i>` |
| OMNIECHO_NAV_AUDIO_SIDECAR | Explicit override; normally unset for matching shards |
| OMNIECHO_NAV_SERVER_URL | Direct config defaults port 8100; launcher uses 8210+i |
| OMNIECHO_NAV_AUDIO_OUT | Initial config output; worker overrides paths |
| OMNIECHO_NAV_OUT_ROOT | Direct worker defaults logs/habitat_omniecho_nav_all |
| LOCAL_RANK | Simulator GPU rank from torchrun |

Worker CLI accepts only --model audio or --model audio_nostop. Other baselines
are not registered by this entry point.

### Python configuration

Edit active configs in external InternNav, or edit source configs and reinstall.
These are not generic command-line or environment switches.

| model_settings | Shipped value | Effect |
| --- | --- | --- |
| num_history | 8 | Uniformly sampled past frames; 0 removes history |
| num_waypoints | 8 | Prompt request, not parser enforcement |
| exec_waypoints | 8 | Prefix converted/executed before replan |
| max_new_tokens | 512 | Generation cap |
| stop_disp | 0.3 m | Stationary threshold with fixed 15-degree yaw criterion |
| max_replans | 50 | Call cap; raw agent fallback is 10 |
| oracle_stop_dist | 1.0 / 0.0 | Oracle shortcut |
| max_audio_sec | 10.0 | Initial clip; <=0 uses all |
| request_timeout | 600 s | HTTP timeout |
| worker_id / num_workers | 0 / 1 | Internal sharding disabled; do not stack on launcher sharding |
| clean_scratch | False, implicit | Delete previous episode scratch at reset |
| scratch_dir / pred_jsonl | Derived | Overridden by worker |
| server_url / audio_jsonl | Environment-derived | Endpoint and sidecar |

| eval_settings | Shipped value | Effect |
| --- | --- | --- |
| max_steps_per_episode | 500 | <= comparison can execute 501 actions |
| stuck_window | 16 | Simulator-action interval; class fallback 4 |
| stuck_eps | 0.25 m | Net-motion threshold |
| max_episodes | 0 | Unlimited; positive value caps each worker |
| save_video | False | Framework option; stuck loop does not implement video export |
| epoch | 0 | Bookkeeping |
| port / dist_url | 2333 / env:// | External framework settings |

YAML uses RGB/depth 640x480, HFOV 79 degrees, height 0.45 m, forward 0.25 m,
turn/tilt 15 degrees, success distance 1 m. Agent also hard-codes action sizes.
Update both consistently if changing the protocol. Replan cadence, history,
audio length, thresholds, tokens, and controller heuristics affect results.

Helper CLI:
- Splitter: --split (case3_smoke), --n (required), --ep_dir (derived).
- Merger: --split (case3_smoke), --n (required),
  --out_root (logs/habitat_omniecho_nav_all), --model (audio).
- Shards are written beside episode inputs; that directory must be writable.
  Sidecars must cover all episode keys.

## Navigation commands and outputs

Install overlay from OmniEcho root:

~~~bash
python scripts/install_nav_overlay.py /path/to/InternNav --check
python scripts/install_nav_overlay.py /path/to/InternNav
~~~

The installer rejects differing files. --force overwrites without backups.

Start services in separate terminals in the inference environment:

~~~bash
export OMNIECHO_MODEL_PATH=/path/to/OmniEcho
CUDA_VISIBLE_DEVICES=0 PORT=8210 bash evaluate/run_nav_server.sh
~~~

~~~bash
export OMNIECHO_MODEL_PATH=/path/to/OmniEcho
CUDA_VISIBLE_DEVICES=1 PORT=8211 bash evaluate/run_nav_server.sh
~~~

In the Habitat environment:

~~~bash
cd /path/to/InternNav
export OMNIECHO_BENCH_ROOT=/path/to/OmniEchoBench
export OMNIECHO_NAV_OUT_ROOT=/path/to/results/nav-smoke-001
SERVER_BASE=8210 bash scripts/eval/run_nav_audio_parallel.sh 2 10

OMNIECHO_NAV_OUT_ROOT=/path/to/results/nav-full-001 \
FULL_SPLIT=case3_vln MODES=audio_nostop SERVER_BASE=8210 \
  bash scripts/eval/run_nav_audio_parallel.sh 2
~~~

Single worker on a prepared split:

~~~bash
OMNIECHO_NAV_SPLIT=case3_smoke \
OMNIECHO_NAV_SERVER_URL=http://127.0.0.1:8210 \
OMNIECHO_NAV_OUT_ROOT=/path/to/results/nav-single \
  torchrun --nproc_per_node=1 --master_port=29610 \
    scripts/eval/run_nav_eval.py --model audio_nostop
~~~

Direct workers do not automatically run splitter completeness checks.

~~~text
<out_root>/
  <split>_g0/<mode>/
    result.json
    pred_audio.jsonl
    scratch/
  <split>_g1/<mode>/...
  <split>/<mode>/result.json
  _run/<split>/
    run.status
    worker_<mode>_g0.log
    worker_<mode>_g1.log
    merged_<mode>.txt
    split.log
    make_smoke.log                 # generated smoke only
~~~

Predictions are per model call, with episode/replan tags, media paths, prompt,
and raw output. JSON stores fractions for SR/SPL/OS and meters for NE/TL;
merge tables display percentages. Check expected length and worker logs for
request failures, missing audio, malformed outputs, and early stopping.
Use fresh output roots for independent runs to avoid external resume state.

## English-revision verification

Prompts, labels, logs, errors, help, and related English text-matching rules were
updated. Benchmark annotations and evaluation algorithms were preserved.
Generation wording may change with English prompts.

- 49 local tests passed: API, benchmark paths/inventory, navigation geometry and
  split/merge, QA geometry/rendering, English prompt/matching.
- 18 isolated checks confirmed scoring rules, blind extraction, letter parsing,
  trajectory parsing, and action conversion.
- All Python files parsed; scans also covered decoded Python/JSON strings and
  decompressed gzip metadata, with no remaining Chinese text.

CPU checks do not execute the trained checkpoint, remote providers, or a full
Habitat episode. Existing evaluation behaviors above were documented rather
than silently changing the benchmark protocol.
