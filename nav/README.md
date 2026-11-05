# OmniEchoBench-Nav evaluation

This directory is a **code-only InternNav overlay**. It includes the OmniEcho
agent, both the base Habitat evaluator and its stuck-detection subclass,
configuration files, scene-based splitting, and complete-result merging.
It does not contain InternNav itself, Habitat, scenes, episodes, recordings,
checkpoints, or result files. The model service is provided by `../evaluate/`.

## Two environments

- **Inference environment:** runs the OmniEcho checkpoint service with Python
  3.11 / Transformers 5.2.0, one service per GPU.
- **Simulation environment:** runs the InternNav client and Habitat-Sim/Lab/
  Baselines 0.2.4 under Python 3.9. The inspected runtime uses Torch 2.8.0,
  Transformers 4.51.0, and NumPy 1.26.4.

Follow `../environments/README.md`; installing the model requirements inside the
Habitat environment is not supported. A current arbitrary InternNav checkout
is not guaranteed to implement the same interfaces as the supplied source.

## Install the overlay

From OmniEcho's root:

```bash
python scripts/install_nav_overlay.py /path/to/InternNav --check
python scripts/install_nav_overlay.py /path/to/InternNav
```

The installer checks required framework files and refuses differing existing
files. Review conflicts before using `--force`; that option overwrites only
listed overlay code/config files and creates no historical backups. It never
copies data, checkpoints, caches, or a `.git` directory. Merely checking the
layout is not a full dependency/import test.

The overlay requires the existing InternNav Agent/Evaluator registries,
DistributedEvaluator, Habitat environment implementation, and VLN sensor/measure
extensions. Its own `habitat_vlnce_evaluator.py` is included because the
stuck evaluator depends on it; it is not assumed to exist upstream.

## External benchmark assets

Download the published benchmark without renaming directories or splits:

```text
${OMNIECHO_BENCH_ROOT}/OmniEchoBench-NAV/
  audio/case3/<scene>/<sample>/rec_pos_*.wav
  audio/case3/<scene>/source_audio/<sample>.wav
  scene_data/case3_ce/
    case3/<scene>/<scene>.glb
    case3/<scene>/<scene>.navmesh
    case3/<scene>/<scene>.stage_config.json
    case3/<scene>/<scene>_navproxy.glb
    case3/<scene>/<scene>_navproxy.stage_config.json
    case3/<scene>/nav_meta.json
    episodes/case3_vln.json.gz
    episodes/case3_vln_audio_sidecar.jsonl
    episodes/case3_smoke.json.gz
    episodes/case3_smoke_audio_sidecar.jsonl
```

The directory name is `OmniEchoBench-NAV` (uppercase `NAV`), not the descriptive
benchmark name `OmniEchoBench-Nav`. The full split has 900 episodes across 30
scenes; the provided smoke split has 30 episodes. Required metadata comprises:

- A Habitat episode file `<split>.json.gz` containing `episodes` and optional
  `instruction_vocab`, with scene IDs, start poses, goal positions and instructions.
- The corresponding `<split>_audio_sidecar.jsonl`. Each record has `scene_id`,
  integer-compatible `episode_id`, and the geometry/audio fields expected by
  `internnav/agent/spatial_audio_agent.py`.
- Matching textured meshes, navmeshes/stage configs, and four-channel FOA WAVs.

Set these in the simulation environment:

```bash
export OMNIECHO_BENCH_ROOT=/path/to/OmniEchoBench
export OMNIECHO_NAV_SCENES_DIR="$OMNIECHO_BENCH_ROOT/OmniEchoBench-NAV/scene_data/case3_ce"
export OMNIECHO_NAV_EP_DIR="$OMNIECHO_NAV_SCENES_DIR/episodes"
export OMNIECHO_NAV_AUDIO_ROOT="$OMNIECHO_BENCH_ROOT/OmniEchoBench-NAV"
export OMNIECHO_NAV_OUT_ROOT=/path/to/results/OmniEchoBench-Nav
```

Only `OMNIECHO_BENCH_ROOT` is necessary when using the published layout; the
three NAV paths above are derived automatically unless explicitly overridden.
Its default is `<InternNav>/data/OmniEchoBench`.

Relative sidecar `audio_path` values such as `audio/case3/...` resolve against
`OMNIECHO_NAV_AUDIO_ROOT`, which defaults to the NAV dataset directory (not its
`audio/` subdirectory). The published sidecars contain legacy absolute paths:
the agent rebases their `audio/case3/...` suffix onto this local root without
modifying the downloaded sidecars or relying on the old machine's filesystem.
Scene IDs such as `case3/scene_003/scene_003.glb` resolve against
`OMNIECHO_NAV_SCENES_DIR`, the `scene_data/case3_ce` directory.
Each sidecar must match all episode `(scene, episode_id)` pairs. Splitting
rejects missing sidecars and more shards than scenes instead of silently
producing empty workers. Shards are written beside the input episode files,
so that external episodes directory must be writable.

## Start model services

From OmniEcho's root in the inference environment:

```bash
export OMNIECHO_MODEL_PATH=/path/to/OmniEcho
CUDA_VISIBLE_DEVICES=0 PORT=8210 bash evaluate/run_nav_server.sh
```

For two GPUs start another process in a separate terminal:

```bash
CUDA_VISIBLE_DEVICES=1 PORT=8211 bash evaluate/run_nav_server.sh
```

Default attention is SDPA and default bind address is `127.0.0.1`.
The local service has no authentication. Use a private authenticated gateway
rather than exposing it directly. Both environments must see the same absolute
paths: the worker writes image/FOA scratch files and sends their paths to the
server. The server does not accept uploaded file bodies through this contract.

```text
GET /health
  -> {status: "ok", model: "OmniEcho"}
POST /infer
  <- {content_parts: [{type: "text" | "image" | "audio", value: "..."}],
      max_new_tokens: 512, assistant_prefix: "..."}
  -> {trajectory: [[x_forward_m, y_left_m, yaw_rad], ...],
      raw: "...", has_foa: true, n_foa_pad: 0}
```

`assistant_prefix` is optional. `<stop>`-only output yields an empty trajectory.
No API gateway specific to a cloud vendor is needed for model inference.

## Launch evaluation

Activate the Habitat environment and change to the external InternNav checkout:

```bash
cd /path/to/InternNav
SRC_SPLIT=case3_vln SERVER_BASE=8210 \
  bash scripts/eval/run_nav_audio_parallel.sh 2 10
```

This selects one episode from each of up to 10 scenes, shards by scene, waits
for two already-running services, and runs both stop modes. Use no more GPUs
than the number of selected scenes. The launcher does not automatically start
model services or switch Python environments.

For a full split and policy stopping only:

```bash
FULL_SPLIT=case3_vln MODES=audio_nostop SERVER_BASE=8210 \
  bash scripts/eval/run_nav_audio_parallel.sh 2
```

`PY39` and `TORCHRUN` override simulator-side executables. `READY_TIMEOUT`
controls the wait for model loading. Each worker uses its GPU's service;
server and simulator memory coexist on that GPU, so leave room for rendering.

## Metrics and stop conditions

- `audio`: oracle stop at distance <= 1 m.
- `audio_nostop`: policy-issued stop; no oracle distance check.
- Both modes retain the evaluator's stuck-stop safeguard (16-step window,
  0.25 m net displacement threshold) and a 500-step safety cap.

These are different protocols; report the selected one rather than combining
scores. Results contain success, SPL, oracle success, distance to goal,
trajectory length, and episode count. Per-shard means are weighted by episode
count during merging. Missing/empty shard results cause an error instead of
being reported as a complete benchmark run; failed workers propagate a nonzero
launcher exit code. Failed episodes within a worker must also be checked in its
logs and expected episode counts before publication.

Outputs are `${OMNIECHO_NAV_OUT_ROOT}/<split>/<mode>/result.json` and
`_run/<split>/merged_<mode>.txt` with worker logs. All are runtime outputs,
not files shipped with this code-only distribution.
