# OmniEchoBench-QA data construction

## What is included

`scripts/generate_batch.py` / `generate_sample.py` generate scene descriptions,
main and source-specific videos, audio stems, and synchronized FOA. Visual
correction uses SAM3 tracking and Depth Anything V2 to update geometry.
`scripts/filter_samples.py` quarantines scene/visibility inconsistencies.
`annotate_qa.py` creates aligned spatial QA with static/camera-rotation crops,
off-screen sources, FOA re-rendering, and synchronized output media.
Only the active generation/annotation path and its helpers are included.

## Setup

Use the QA environment in `../../environments/README.md`. Prepare FFmpeg,
SAM3 and depth auxiliary assets externally. Set the API and asset variables
from `../../configs/api.env.example`; that file is a template, not a live service.

Text annotation calls `POST ${OPENAI_BASE_URL}/chat/completions` with
`model`, `messages`, `max_tokens`, and `temperature`; it reads
`choices[0].message.content`. Credentials come only from environment variables.
No shell profile, credential file, or embedded fallback key is read.

## Video service adapter contract

`omniecho_audio/video_prompt.py` defines a **generic example contract**, not a
promise that all video vendors implement the same API. Adapt this one module
to your provider, or implement the contract in your own gateway:

```text
POST ${VIDEO_BASE_URL}/videos
Authorization: Bearer ${VIDEO_API_KEY}
Body: {model: "OmniEcho", prompt: "...", duration: 6,
       resolution: "720p", aspect_ratio: "16:9"}
Response: {id: "job-id"}

GET ${VIDEO_BASE_URL}/videos/{id}
Response while running: {status: "queued" | "running"}
Response on completion: {status: "completed", url: "https://example.com/video.mp4"}
Response on failure: {status: "failed", error: "..."}
```

The download URL must be usable without forwarding API credentials, for example
a short-lived signed URL. Generated main/sub-videos must contain synchronized
audio: stem extraction depends on it. A silent video provider is insufficient.
`VIDEO_API_KEY` falls back to `OPENAI_API_KEY` when unset. Failed jobs are not
silently substituted with unrelated media.

## Full pipeline

```bash
export OMNIECHO_QA_OUTPUT=/path/to/outputs/OmniEchoBench-QA
TARGET_TOTAL=10 BATCH_SIZE=10 WORKERS=1 MAX_BATCHES=5 \
  bash pipelines/qa/run_pipeline.sh
```

Without overrides the target is 10,000 samples. Start small: video generation
and annotation can incur substantial provider costs. The launcher stops on
zero progress or `MAX_BATCHES` instead of generating indefinitely. Generation
workers use `spawn`; select worker count based on segmentation/depth VRAM.
Every worker can load auxiliary models, so more workers can cause GPU OOM.

Runtime outputs are external: `raw/omniecho_qa_*/`, `quarantine/`, and `aligned/`.
The launcher counts completed sample metadata, not merely a newly written
scene description. Its output directory is for one run; do not concurrently
launch multiple writers into it.

## Annotation-only path

Use existing raw samples without video API calls:

```bash
python pipelines/qa/annotate_qa.py \
  --input /path/to/generated/raw \
  --out /path/to/outputs/OmniEchoBench-QA/aligned \
  --workers 4 --seed 42 --rotating-ratio 0.4666667
```

Each input sample needs `video.mp4`, `corrected_scene.json` (preferred) or
`scene.json`, and the referenced audio stems. Scene objects contain `id`,
`label`, `wav_path`, and timestamped `trajectory` entries with azimuth,
elevation, distance, and optional gain. Geometry uses positive azimuth to
the listener's left; `body_azimuth = world_azimuth - camera_yaw`.

Outputs include `aligned_qa_pairs.jsonl` (task metadata and ground truth),
`aligned_qa_messages.jsonl` (training messages), augmented scene descriptions,
and `merged_av/*.mp4` with spatial audio. QA covers visibility, direction,
motion, 3D localization, and camera rotation. `--sources-target`,
`--max-sources`, `--frac-w`, and `--no-topdown` configure augmentation/rendering.

`--qa-only` recomputes labels against existing rendered media; it skips missing
media and overwrites aggregate JSONL outputs. Use the same complete input list,
seed, and augmentation options. Running it on one sample can replace a whole
batch's aggregate output. Reproducibility depends on the full sample ordering,
not just the per-sample seed. Do not use `--skip-existing` when you need a newly
assembled aggregate file unless you have checked that all desired rows remain.

Operational prompts, generated labels, logs, and command-line help use English.
Direction matching and production-audit checks use the corresponding English
vocabulary. The language conversion preserves the spatial conventions, numerical
thresholds, and output schemas; generated descriptions may differ because the
annotation models now receive English prompts.
