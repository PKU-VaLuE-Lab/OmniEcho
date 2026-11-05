# OmniEchoBench-Nav data construction

## Inputs and boundaries

This pipeline consumes existing trajectory poses and rendered RGB observations.
It constructs audio-guided training samples; it does **not** generate arbitrary
Habitat trajectories from scene meshes. Preparing those trajectories is a
separate upstream asset-generation step.

`OMNIECHO_TRAJECTORY_ROOT` is a training-trajectory root, not
`OMNIECHO_BENCH_ROOT`. The published test package uses `OmniEchoBench-NAV` with
`audio/` and `scene_data/case3_ce/`; it does not contain these training parquet
files or rendered trajectory observations. Do not point this construction
pipeline at the test package. Test-path conventions are in
`../../docs/benchmark_data.md`.

```text
${OMNIECHO_TRAJECTORY_ROOT}/
  OmniEchoBench-Nav/<scene>/
    data/chunk-000/episode_000000.parquet
    meta/episodes.jsonl
    videos/chunk-000/observation.images.rgb.125cm_0deg/episode_000000_000000.jpg
```

Keep the original scene and episode IDs. `geometry.load_episode` specifies the
required parquet columns and uses camera poses at `125cm_0deg` and
`125cm_30deg`. The public dataset directory is `OmniEchoBench-Nav`; arrange
external trajectories under that name. RGB image paths in generated SFT are
resolved against `OMNIECHO_TRAJECTORY_ROOT`, not the output directory.

## Main stages

| Stage | Entry point | Output / requirements |
| --- | --- | --- |
| 1 | `01_generate_audio_scripts.py` | `audio_scripts.jsonl`; image-capable annotation API |
| 2 | `02_batch_tts.py` | `audio_scripts_with_audio.jsonl`, mono speech/SFX; GPU auxiliary models |
| 3 | `03_make_spatial_voice_vln.py` | `spatial_voice_vln_full.jsonl`, 4-channel FOA; CPU geometry |
| 4 | `04_convert_to_omni_sft.py` | `spatial_voice_vln_omni_sft.jsonl`; checks referenced media |
| 5 | `05_make_vln_cot_dataset.py` | `spatial_voice_vln_omni_sft_cot.jsonl`; visual reasoning annotation |

```bash
export OMNIECHO_TRAJECTORY_ROOT=/path/to/trajectories
export OMNIECHO_NAV_OUTPUT=/path/to/outputs/OmniEchoBench-Nav
export OPENAI_BASE_URL=https://api.example.com/v1
export OPENAI_API_KEY=YOUR_API_KEY
export OPENAI_MODEL=OmniEcho
export OMNIECHO_TTS_MODEL=/path/to/tts-model
export OMNIECHO_SOUND_SOURCE=/path/to/stable-audio-source
export OMNIECHO_SOUND_MODEL=/path/to/sound-model
export PY_RENDER=/path/to/nav-data-env/bin/python
export PY_ANNOTATE=/path/to/nav-data-env/bin/python
export PY_VOICE=/path/to/voice-env/bin/python
SMOKE=1 bash pipelines/nav/run_pipeline.sh all
```

Run `bash pipelines/nav/run_pipeline.sh 3 4` to render and convert with no API
calls when stages 1/2 are already available. `SMOKE=1` limits scale but still
requires valid trajectories, auxiliary models, and any selected API stages.
Output files are created only when the pipeline runs, outside this distribution.
Individual scripts expose `--help`, resume, and stage-specific shard controls.

## Spatial conventions

The sound source is the final front-camera trajectory position. A receiver uses
the current camera pose. FOA channels are `[W,Y,Z,X]`: omnidirectional, left,
up, front. The direct-path renderer applies distance attenuation proportional
to `0.3 / max(distance, 0.3)`. It is **free-field**, not a reverberant room
simulation. It does not require mesh acoustics or Habitat for stages 3/4.
Waypoints are relative `[x_forward_m, y_left_m, yaw_rad]` coordinates, normally
the next eight points. Pose conversion and interpolation live in `geometry.py`.

## Optional supervision

- `03b_make_spatial_voice_vln_noise.py`: add spatial noise/distractor augmentation.
- `06_cot_to_text_instruction.py`: extract text-instruction supervision.
- `07_make_stop_vln_cot_dataset.py`: generate within-1m stop-token examples.
- `08_align_stop_prompt_and_merge.py`: align the stop contract and merge examples.

These are complementary tasks, not archived versions. They are not automatically
run by the five-stage launcher. Pass explicit input/output arguments when
combining alternative supervision sets, and inspect each `--help` before use.
The annotation service must support image inputs. Template fallback, where
available, is opt-in and should not be mixed into scored data without disclosure.

## Environments

See `../../environments/README.md`: stage 2 needs the voice GPU environment;
stages 1/3/4/5 use the navigation-data environment. The separate InternNav +
Habitat environment is needed for **closed-loop benchmark evaluation**, not for
encoding FOA against precomputed trajectories. No weights, scenes, trajectory
parquet files, audio, images, or generated QA/SFT records are included.
