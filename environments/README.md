# Environment setup

Use separate environments. Do not install all requirement files together.
Versions below are derived from the supplied local source/environment metadata;
they are not a claim that arbitrary current upstream checkouts are compatible.
No environment directories or auxiliary weights are distributed.

## System prerequisites

Linux, Conda (or Python virtual environments), FFmpeg/ffprobe, libsndfile,
and an NVIDIA driver compatible with your chosen Torch/CUDA wheel are required.
Rendering needs OpenGL/EGL support for Habitat's headless GPU simulator.

```bash
sudo apt-get install ffmpeg libsndfile1 libegl1 libgl1
```

## Model inference / navigation server

```bash
conda create -n omniecho-infer python=3.11 -y
conda activate omniecho-infer
python -m pip install -r environments/inference.txt
```

Select the Torch CUDA wheel for your driver if overriding the default index.
Start with `--attn sdpa`, which does not require FlashAttention. The source
inference stack uses Transformers 5.2.0; the Habitat stack uses 4.51.0 and must
not overwrite it. The upstream model classes and `qwen-omni-utils` are required
by OmniEcho's checkpoint format and multimodal preprocessing.

Model VRAM depends on checkpoint size, dtype, and video frame limits. No GPU
checkpoint load is part of the code-only validation. Test one sample before
starting a large batch. Video sampling can be configured with
`OMNIECHO_VIDEO_FPS`, `OMNIECHO_VIDEO_MAX_FRAMES`, and `OMNIECHO_VIDEO_MAX_PIXELS`.

## InternNav / Habitat simulator workers

The inspected working simulator environment contains:

| Component | Version |
| --- | --- |
| Python | 3.9.23 |
| Habitat-Sim | 0.2.4, headless + Bullet |
| Habitat-Lab / Habitat-Baselines | 0.2.4 |
| Torch | 2.8.0 |
| Transformers | 4.51.0 |
| NumPy | 1.26.4 |
| SciPy | 1.13.1 |

**Preferred:** use an existing compatible InternNav environment and install
`environments/nav-eval.txt` there. For a clean rebuild, prepare the matching
upstream sources externally:

```bash
conda create -n omniecho-nav python=3.9 -y
conda activate omniecho-nav
conda install habitat-sim=0.2.4 withbullet headless -c conda-forge -c aihabitat -y
git clone --branch v0.2.4 --depth 1 https://github.com/facebookresearch/habitat-lab.git /path/to/habitat-lab
python -m pip install -e /path/to/habitat-lab/habitat-lab
python -m pip install -e /path/to/habitat-lab/habitat-baselines
python -m pip install torch==2.8.0 transformers==4.51.0
```

Use a **compatible InternNav source checkout**, initialize its required
submodules, and follow its dependency setup. The overlay relies on
`internnav.agent.base.Agent`, `internnav.evaluator.DistributedEvaluator`, the
Habitat environment implementation, and `internnav.habitat_extensions.vln.measures`.
It does not vendor the whole framework. The supplied InternNav requirements can
contain Python >=3.10 packages despite the inspected runtime being 3.9; blindly
installing all current extras into 3.9 is not a reproducible environment recipe.
Reuse a validated environment or resolve those upstream constraints explicitly.

```bash
python -m pip install -e /path/to/InternNav --no-deps
python -m pip install -r environments/nav-eval.txt
python -c "import habitat, habitat_sim, habitat_baselines, internnav"
```

`--no-deps` assumes the compatible framework's dependencies have already been
installed; it is not a substitute for InternNav setup. The source-only overlay
installer does not change dependencies or copy weights/data. See `nav/README.md`.

## QA rendering and visual correction

```bash
conda create -n omniecho-qa python=3.11 -y
conda activate omniecho-qa
python -m pip install torch
python -m pip install -r environments/qa-render.txt
```

The inspected QA environment uses Python 3.11, NumPy 1.26.4, spaudiopy 0.2.0,
and Ultralytics 8.4.78. The segmentation code requires that release's
`SAM3SemanticPredictor` and video tracking APIs, not a generic older YOLO install.
Prepare SAM3 weights externally and set `OMNIECHO_SEGMENTATION_MODEL`.

Clone Depth Anything V2 outside OmniEcho, install its requirements in this
environment, and set `OMNIECHO_DEPTH_SOURCE` to that source checkout and
`OMNIECHO_DEPTH_MODEL` to its ViT-L checkpoint. Other encoders require changing
`ValidationConfig.depth_encoder` consistently. The model source and weights
are deliberately not included.

The main path uses generated-video audio stems, so a separate commercial
sound-retrieval platform is not required. Optional room-acoustics paths in
`omniecho_audio/render.py` need `pyroomacoustics`; the direct FOA path does not.
The QA renderer emits N3D/ACN FOA. The navigation free-field renderer uses its
original `[W,Y,Z,X]` directional convention. Inference preserves raw scaling;
do not silently normalize or downmix spatial inputs when preparing checkpoints.

## Navigation rendering / API annotation

```bash
conda create -n omniecho-nav-data python=3.11 -y
conda activate omniecho-nav-data
python -m pip install -r environments/nav-render.txt
```

Stages 1/5 use an image-capable chat-completions endpoint. Stages 3/4 are local
CPU geometry, FOA encoding, and SFT conversion and do not require Habitat or GPUs
when trajectory poses and RGB frames have already been rendered.

## Navigation speech and sound synthesis

```bash
conda create -n omniecho-voice python=3.11 -y
conda activate omniecho-voice
python -m pip install -r environments/voice.txt
```

`02_batch_tts.py` uses the public `qwen_tts.Qwen3TTSModel` API for speech and
`stable_audio_3.StableAudioModel` / `loading_utils.load_diffusion_cond` for sound.
Install the auxiliary sound model's own dependencies from its external source
checkout. Set `OMNIECHO_TTS_MODEL`, `OMNIECHO_SOUND_SOURCE`, and
`OMNIECHO_SOUND_MODEL`. The sound-model directory must include its configuration,
checkpoint, and required text encoder assets. These auxiliary models retain
their actual dependency identities; relabeling their Python imports breaks them.
Synthesis is offline by default: fully prepare auxiliary assets before running.

The Nav launcher accepts `PY_ANNOTATE`, `PY_RENDER`, and `PY_VOICE` as executable
paths so stages can run in different environments without nested activation.
