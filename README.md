<div align="center">

# 🔊 OmniEcho

### Audio-Visual Spatial Understanding for Omni-Modal Embodied Agents

<p align="center">
  <a href="https://www.pku.edu.cn/"><img src="docs/assets/pku-logo.png" alt="Peking University" width="210" height="77"></a>
  &nbsp;&nbsp;
  <a href="https://qwen.ai/"><img src="docs/assets/qwen-logo.png" alt="Qwen" width="210" height="77"></a>
  &nbsp;&nbsp;
  <a href="https://www.tsinghua.edu.cn/"><img src="docs/assets/tsinghua-logo.png" alt="Tsinghua University" width="210" height="77"></a>
</p>

*Can your embodied agent understand **where** a sound comes from — and navigate to it?*

<p>
  <a href="https://arxiv.org/html/2609.23407v1">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2609.23407-b31b1b.svg?style=for-the-badge&logo=arxiv&logoColor=white">
  </a>
  <a href="https://huggingface.co/datasets/PKU-VaLuE-Lab/OmniEchoBench">
    <img alt="Hugging Face · OmniEchoBench" src="https://img.shields.io/badge/Hugging%20Face-OmniEchoBench-FFD21E.svg?style=for-the-badge&amp;logo=huggingface&amp;logoColor=FFD21E">
  </a>
  <a href="#code-models">
    <img alt="Code" src="https://img.shields.io/badge/Code-Available-2ea44f.svg?style=for-the-badge&logo=github&logoColor=white">
  </a>
  <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en">
    <img alt="Data license: CC BY-NC-SA 4.0" src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-2ea44f.svg?style=for-the-badge">
  </a>
</p>

**[Project homepage](https://pku-value-lab.github.io/OmniEcho/)** ·
**[Leaderboard](https://pku-value-lab.github.io/OmniEcho/leaderboard/)** ·

**First-order ambisonics (FOA) spatial audio + vision + language, in real-world embodied settings** — a unified benchmark for spatial audio-visual perception **and** sound-guided navigation, plus a spatially aware omni-modal model.

</div>

The homepage links target the planned GitHub Pages deployment at
`PKU-VaLuE-Lab/OmniEcho`; see [WEBSITE.md](WEBSITE.md) for deployment and local preview.

---

## Contents

- [Overview](#overview) · [Benchmark](#benchmark) · [Headline results](#headline-results)
- [Release status and data](#release-status) · [Code and models](#code-models)
- [Environments](#environments) · [API configuration](#api-configuration)
- [QA inference and scoring](#qa-inference-and-scoring) · [Navigation evaluation](#navigation-evaluation)
- [Training-data construction](#training-data-construction) · [Validation](#validation)
- [Detailed evaluation guide](EVALUATION.md)
- [Citation](#citation) · [Acknowledgements](#acknowledgements) · [License](#license)

---

<a id="news"></a>

## 📰 News

- **`2026-09`** 🎉 Our paper **[OmniEcho: Audio-Visual Spatial Understanding for Omni-Modal Embodied Agents](https://arxiv.org/html/2609.23407v1)** is released on arXiv!
- **`2026-09`** 🌟 **OmniEchoBench** is cited by the **[Qwen3.8-Omni Technical Report](https://arxiv.org/pdf/2609.25611)** as a benchmark for spatial audio understanding.

---

<a id="overview"></a>

## 🧭 Overview

Humans effortlessly localize the direction of a sound source and fuse it with what they see to reason and act — an ability that remains hard for embodied agents. **OmniEcho** studies **spatial audio understanding for omni-modal embodied agents**: given first-order ambisonics (FOA) spatial audio together with visual observations and, when applicable, language, the agent must reason about its surroundings to **answer spatial questions** or **navigate toward a sounding target**.

This project contributes:

- 🎯 **OmniEchoBench** — a unified, real-world benchmark for **spatial audio-visual perception (QA)** and **audio-vision-language navigation (Nav)**, with human-verified annotations.
- 🛠️ **A controllable spatial-audio rendering pipeline** that keeps geometry consistent across sound sources, visual observations, and agent trajectories — enabling scalable training supervision.
- 🤖 **OmniEcho** — a spatially aware omni-modal model built on Qwen3-Omni, introducing a dedicated **FOA spatial encoder** alongside the pretrained semantic audio pathway.

> **TL;DR** — OmniEcho combines a dedicated FOA spatial encoder with vision and language for spatial QA and sound-guided navigation. It achieves **28.5%** overall Audio & Vision QA accuracy; fine-grained localization and distance estimation remain open challenges.

---

<a id="benchmark"></a>

## 📊 Benchmark

**OmniEchoBench** spans **six tasks** over **197 real-world audio-visual videos**, **2,972 QA pairs**, and **900 navigation samples** with FOA audio collected from **30 real-world environments**. It is split into two complementary tracks.

### 🎧 OmniEchoBench-QA — spatial audio-visual perception

All data are recorded from **real human performances**, deliberately emphasizing sound sources that are **off-screen** or that **cross in and out of the field of view**, so that vision alone is insufficient. Each take is captured as a synchronized first-person main video, a 360° panorama (annotation-only), and a 4-channel FOA (ACN/SN3D) track.

| Track | Task | # QA | What it probes |
|:--|:--|--:|:--|
| **Temporal reasoning** | Source Direction | 1,069 | Bearing of on/off-screen sources over time |
| *(2,372 QA over videos)* | 3D Localization | 521 | Metric 3D position of the source |
| | Source Motion | 473 | World-frame heading & motion changes |
| | Camera Rotation | 309 | Viewpoint change from audio + vision |
| **Cognitive Map** | Bird's-eye Localization | 600 | Match heard direction to a top-down map (A/B/C/D) |
| | **Total** | **2,972** | |

### 🧭 OmniEchoBench-Nav — sound-guided navigation

A **real-world indoor** navigation benchmark that jointly supports **spatial-audio** and **language-instruction** navigation over the same scenes and targets. At test time the agent receives the FOA recorded at the receiver nearest its current pose (directionally remapped to its orientation), must infer the target's location, and navigate to it (**success = within 1 m**).

| Property | Value |
|:--|:--|
| Real-scanned scenes | **30** (10 single-room · 20 multi-room, 2–4 connected rooms) |
| Navigation samples | **900** (30 per scene) |
| FOA receiver recordings | **12,390** unique recordings — 4-channel, 48 kHz |
| Original source audio files | **900**, as listed in the published asset inventory |
| Source types | Sound events **489** · Speech **411**, as labeled in the bundled navigation sidecar |
| Scene format | Textured mesh (`.glb`) + top-down floor render; Habitat-based closed-loop eval |

Each source is annotated with its device, room, semantic description, and a natural-language navigation target.

The episode, task-type, and receiver-recording counts above follow the bundled
navigation sidecar. See the [published asset inventory](docs/benchmark_data.md)
for the complete media inventory and verification procedure.

---

<a id="headline-results"></a>

## 📈 Headline Results

**Spatial audio-visual QA** (Audio & Vision accuracy, %) — results from the
[bundled leaderboard data](docs/assets/leaderboard-data.json), updated **2026-10-18**:

| Model | Camera Rotation | Source Direction | 3D Localization | Source Motion | Cognitive Map | Overall |
|:--|--:|--:|--:|--:|--:|--:|
| Qwen3.8-omni-flash | 74.1 | 47.9 | 28.6 | 37.4 | 39.5 | 43.9 |
| Gemini3.8-flash | 53.7 | 34.1 | 22.1 | 31.3 | 26.0 | 32.0 |
| Gemini3.7-flash | 45.3 | 32.8 | 24.4 | 33.4 | 26.2 | 31.4 |
| muse-spark-1.2 | 55.0 | 34.2 | 19.6 | 29.6 | 25.2 | 31.3 |
| **OmniEcho (Ours)** | **41.7** | **22.6** | **14.2** | **24.7** | **47.5** | **28.5** |
| doubao-seed-2-0-lite-260428 | 34.3 | 30.6 | 22.1 | 29.6 | 25.3 | 28.3 |
| Qwen3.5-Omni-Plus | 52.4 | 16.1 | 19.6 | 22.2 | 25.2 | 23.3 |
| Qwen3-Omni-30B-A3B | 32.7 | 16.3 | 7.7 | 20.9 | 22.3 | 18.5 |
| Qwen2.5-Omni-7B | 24.9 | 8.0 | 5.4 | 14.0 | 20.0 | 12.6 |
| SO-7B | 21.4 | 5.2 | 0.0 | 17.1 | 22.8 | 11.4 |


**Sound-guided navigation** on OmniEchoBench-Nav (Val-Unseen) — audio guidance is harder than text, yet OmniEcho is competitive with text-guided VLN:

| Model | Guidance | SR ↑ | SPL ↑ | NE ↓ |
|:--|:--|--:|--:|--:|
| Seq2Seq | Text | 11.3 | 9.4 | 4.14 |
| CMA | Text | 9.8 | 8.5 | 4.34 |
| InternVLA-N1 | Text | 17.8 | 16.7 | 2.87 |
| SoundSpaces | Binaural audio | 5.4 | 3.9 | 5.32 |
| **OmniEcho** | **FOA audio** | **16.2** | **11.5** | 4.09 |

See the [paper](https://arxiv.org/html/2609.23407v1) for the full tables, ablations,
and analyses, and the [leaderboard](https://pku-value-lab.github.io/OmniEcho/leaderboard/)
for the project’s Audio & Vision comparison.

---

<a id="release-status"></a>

## 📦 Release Status & Data

This repository contains the implementation and project website for **OmniEcho**,
**OmniEchoBench-QA**, and **OmniEchoBench-Nav**. Inference, evaluation, QA video/audio
generation and aligned annotation, and navigation training-data construction are
included. Model checkpoints, benchmark media, generated outputs, and caches are
distributed separately or excluded from this source package.

| Component | Status in this source package | Location |
|:--|:--|:--|
| 🎧 **OmniEchoBench-QA** annotations | ✅ Included: **2,972** QA pairs; media separate | [QA annotations](data/OmniEchoBench-QA/OmniEchoBench-QA.jsonl) |
| 🧭 **OmniEchoBench-Nav** indices | ✅ Included: **900** episodes and matching audio sidecars; media separate | [Navigation indices](data/OmniEchoBench-NAV/scene_data/case3_ce/episodes/) |
| 🛠️ Training-data generation and rendering code | ✅ Included | [QA pipeline](pipelines/qa/README.md) · [Navigation pipeline](pipelines/nav/README.md) |
| 🧪 QA inference and evaluation code | ✅ Included | [Inference](evaluate/infer_qa.py) · [Scoring](qa_eval/eval_qa_metrics.py) |
| 🤖 Navigation inference and evaluation code | ✅ Included | [Model service](evaluate/nav_server.py) · [InternNav overlay](nav/README.md) |
| 🌐 Project website and leaderboard | ✅ Included | [Website](docs/index.html) · [Leaderboard](docs/leaderboard/) · [Setup](WEBSITE.md) |
| 📥 Benchmark videos, FOA audio, and scene assets | ✅ Included | [Data layout and verification](docs/benchmark_data.md) |

Website images and fonts are bundled under `docs/assets/`; the demo video is
streamed from its hosted URL. Keep external assets and generated outputs outside
this source directory.

### Bundled annotation layout

```text
data/
├── OmniEchoBench-QA/
│   └── OmniEchoBench-QA.jsonl           # 2,972 QA pairs
└── OmniEchoBench-NAV/
    └── scene_data/case3_ce/
        └── episodes/
            ├── vln.json.gz             # 900 Habitat navigation episodes
            └── vln_audio_sidecar.jsonl # Per-episode FOA recording index
```

**QA format.** Each line is a chat-style sample with a `messages` list. Media is
referenced by relative path: `videos/*.mp4` for the **2,372** video samples with
muxed four-channel FOA, and a top-down image plus a separate four-channel FOA
`audio/*.wav` for the **600** cognitive-map samples. Spatial QA rows carry typed
`meta` fields such as `category`, `subtype`, `coordinate_frame`, and `answer_bucket`;
cognitive-map questions are scored against the assistant reference letter.

**Nav format.** Habitat episodes are paired with an audio *sidecar* mapping
receiver poses to FOA `.wav` files. The published sidecar's `audio/case3/...`
suffix is resolved under the local `OmniEchoBench-NAV` directory, including
legacy absolute paths that contain this suffix. The bundled `vln.*` snapshot
instead references `audio/OmniEchoBench-Nav/...` and older mesh paths. It needs
matching legacy assets or an explicit path conversion; simply renaming it to
`case3_vln` does not make it compatible with the published media layout.

### Published benchmark layout

The external benchmark uses the following runtime layout. Its full navigation
split is named `case3_vln`, and its 30-episode smoke split is named `case3_smoke`.
The bundled `vln.*` annotation files above retain their original names and do
not include the media or the separately published split files.

```text
OmniEchoBench/                         # OMNIECHO_BENCH_ROOT
├── OmniEchoBench-QA/
│   ├── OmniEchoBench-QA.jsonl
│   ├── videos/
│   ├── audio/
│   └── images/
└── OmniEchoBench-NAV/
    ├── audio/case3/<scene>/
    │   ├── <sample>/rec_pos_*.wav
    │   └── source_audio/<sample>.wav
    └── scene_data/case3_ce/
        ├── case3/<scene>/             # Meshes, navmeshes, configs, and metadata
        └── episodes/
            ├── case3_vln.json.gz
            ├── case3_vln_audio_sidecar.jsonl
            ├── case3_smoke.json.gz
            └── case3_smoke_audio_sidecar.jsonl
```

Set `OMNIECHO_BENCH_ROOT` to the local mirror of the complete `OmniEchoBench/`
prefix. Preserve names and capitalization, including uppercase `OmniEchoBench-NAV`.
There is no intermediate `media/` directory inside `OmniEchoBench-QA`.
See [docs/benchmark_data.md](docs/benchmark_data.md) for the complete inventory,
path handling, and verification commands.

> 📥 **Media download.** Annotations and episode indices are included, but the
> referenced videos, FOA recordings, and scene meshes must be obtained separately.
> Public media download links remain to be added; the evaluation code is already
> included in this repository.

---

<a id="code-models"></a>

## 🧪 Code & Models

Inference and evaluation code for **both** tracks is included:

- **OmniEchoBench-QA** — model inference, LLM answer extraction, and deterministic
  rule-based scoring: strict two-axis direction, angle/distance tolerances, and
  exact-letter multiple choice.
- **OmniEchoBench-Nav** — a Habitat agent that navigates to a sound source from
  per-step FOA, driven by an OmniEcho `/infer` server through InternNav.


### Code layout

| Path | Purpose |
| --- | --- |
| `data/` | Bundled QA annotations and legacy-named navigation indices |
| [docs/benchmark_data.md](docs/benchmark_data.md) | Published benchmark layout and verification guide |
| `docs/index.html` | Project homepage, with an online demo |
| `docs/leaderboard/` | Audio & Vision benchmark leaderboard |
| `scripts/build_leaderboard.py` | Regenerate the leaderboard HTML, JSON, and CSV |
| `.github/workflows/pages.yml` | Publish `docs/` through GitHub Pages after migration |
| `evaluate/infer_qa.py` | Load OmniEcho and generate indexed QA predictions |
| `evaluate/nav_server.py` | Model-side navigation HTTP service |
| `omniecho/foa_features.py` | Four-channel FOA feature extraction |
| `qa_eval/eval_qa_metrics.py` | Answer extraction and rule-based QA scoring |
| `nav/` | InternNav overlay, Habitat configs, sharding, and metric merging |
| `pipelines/qa/` | Generate videos/stems, correct geometry, render FOA, annotate QA |
| `pipelines/nav/` | Generate audio scripts, speech/sound, FOA navigation SFT and CoT |
| `environments/` | Separate dependency sets; not one combined environment |
| `configs/api.env.example` | Placeholder-only API and external asset configuration |
| `tests/` | Geometry, rendering, API, and release regression checks |

<a id="environments"></a>

## Environments

The commands below use Bash on Linux. System prerequisites include FFmpeg/ffprobe,
libsndfile, a compatible NVIDIA driver for GPU inference, and OpenGL/EGL for
Habitat rendering.

Read [environments/README.md](environments/README.md) before installation. There are distinct environments:

1. **Model inference**: Python 3.11, Torch 2.9.1, Transformers 5.2.0.
2. **Habitat navigation evaluation**: Python 3.9, Habitat-Sim/Lab/Baselines 0.2.4,
   InternNav, NumPy 1.26.4, Transformers 4.51.0. This runs the simulator/client,
   not OmniEcho model weights.
3. **QA rendering**: Python 3.11, FFmpeg, spatial audio tools, SAM3 and Depth
   Anything V2 for visual geometry correction. Auxiliary weights are external.
4. **Navigation data rendering/annotation**: Python 3.11, NumPy/SciPy/PyArrow,
   an image-capable annotation API. Speech/sound synthesis uses a separate GPU
   environment and external auxiliary models.


<a id="api-configuration"></a>


## QA inference and scoring

See [EVALUATION.md](EVALUATION.md) for every inference/scoring option, actual
scoring rules, denominator behavior, and current implementation limitations.

Run from the repository root in the inference environment:

```bash
export OMNIECHO_BENCH_ROOT=/path/to/OmniEchoBench
python evaluate/infer_qa.py \
  --model-path /path/to/OmniEcho \
  --test-jsonl "$OMNIECHO_BENCH_ROOT/OmniEchoBench-QA/OmniEchoBench-QA.jsonl" \
  --media-root "$OMNIECHO_BENCH_ROOT/OmniEchoBench-QA" \
  --out-jsonl /path/to/results/predictions.jsonl \
  --foa --attn sdpa

export OPENAI_BASE_URL=https://api.example.com/v1
export OPENAI_API_KEY=YOUR_API_KEY
export OPENAI_MODEL=OmniEcho
python qa_eval/eval_qa_metrics.py \
  --gold "$OMNIECHO_BENCH_ROOT/OmniEchoBench-QA/OmniEchoBench-QA.jsonl" \
  --predictions /path/to/results/predictions.jsonl \
  --output-dir /path/to/results/qa
```

The checkpoint must contain the trained `thinker.foa_encoder.*` and
`thinker.foa_projector.*` tensors, not just the upstream base model. Both sharded
and single-file Safetensors layouts are supported. Missing FOA parameters fail
loading rather than producing predictions from a randomly initialized encoder.
`--foa` explicitly enables the spatial branch; `--no-foa` disables it.
Multi-GPU batch inference uses `--num-shards N --shard-id I`, one process/output
per GPU; preserve original `index` fields when combining outputs.

The input JSONL uses a `messages` list with user `video`/`audio`/`image`/`text`
parts and an assistant reference. Scoring matches the prediction `index` to the
zero-based gold record index. Gold `meta.category`, `meta.subtype`, and
`meta.answer_bucket` drive spatial metrics; cognitive-map items use the reference
letter. Feed the benchmark annotation schema to scoring, not a bare training
messages file with its task metadata removed.

Preserve the published `OmniEchoBench-QA/{videos,audio,images}` layout: there is
no intermediate `media/` directory. The NAV directory is case-sensitive
`OmniEchoBench-NAV`, and its published splits are `case3_vln` and `case3_smoke`.
See [docs/benchmark_data.md](docs/benchmark_data.md) for the complete layout, inventory, and verification
commands. `OMNIECHO_BENCH_ROOT` refers to the local mirror of the OSS benchmark
prefix, not to either dataset subdirectory or a training-output directory.

Use `--limit 2` for an initial inference/scoring check. The scorer can spend money
on an external API; `--rpm`, `--workers`, and `--limit` bound request throughput.
`--rejudge` recomputes rules from existing extracted judgments without API calls.

<a id="navigation-evaluation"></a>

## Navigation evaluation

See [EVALUATION.md](EVALUATION.md) for the closed-loop behavior, stopping rules,
configuration options, and metric aggregation details.

See [nav/README.md](nav/README.md) for the full two-environment setup. In brief:

1. Install the overlay into a compatible external InternNav checkout.
2. Download the benchmark preserving its relative paths and set `OMNIECHO_BENCH_ROOT`.
3. Start one `evaluate/run_nav_server.sh` process per GPU in the inference environment.
4. Run `nav/scripts/eval/run_nav_audio_parallel.sh` from InternNav's Habitat environment.

The servers and simulator workers must see the **same absolute media and scratch
paths**. The service exchanges paths, not uploaded media. The local server defaults
to loopback and has no authentication; do not expose it to an untrusted network.
Report oracle-stop (`audio`) and policy-stop (`audio_nostop`) results separately.

<a id="training-data-construction"></a>

## Training-data construction

**QA:** [pipelines/qa/README.md](pipelines/qa/README.md) describes raw video/audio generation, visual
metadata correction, FOA rendering, view/camera augmentation, and aligned QA.

```bash
export OMNIECHO_QA_OUTPUT=/path/to/outputs/OmniEchoBench-QA
TARGET_TOTAL=10 BATCH_SIZE=10 WORKERS=1 bash pipelines/qa/run_pipeline.sh
```

**Navigation:** [pipelines/nav/README.md](pipelines/nav/README.md) describes the five main stages and optional
noise, text-instruction, and stop-token supervision. Existing trajectory parquet
files and rendered RGB frames are prerequisites; this pipeline is not a general
scene-mesh-to-trajectory renderer.

```bash
export OMNIECHO_TRAJECTORY_ROOT=/path/to/trajectories
export OMNIECHO_NAV_OUTPUT=/path/to/outputs/OmniEchoBench-Nav
SMOKE=1 bash pipelines/nav/run_pipeline.sh all
```

The model training engine and cluster job submission scripts are outside this
distribution's inference-and-data-rendering scope. Training-ready messages are
produced, but a distributed training launch is not claimed to be bundled.

<a id="validation"></a>

## Validation

```bash
python scripts/check_release.py
python -m pytest -p no:cacheprovider tests/test_api.py tests/test_nav_split.py
python -m pytest -p no:cacheprovider tests/test_qa_render.py tests/test_qa_geometry.py
```

The release checker targets a code-only file set; this source package also
contains website assets and benchmark indices that its current allowlist does
not cover. Review those asset-type findings separately.

Run rendering tests in the QA environment. Tests create synthetic inputs in
temporary directories; no test datasets or weights are bundled. Syntax/static
checks and CPU tests do not certify GPU checkpoint loading, remote providers,
or a full Habitat episode. End-to-end execution requires your external assets.

---

<a id="citation"></a>

## 📚 Citation

If you find **OmniEcho** or **OmniEchoBench** useful, please cite:

```bibtex
@misc{liu2026omniechoaudiovisualspatialunderstanding,
      title={OmniEcho: Audio-Visual Spatial Understanding for Omni-Modal Embodied Agents},
      author={Ruixun Liu and Yuxuan Wang and Jiacheng Xie and Yuhuan You and Donghua Cai and Junming Lin and Xiong-Hui Chen and Zhifang Guo and Yunfei Chu and Qize Yang and Xize Cheng and Jin Xu and Yiwu Zhong},
      year={2026},
      eprint={2609.23407},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2609.23407},
}
```

---

<a id="acknowledgements"></a>

## 🙏 Acknowledgements

OmniEcho is built on top of [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni), and its navigation evaluation builds on [InternNav](https://internrobotics.github.io/user_guide/internnav/) and the [Habitat](https://github.com/facebookresearch/habitat-sim) simulator. The training data pipeline draws on VLN corpora including R2R, RxR, VLN-PE, and ScaleVLN. We thank the authors of these projects.

<a id="license"></a>

## 📄 License

The **OmniEcho** data and code is released for **non-commercial research use** under
[**CC BY-NC-SA 4.0**](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en).

