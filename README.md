<div align="center">

# 🔊 OmniEcho

### Audio-Visual Spatial Understanding for Omni-Modal Embodied Agents

*Can your embodied agent hear **where** things are — and act on it?*

<p>
  <a href="https://arxiv.org/html/2609.23407v1">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2609.23407-b31b1b.svg?style=for-the-badge&logo=arxiv&logoColor=white">
  </a>
  <a href="#-benchmark">
    <img alt="Benchmark" src="https://img.shields.io/badge/OmniEchoBench-QA%20%2B%20Nav-4c8bf5.svg?style=for-the-badge&logo=databricks&logoColor=white">
  </a>
  <a href="#-code--models">
    <img alt="Code" src="https://img.shields.io/badge/Code%20%26%20Models-Coming%20Soon-f59e0b.svg?style=for-the-badge&logo=github&logoColor=white">
  </a>
  <a href="#-license">
    <img alt="License" src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-2ea44f.svg?style=for-the-badge">
  </a>
</p>

**First-order ambisonics (FOA) spatial audio + vision + language, in real-world embodied settings** — a unified benchmark for spatial audio-visual perception **and** sound-guided navigation, plus a spatially aware omni-modal model.

</div>

---

## 📰 News

- **`2026-09`** 🎉 Our paper **[OmniEcho: Audio-Visual Spatial Understanding for Omni-Modal Embodied Agents](https://arxiv.org/html/2609.23407v1)** is released on arXiv!
- **`2026-09`** 🌟 **OmniEchoBench** is cited by the **[Qwen3.8-Omni Technical Report](https://arxiv.org/pdf/2609.25611)** as a benchmark for spatial audio understanding.
- **`2026-09`** 📦 The **OmniEchoBench** benchmark data (QA + Navigation) is now open-sourced in this repository. Inference & evaluation code — *coming soon*.

---

## 🧭 Overview

Humans effortlessly localize the direction of a sound source and fuse it with what they see to reason and act — an ability that remains hard for embodied agents. **OmniEcho** studies **spatial audio understanding for omni-modal embodied agents**: given first-order ambisonics (FOA) spatial audio together with visual observations and, when applicable, language, the agent must reason about its surroundings to **answer spatial questions** or **navigate toward a sounding target**.

This project contributes:

- 🎯 **OmniEchoBench** — a unified, real-world benchmark for **spatial audio-visual perception (QA)** and **audio-vision-language navigation (Nav)**, with human-verified annotations.
- 🛠️ **A controllable spatial-audio rendering pipeline** that keeps geometry consistent across sound sources, visual observations, and agent trajectories — enabling scalable training supervision.
- 🤖 **OmniEcho** — a spatially aware omni-modal model built on Qwen3-Omni, introducing a dedicated **FOA spatial encoder** alongside the pretrained semantic audio pathway.

> **TL;DR** — OmniEcho reaches **state-of-the-art** spatial audio-visual perception, and its sound-guided navigation approaches the level of traditional *text-guided* VLN, while highlighting fine-grained localization and distance estimation as open challenges.

---

## 📊 Benchmark

**OmniEchoBench** spans **six tasks** over **197 real-world audio-visual scenes**, **2,972 QA pairs**, and **900 navigation samples** with FOA audio collected from **30 real-world environments**. It is split into two complementary tracks.

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
| FOA recordings | **12,900** — 4-channel, 48 kHz — a spatially dense acoustic field |
| Source types | Non-speech events **505** (alarm, doorbell, footsteps, …) · Spoken commands **395** |
| Scene format | Textured mesh (`.glb`) + top-down floor render; Habitat-based closed-loop eval |

Each source is annotated with its device, room, semantic description, and a natural-language navigation target.

---

## 📈 Headline Results

**Spatial audio-visual QA** (overall accuracy, %) — OmniEcho leads in both settings:

| Model | Base | Setting | Overall |
|:--|:--|:--|--:|
| Qwen2.5-Omni-7B | Qwen2.5-Omni-7B | A-foa + V | 12.6 |
| SO-7B | Qwen2.5-Omni-7B | A-foa + V | 11.4 |
| Qwen3-Omni-30B-A3B | Qwen3-Omni-30B-A3B | A-single + V | 18.5 |
| **OmniEcho (Ours)** | Qwen3-Omni-30B-A3B | **A-foa + V** | **28.5** |

**Sound-guided navigation** on OmniEchoBench-Nav (Val-Unseen) — audio guidance is harder than text, yet OmniEcho is competitive with text-guided VLN:

| Model | Guidance | SR ↑ | SPL ↑ | NE ↓ |
|:--|:--|--:|--:|--:|
| Seq2Seq | Text | 11.3 | 9.4 | 4.14 |
| CMA | Text | 9.8 | 8.5 | 4.34 |
| InternVLA-N1 | Text | 17.8 | 16.7 | 2.87 |
| SoundSpaces | Binaural audio | 5.4 | 3.9 | 5.32 |
| **OmniEcho** | **FOA audio** | **16.2** | **11.5** | 4.09 |

See the [paper](https://arxiv.org/html/2609.23407v1) for the full tables, ablations, and analyses.

---

## 📦 What's Open-Sourced

| Component | Status | Location |
|:--|:--|:--|
| 🎧 **OmniEchoBench-QA** data | 🕓 **Coming soon** | — |
| 🧭 **OmniEchoBench-Nav** data | 🕓 **Coming soon** | — |
| 🧪 QA inference & evaluation code | 🕓 **Coming soon** | — |
| 🤖 Navigation inference & evaluation code | 🕓 **Coming soon** | — |


### Data layout

```
data/
├── OmniEchoBench-QA/
│   └── OmniEchoBench-QA.jsonl          # 2,972 QA pairs (messages + typed meta / gold answers)
└── OmniEchoBench-NAV/
    └── scene_data/case3_ce/
        └── episodes/
            ├── vln.json.gz             # Habitat navigation episodes (900 tasks)
            └── vln_audio_sidecar.jsonl # per-episode FOA recording index (pose → .wav)
```

**QA format.** Each line is a chat-style sample. Media is referenced by **relative path** (`videos/*.mp4` for the 2,372 video+muxed-4ch-FOA rows; a top-down image + separate 4-channel FOA `.wav` for the 600 cognitive-map rows). Every row carries a typed `meta` block (`category`, `subtype`, `coordinate_frame`, gold `answer_bucket`, …) so answers can be scored deterministically.

**Nav format.** Habitat episodes plus an audio *sidecar* mapping each receiver pose to its FOA `.wav`. The large media assets (videos, FOA `.wav` recordings, `.glb` meshes) are hosted separately — download links will be provided alongside the evaluation code release.

> 📥 **Media download.** The `.jsonl` / episode indices in this repo reference media by relative path. Hosted download links for the referenced videos, FOA `.wav` files, and scene meshes are **coming soon** with the code release.

---

## 🧪 Code & Models

Inference and evaluation code for **both** tracks is **coming soon**:

- **OmniEchoBench-QA** — LLM answer-extraction + deterministic rule-based scorer (strict two-axis direction, ±tolerance angle/distance, exact-letter MC).
- **OmniEchoBench-Nav** — a Habitat agent that navigates to a sound source from per-step FOA, driven by an OmniEcho `/infer` server (InternNav + Habitat closed-loop).

⭐ **Star / watch this repo** to be notified when the code and model checkpoints land.

---

## 📚 Citation

If you find **OmniEcho** or **OmniEchoBench** useful, please cite:

```bibtex

```

---

## 🙏 Acknowledgements

OmniEcho is built on top of [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni), and its navigation evaluation builds on [InternNav](https://internrobotics.github.io/user_guide/internnav/) and the [Habitat](https://github.com/facebookresearch/habitat-sim) simulator. The training data pipeline draws on VLN corpora including R2R, RxR, VLN-PE, and ScaleVLN. We thank the authors of these projects.

## 📄 License

The **OmniEchoBench** data is released for **non-commercial research use** under **CC BY-NC 4.0**. Code and model licenses will accompany their respective releases.

<div align="center">

**[📄 Paper](https://arxiv.org/html/2609.23407v1)** · **[🎧 QA Data](data/OmniEchoBench-QA/)** · **[🧭 Nav Data](data/OmniEchoBench-NAV/)** · **Code — coming soon**

*Toward embodied agents that reason about **what** they perceive and **where** it originates.*

</div>
