# Published benchmark data and path contract

## Root layout

`OMNIECHO_BENCH_ROOT` is the local mirror of the OSS `OmniEchoBench/` prefix.
Keep all names, capitalization, scene identifiers, and split identifiers intact:

```text
OmniEchoBench/
  OmniEchoBench-QA/
    OmniEchoBench-QA.jsonl
    videos/vid_0001.mp4 ... vid_0197.mp4
    audio/mc_0001.wav ... mc_0600.wav
    images/mc_0001.png ... mc_0600.png
  OmniEchoBench-NAV/
    audio/case3/scene_003/ ... scene_032/
      sample_001/ ... sample_030/
        rec_pos_*.wav
      source_audio/sample_001.wav ... sample_030.wav
    scene_data/case3_ce/
      case3/scene_003/ ... scene_032/
        scene_NNN.glb
        scene_NNN.navmesh
        scene_NNN.stage_config.json
        scene_NNN_navproxy.glb
        scene_NNN_navproxy.stage_config.json
        nav_meta.json
      episodes/
        case3_vln.json.gz
        case3_vln_audio_sidecar.jsonl
        case3_smoke.json.gz
        case3_smoke_audio_sidecar.jsonl
```

The `reference_photos/` directory markers have no files in the inspected
package. They are not referenced by the evaluation manifests and are not
required by the runtime. Navigation RGB observations are rendered from meshes.

## Inventory verified on October 18, 2026

The audit used a complete `ossutil ls` listing and downloaded all 95 JSON,
JSONL, and gzip metadata files, not the entire media corpus.

| Component | Count |
| --- | ---: |
| Non-directory objects | 14,872 |
| Total object bytes | 208,431,607,935 |
| QA questions | 2,972 |
| Video/spatial QA questions | 2,372 |
| Cognitive-map multiple-choice questions | 600 |
| QA videos | 197 |
| QA standalone audio files | 600 |
| QA images | 600 |
| Full NAV episodes / sidecar rows | 900 / 900 |
| Smoke NAV episodes / sidecar rows | 30 / 30 |
| NAV scenes | 30 |
| Unique NAV recorded audio files | 12,390 |
| NAV original source audio files | 900 |
| NAV scene meshes / proxy meshes / navmeshes | 30 / 30 / 30 |
| NAV stage configurations / navigation metadata | 60 / 30 |

All 14,872 files are covered by the manifests and associated asset checks.
There were no missing required objects, empty required files, duplicate
episode/sidecar keys, unmatched episode/sidecar keys, or unreferenced QA media.
The smoke split reuses recordings from the full split; do not add its 413
recordings to the unique full-corpus audio count.

This verifies completeness relative to the published manifests, not against an
independent release manifest. It does not prove every media payload decodes,
validate full-file checksums, or replace a Habitat/model evaluation run.

Additional range-read checks covered all 60 GLB headers and JSON chunks: none
reference missing external buffers or images (their resources are embedded).
Audio header samples from all 30 NAV scenes and two QA recordings were all
four-channel, 48 kHz, 24-bit PCM. Two QA PNG headers were valid. One downloaded
QA video was successfully inspected with FFprobe and contains H.264 video and
four-channel 48 kHz AAC audio. These are structural checks, not full decoding
of all media.

## Runtime paths

```bash
export OMNIECHO_BENCH_ROOT=/path/to/OmniEchoBench
```

- QA JSONL: `$OMNIECHO_BENCH_ROOT/OmniEchoBench-QA/OmniEchoBench-QA.jsonl`.
- QA media root: `$OMNIECHO_BENCH_ROOT/OmniEchoBench-QA`. No `media/` layer.
- NAV scenes root: `$OMNIECHO_BENCH_ROOT/OmniEchoBench-NAV/scene_data/case3_ce`.
- NAV episodes root: the scenes root followed by `/episodes`.
- NAV audio root: `$OMNIECHO_BENCH_ROOT/OmniEchoBench-NAV`, not `/audio`.
- NAV full split: `case3_vln`; provided smoke split: `case3_smoke`.

QA inference `--test-jsonl` and scoring `--gold` default to the published QA
JSONL under `OMNIECHO_BENCH_ROOT`; `OMNIECHO_QA_GOLD` overrides that default.
Without `--media-root`, inference resolves media relative to the JSONL directory.

The NAV launcher, Python configurations, splitter, and evaluator use these
defaults. `OMNIECHO_NAV_SCENES_DIR`, `OMNIECHO_NAV_EP_DIR`, and
`OMNIECHO_NAV_AUDIO_ROOT` remain available for explicit overrides.
When installing the overlay into InternNav, reinstall the updated helper
`internnav/benchmark_paths.py` as well as the scripts and agent.

The published sidecars contain absolute audio paths from an older filesystem.
The runtime preserves the suffix `audio/case3/<scene>/<sample>/rec_pos_*.wav`
and resolves it under the local NAV audio root. This remapping does not edit
the sidecars or OSS objects. Legacy `source_glb` and `proxy_glb` strings in
`nav_meta.json` are provenance fields; evaluation loads meshes through episode
`scene_id` and stage-config-relative assets, not through those old strings.

Training-data construction uses independent `OMNIECHO_TRAJECTORY_ROOT`,
`OMNIECHO_NAV_OUTPUT`, and `OMNIECHO_QA_OUTPUT` directories. The test package
does not contain trajectory parquet files, generation model weights, or raw
QA generation scenes/stems. Those are not missing test assets.

## Repeat the checks

For a complete local mirror:

```bash
python scripts/check_benchmark_data.py \
  --root "$OMNIECHO_BENCH_ROOT" \
  --report /path/to/reports/benchmark_data.json
```

To verify OSS objects without downloading approximately 208 GB of media,
set `OSS_BENCH_PREFIX` to the published `oss://bucket/.../OmniEchoBench` prefix
and use an external audit directory:

```bash
export AUDIT_DIR=/path/to/audit
mkdir -p "$AUDIT_DIR"
ossutil ls "$OSS_BENCH_PREFIX/" > "$AUDIT_DIR/oss-list.txt"
ossutil cp "$OSS_BENCH_PREFIX/" "$AUDIT_DIR/metadata/" -r \
  --include '*.json' --include '*.jsonl' --include '*.json.gz'
python scripts/check_benchmark_data.py \
  --root "$AUDIT_DIR/metadata" \
  --oss-list "$AUDIT_DIR/oss-list.txt" \
  --oss-prefix "$OSS_BENCH_PREFIX/" \
  --report "$AUDIT_DIR/benchmark_data.json"
```

Do not use `--limited-num` or short-format output for the inventory. The
checker returns a nonzero exit code for missing/empty assets or invalid
required metadata. The report records legacy audio-path counts so that path
compatibility is distinguishable from missing files.
