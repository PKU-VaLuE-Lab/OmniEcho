#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${OMNIECHO_NAV_OUTPUT:?Set OMNIECHO_NAV_OUTPUT to an external output directory}"
: "${OMNIECHO_TRAJECTORY_ROOT:?Set OMNIECHO_TRAJECTORY_ROOT to the parent of OmniEchoBench-Nav}"
mkdir -p "$OMNIECHO_NAV_OUTPUT"
export OMNIECHO_NAV_OUTPUT="$(cd "$OMNIECHO_NAV_OUTPUT" && pwd)"
export OMNIECHO_TRAJECTORY_ROOT="$(cd "$OMNIECHO_TRAJECTORY_ROOT" && pwd)"
PYTHON="${PYTHON:-python}"
PY_RENDER="${PY_RENDER:-$PYTHON}"
PY_VOICE="${PY_VOICE:-$PYTHON}"
PY_ANNOTATE="${PY_ANNOTATE:-$PYTHON}"
STAGES=("$@")
[[ ${#STAGES[@]} -gt 0 ]] || STAGES=(1 2 3 4 5)
[[ ${STAGES[0]} != all ]] || STAGES=(1 2 3 4 5)
for stage in "${STAGES[@]}"; do
  extra=()
  case "$stage" in
    1)
      : "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
      [[ ${SMOKE:-0} != 1 ]] || extra=(--max_scenes 1 --max_episodes_per_scene 2)
      "$PY_ANNOTATE" "$HERE/01_generate_audio_scripts.py" "${extra[@]}"
      ;;
    2)
      [[ ${SMOKE:-0} != 1 ]] || extra=(--max 4)
      "$PY_VOICE" "$HERE/02_batch_tts.py" "${extra[@]}"
      ;;
    3)
      [[ ${SMOKE:-0} != 1 ]] || extra=(--max_samples 4)
      "$PY_RENDER" "$HERE/03_make_spatial_voice_vln.py" --scene all --episodes all "${extra[@]}"
      ;;
    4) "$PY_RENDER" "$HERE/04_convert_to_omni_sft.py" --check_exist ;;
    5)
      : "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
      [[ ${SMOKE:-0} != 1 ]] || extra=(--max-samples 4)
      "$PY_ANNOTATE" "$HERE/05_make_vln_cot_dataset.py" --workers "${WORKERS:-4}" "${extra[@]}"
      ;;
    *) echo "Supported stages: 1 2 3 4 5 or all" >&2; exit 2 ;;
  esac
done
