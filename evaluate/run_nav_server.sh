#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${OMNIECHO_MODEL_PATH:?Set OMNIECHO_MODEL_PATH to an external OmniEcho checkpoint}"
exec "${PYTHON:-python}" -u "$ROOT/evaluate/nav_server.py" \
  --model-path "$OMNIECHO_MODEL_PATH" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8210}" \
  --attn "${ATTENTION:-sdpa}" "$@"
