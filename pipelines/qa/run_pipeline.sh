#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
: "${OMNIECHO_QA_OUTPUT:?Set OMNIECHO_QA_OUTPUT to an external output directory}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
TARGET_TOTAL="${TARGET_TOTAL:-10000}"
BATCH_SIZE="${BATCH_SIZE:-100}"
WORKERS="${WORKERS:-4}"
MAX_BATCHES="${MAX_BATCHES:-1000}"
for value in "$TARGET_TOTAL" "$BATCH_SIZE" "$WORKERS" "$MAX_BATCHES"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Counts must be positive integers" >&2; exit 2; }
done
mkdir -p "$OMNIECHO_QA_OUTPUT"
OUTPUT="$(cd "$OMNIECHO_QA_OUTPUT" && pwd)"
INPUT="$OUTPUT/raw"
mkdir -p "$INPUT"
count_samples() { find "$INPUT" -mindepth 2 -maxdepth 2 -name metadata.json -type f | wc -l; }
for ((batch=1; batch<=MAX_BATCHES; batch++)); do
  current="$(count_samples)"
  (( current < TARGET_TOTAL )) || break
  start_id="$($PYTHON - "$INPUT" <<'PY'
from pathlib import Path
import sys
identifiers = [int(path.name.rsplit('_', 1)[-1]) for path in Path(sys.argv[1]).glob('omniecho_qa_*')
               if path.is_dir() and path.name.rsplit('_', 1)[-1].isdigit()]
print(max(identifiers, default=0) + 1)
PY
)"
  echo "Generating batch $batch: valid=$current target=$TARGET_TOTAL"
  "$PYTHON" "$HERE/scripts/generate_batch.py" --num_samples "$BATCH_SIZE" \
    --workers "$WORKERS" --start_id "$start_id" --output_root "$INPUT"
  "$PYTHON" "$HERE/scripts/filter_samples.py" --input "$INPUT" --quarantine "$OUTPUT/quarantine"
  updated="$(count_samples)"
  (( updated > current )) || { echo "No progress; inspect batch_report.json and API/model configuration" >&2; exit 1; }
done
(( $(count_samples) >= TARGET_TOTAL )) || { echo "MAX_BATCHES reached before target" >&2; exit 1; }
exec "$PYTHON" "$HERE/annotate_qa.py" --input "$INPUT" --out "$OUTPUT/aligned" \
  --workers "$WORKERS" --seed "${SEED:-42}" "$@"
