#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

ensure_run_dirs

print_run_context

if compgen -G "$BENCH_DIR/*.json" >/dev/null; then
  python "$SCRIPT_DIR/collect_metrics.py" \
    --bench-dir "$BENCH_DIR" \
    --metrics-dir "$METRICS_DIR" \
    --results-md "$METRICS_DIR/RESULTS_TUNED.md"
else
  echo "No benchmark JSON files found in $BENCH_DIR." >&2
  echo "Run task4_benchmark_sweep.sh successfully before collecting final artifacts." >&2
  exit 1
fi

{
  echo "# Artifact Sizes"
  echo
  echo "## Processed Data"
  du -sh "$ROOT_DIR"/data/processed/* 2>/dev/null || true
  echo
  echo "## Hidden States"
  du -sh "$ROOT_DIR"/data/hidden_states/* 2>/dev/null || true
  echo
  echo "## Models"
  du -sh "$ROOT_DIR"/models/* 2>/dev/null || true
  echo
  echo "## Checkpoints"
  du -sh "$ROOT_DIR"/output/checkpoints/* 2>/dev/null || true
} | tee "$METRICS_DIR/artifact_sizes.txt"

grep -R -E "val/(loss|full_acc|cond_acc)" "$ROOT_DIR/output/logs" "$ROOT_DIR/output/checkpoints" 2>/dev/null \
  | tail -200 \
  | tee "$METRICS_DIR/training_validation_metrics.txt" || true

python - "$ROOT_DIR/spec_dec+quantization_homework.ipynb" <<'PY' 2>&1 | tee "$METRICS_DIR/notebook_todo_check.txt"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(f"Notebook missing: {path}")
    raise SystemExit(0)

nb = json.loads(path.read_text())
todo_cells = []
for i, cell in enumerate(nb.get("cells", [])):
    text = "".join(cell.get("source", []))
    if "TODO" in text:
        todo_cells.append(i)
print("TODO cells:", todo_cells)
PY

cat >"$METRICS_DIR/FINAL_COLLECTION.md" <<EOF
# Final Collection

- Run id: \`$RUN_ID\`
- Run dir: \`$RUN_DIR\`
- Benchmark outputs: \`$BENCH_DIR\`
- Tuned benchmark summary: \`$METRICS_DIR/RESULTS_TUNED.md\`
- Benchmark metrics JSON: \`$METRICS_DIR/task4_metrics.json\`
- Artifact sizes: \`$METRICS_DIR/artifact_sizes.txt\`
- Training validation metrics: \`$METRICS_DIR/training_validation_metrics.txt\`
- Notebook TODO check: \`$METRICS_DIR/notebook_todo_check.txt\`

Paste the best-run text blocks from \`RESULTS_TUNED.md\` into the notebook's benchmark result sections.
EOF

echo "Wrote $METRICS_DIR/FINAL_COLLECTION.md"
