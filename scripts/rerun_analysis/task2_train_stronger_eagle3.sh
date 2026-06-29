#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

ensure_run_dirs
activate_named_venv speculators

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-8}"
LR="${LR:-1e-4}"
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
SPECULATORS_DIR="${SPECULATORS_DIR:-$ROOT_DIR/external/speculators}"
PREPROCESSED_DIR="${PREPROCESSED_DIR:-$ROOT_DIR/data/processed/qwen3_8b_sharegpt_${MAX_SAMPLES}_seq${SEQ_LENGTH}}"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-$ROOT_DIR/data/hidden_states/qwen3_8b_sharegpt_${MAX_SAMPLES}_seq${SEQ_LENGTH}}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-$ROOT_DIR/output/checkpoints/eagle3_qwen3_8b_${MAX_SAMPLES}_seq${SEQ_LENGTH}_e${TRAIN_EPOCHS}_vocab${DRAFT_VOCAB_SIZE}}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-$ROOT_DIR/output/logs/eagle3_qwen3_8b_${MAX_SAMPLES}_seq${SEQ_LENGTH}_e${TRAIN_EPOCHS}_vocab${DRAFT_VOCAB_SIZE}}"
TRAIN_LOG="$LOG_DIR/task2_train.log"

if [[ ! -d "$PREPROCESSED_DIR" ]]; then
  echo "Missing preprocessed data: $PREPROCESSED_DIR" >&2
  echo "Run task1_generate_hidden_states.sh first, or set PREPROCESSED_DIR." >&2
  exit 1
fi

if [[ ! -d "$HIDDEN_STATES_DIR" ]]; then
  echo "Missing hidden states: $HIDDEN_STATES_DIR" >&2
  echo "Run task1_generate_hidden_states.sh first, or set HIDDEN_STATES_DIR." >&2
  exit 1
fi

mkdir -p "$TRAIN_RUN_DIR" "$TRAIN_LOG_DIR"
print_run_context

echo "Training stronger EAGLE-3 draft head"
echo "Preprocessed: $PREPROCESSED_DIR"
echo "Hidden states: $HIDDEN_STATES_DIR"
echo "Checkpoints:  $TRAIN_RUN_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "$SPECULATORS_DIR/scripts/train.py" \
  --verifier-name-or-path "$MODEL_ID" \
  --speculator-type eagle3 \
  --data-path "$PREPROCESSED_DIR" \
  --hidden-states-path "$HIDDEN_STATES_DIR" \
  --save-path "$TRAIN_RUN_DIR" \
  --log-dir "$TRAIN_LOG_DIR" \
  --run-name "eagle3_qwen3_8b_${MAX_SAMPLES}_seq${SEQ_LENGTH}_e${TRAIN_EPOCHS}_vocab${DRAFT_VOCAB_SIZE}" \
  --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
  --epochs "$TRAIN_EPOCHS" \
  --lr "$LR" \
  --total-seq-len "$SEQ_LENGTH" \
  --on-missing raise \
  --save-best \
  --no-resume-from-checkpoint \
  --seed 42 \
  2>&1 | tee "$TRAIN_LOG"

python - "$TRAIN_RUN_DIR" "$METRICS_DIR/task2_best_checkpoint.json" "$METRICS_DIR/task2_best_checkpoint.md" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
json_out = Path(sys.argv[2])
md_out = Path(sys.argv[3])

records = []
for metrics_path in sorted(run_dir.glob("*/val_metrics.json")):
    metrics = json.loads(metrics_path.read_text())
    records.append({
        "checkpoint": str(metrics_path.parent),
        "metrics_path": str(metrics_path),
        **metrics,
    })

if not records:
    raise SystemExit(f"No val_metrics.json files found below {run_dir}")

best = min(records, key=lambda row: row.get("loss_epoch", float("inf")))
json_out.write_text(json.dumps({"best": best, "all": records}, indent=2) + "\n")

lines = [
    "# Task 2 Stronger Training Summary",
    "",
    f"- Best checkpoint: `{best['checkpoint']}`",
    f"- Best validation loss: `{best.get('loss_epoch')}`",
    f"- Position 0 full accuracy: `{best.get('full_acc_0_epoch')}`",
    f"- Position 1 full accuracy: `{best.get('full_acc_1_epoch')}`",
    f"- Position 2 full accuracy: `{best.get('full_acc_2_epoch')}`",
    "",
    "| Checkpoint | loss_epoch | full_acc_0 | full_acc_1 | full_acc_2 |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for row in records:
    lines.append(
        f"| `{row['checkpoint']}` | `{row.get('loss_epoch')}` | "
        f"`{row.get('full_acc_0_epoch')}` | `{row.get('full_acc_1_epoch')}` | "
        f"`{row.get('full_acc_2_epoch')}` |"
    )
md_out.write_text("\n".join(lines) + "\n")
print(f"Best checkpoint: {best['checkpoint']}")
PY
