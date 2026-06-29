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
SPECULATORS_DIR="${SPECULATORS_DIR:-$ROOT_DIR/external/speculators}"
PREPROCESSED_DIR="${PREPROCESSED_DIR:-$ROOT_DIR/data/processed/qwen3_8b_sharegpt_${MAX_SAMPLES}_seq${SEQ_LENGTH}}"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-$ROOT_DIR/data/hidden_states/qwen3_8b_sharegpt_${MAX_SAMPLES}_seq${SEQ_LENGTH}}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:8000/v1}"
GEN_CONCURRENCY="${GEN_CONCURRENCY:-32}"
RUN_SMOKE="${RUN_SMOKE:-1}"
VALIDATE_FULL="${VALIDATE_FULL:-0}"

PREP_LOG="$LOG_DIR/task1_prepare_data.log"
SMOKE_LOG="$LOG_DIR/task1_hidden_states_smoke.log"
FULL_LOG="$LOG_DIR/task1_hidden_states_full.log"
SMOKE_OUTPUT="$RUN_DIR/task1_smoke_hidden_states"

if [[ ! -f "$SPECULATORS_DIR/scripts/prepare_data.py" ]]; then
  echo "Missing Speculators scripts in $SPECULATORS_DIR" >&2
  exit 1
fi

print_run_context
echo "Waiting for hidden-state endpoint: $VLLM_ENDPOINT/models"
wait_for_http "$VLLM_ENDPOINT/models" 900

echo "Preprocessing $MAX_SAMPLES samples at seq length $SEQ_LENGTH"
python "$SPECULATORS_DIR/scripts/prepare_data.py" \
  --model "$MODEL_ID" \
  --data sharegpt \
  --output "$PREPROCESSED_DIR" \
  --max-samples "$MAX_SAMPLES" \
  --seq-length "$SEQ_LENGTH" \
  2>&1 | tee "$PREP_LOG"

if [[ "$RUN_SMOKE" == "1" ]]; then
  rm -rf "$SMOKE_OUTPUT"
  mkdir -p "$SMOKE_OUTPUT"
  echo "Running hidden-state smoke test into $SMOKE_OUTPUT"
  python "$SPECULATORS_DIR/scripts/data_generation_offline.py" \
    --model "$MODEL_ID" \
    --preprocessed-data "$PREPROCESSED_DIR" \
    --endpoint "$VLLM_ENDPOINT" \
    --output "$SMOKE_OUTPUT" \
    --max-samples 8 \
    --concurrency 4 \
    --validate-outputs \
    --fail-on-error \
    2>&1 | tee "$SMOKE_LOG"
fi

mkdir -p "$HIDDEN_STATES_DIR"
echo "Generating full hidden-state cache into $HIDDEN_STATES_DIR"
full_cmd=(
  python "$SPECULATORS_DIR/scripts/data_generation_offline.py"
  --model "$MODEL_ID"
  --preprocessed-data "$PREPROCESSED_DIR"
  --endpoint "$VLLM_ENDPOINT"
  --output "$HIDDEN_STATES_DIR"
  --max-samples "$MAX_SAMPLES"
  --concurrency "$GEN_CONCURRENCY"
  --max-consecutive-errors 32
)

if [[ "$VALIDATE_FULL" == "1" ]]; then
  full_cmd+=(--validate-outputs)
fi

"${full_cmd[@]}" 2>&1 | tee "$FULL_LOG"

{
  echo "MODEL_ID=$MODEL_ID"
  echo "MAX_SAMPLES=$MAX_SAMPLES"
  echo "SEQ_LENGTH=$SEQ_LENGTH"
  echo "PREPROCESSED_DIR=$PREPROCESSED_DIR"
  echo "HIDDEN_STATES_DIR=$HIDDEN_STATES_DIR"
  echo "hidden_state_files=$(find "$HIDDEN_STATES_DIR" -maxdepth 1 -name 'hs_*.safetensors' | wc -l)"
  du -sh "$PREPROCESSED_DIR" "$HIDDEN_STATES_DIR"
} | tee "$METRICS_DIR/task1_data_summary.txt"
