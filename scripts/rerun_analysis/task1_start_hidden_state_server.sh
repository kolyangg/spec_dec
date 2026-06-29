#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

ensure_run_dirs
activate_named_venv vllm

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
SPECULATORS_DIR="${SPECULATORS_DIR:-$ROOT_DIR/external/speculators}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"
SERVER_LOG="$LOG_DIR/task1_hidden_state_server.log"

if [[ ! -f "$SPECULATORS_DIR/scripts/launch_vllm.py" ]]; then
  echo "Missing Speculators launch script: $SPECULATORS_DIR/scripts/launch_vllm.py" >&2
  exit 1
fi

unset VLLM_ENDPOINT
print_run_context
echo "Starting hidden-state server for $MODEL_ID on $HOST:$PORT"
echo "Log: $SERVER_LOG"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" python "$SPECULATORS_DIR/scripts/launch_vllm.py" \
  "$MODEL_ID" \
  -- \
  --host "$HOST" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  2>&1 | tee "$SERVER_LOG"
