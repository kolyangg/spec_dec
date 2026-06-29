#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

ensure_run_dirs
activate_named_venv vllm

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
FP8_MODEL="${FP8_MODEL:-$ROOT_DIR/models/Qwen3-8B-FP8-Dynamic}"
DRAFT_MODEL="${DRAFT_MODEL:-}"
if [[ -z "$DRAFT_MODEL" && -f "$METRICS_DIR/task2_best_checkpoint.json" ]]; then
  DRAFT_MODEL="$(python - "$METRICS_DIR/task2_best_checkpoint.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["best"]["checkpoint"])
PY
)"
fi
DRAFT_MODEL="${DRAFT_MODEL:-$ROOT_DIR/output/checkpoints/eagle3_qwen3_8b_3k_seq2048/4}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"
SWEEP_CONCURRENCY="${SWEEP_CONCURRENCY:-8 16 24 32}"
PROMPTS_PER_CONCURRENCY="${PROMPTS_PER_CONCURRENCY:-20}"
BF16_SPEC_TOKENS="${BF16_SPEC_TOKENS:-1 2 3 4}"
FP8_SPEC_TOKENS="${FP8_SPEC_TOKENS:-1 2 3 4}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_BF16_SPEC="${RUN_BF16_SPEC:-1}"
RUN_FP8="${RUN_FP8:-1}"
RUN_FP8_SPEC="${RUN_FP8_SPEC:-1}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
ALLOW_EXISTING_SERVER="${ALLOW_EXISTING_SERVER:-0}"
SERVER_PID=""

print_run_context
echo "Base model:  $BASE_MODEL"
echo "FP8 model:   $FP8_MODEL"
echo "Draft model: $DRAFT_MODEL"

if [[ ! -d "$DRAFT_MODEL" ]]; then
  echo "Missing DRAFT_MODEL directory: $DRAFT_MODEL" >&2
  exit 1
fi

if [[ ! -d "$FP8_MODEL" ]]; then
  echo "Missing FP8_MODEL directory: $FP8_MODEL" >&2
  exit 1
fi

server_url="http://$HOST:$PORT/v1/models"

stop_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "Stopping vLLM server pid $SERVER_PID"
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  SERVER_PID=""
}

trap stop_server EXIT

wait_for_server() {
  local log_file="$1"
  local started
  started="$(date +%s)"
  until curl -fsS "$server_url" >/dev/null 2>&1; do
    if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      echo "vLLM server exited before becoming ready. Last log lines:" >&2
      tail -80 "$log_file" >&2 || true
      exit 1
    fi
    if (( "$(date +%s)" - started > SERVER_READY_TIMEOUT )); then
      echo "Timed out waiting for vLLM server. Last log lines:" >&2
      tail -80 "$log_file" >&2 || true
      exit 1
    fi
    sleep 5
  done
}

start_server() {
  local label="$1"
  local model_path="$2"
  local served_name="$3"
  local spec_config="${4:-}"
  local log_file="$BENCH_DIR/server_${label}.log"
  local cmd=(
    vllm serve "$model_path"
    --host "$HOST"
    --port "$PORT"
    --served-model-name "$served_name"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --generation-config vllm
    --no-enable-prefix-caching
  )

  if [[ -n "$spec_config" ]]; then
    cmd+=(--speculative-config "$spec_config")
  fi

  if curl -fsS "$server_url" >/dev/null 2>&1 && [[ "$ALLOW_EXISTING_SERVER" != "1" ]]; then
    echo "A server is already responding at $server_url." >&2
    echo "Stop it first, or set ALLOW_EXISTING_SERVER=1 if this is intentional." >&2
    exit 1
  fi

  echo
  echo "Starting server: $label"
  echo "Log: $log_file"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" "${cmd[@]}" >"$log_file" 2>&1 &
  SERVER_PID="$!"
  wait_for_server "$log_file"
}

run_bench() {
  local label="$1"
  local model_name="$2"
  local prompts="$3"
  local concurrency="$4"

  echo "Benchmark $label: prompts=$prompts concurrency=$concurrency"
  vllm bench serve \
    --backend openai \
    --host "$HOST" \
    --port "$PORT" \
    --model "$model_name" \
    --dataset-name hf \
    --dataset-path philschmid/mt-bench \
    --num-prompts "$prompts" \
    --max-concurrency "$concurrency" \
    --ignore-eos \
    --seed 0 \
    --save-result \
    --result-dir "$BENCH_DIR" \
    --result-filename "${label}.json" \
    2>&1 | tee "$BENCH_DIR/${label}.txt"
}

warmup() {
  local label="$1"
  local model_name="$2"
  run_bench "${label}_warmup" "$model_name" 16 4
}

run_sweep() {
  local label="$1"
  local model_name="$2"
  local concurrency
  local prompts

  warmup "$label" "$model_name"
  for concurrency in $SWEEP_CONCURRENCY; do
    prompts="$((concurrency * PROMPTS_PER_CONCURRENCY))"
    run_bench "${label}_c${concurrency}" "$model_name" "$prompts" "$concurrency"
  done
}

if [[ "$RUN_BASELINE" == "1" ]]; then
  start_server "baseline_bf16" "$BASE_MODEL" "$BASE_MODEL"
  run_sweep "baseline_bf16" "$BASE_MODEL"
  stop_server
fi

if [[ "$RUN_BF16_SPEC" == "1" ]]; then
  for spec_tokens in $BF16_SPEC_TOKENS; do
    spec_config='{"method":"eagle3","model":"'"$DRAFT_MODEL"'","num_speculative_tokens":'"$spec_tokens"'}'
    start_server "spec_bf16_n${spec_tokens}" "$BASE_MODEL" "$BASE_MODEL" "$spec_config"
    run_sweep "spec_bf16_n${spec_tokens}" "$BASE_MODEL"
    stop_server
  done
fi

if [[ "$RUN_FP8" == "1" ]]; then
  start_server "fp8" "$FP8_MODEL" "$FP8_MODEL"
  run_sweep "fp8" "$FP8_MODEL"
  stop_server
fi

if [[ "$RUN_FP8_SPEC" == "1" ]]; then
  for spec_tokens in $FP8_SPEC_TOKENS; do
    spec_config='{"method":"eagle3","model":"'"$DRAFT_MODEL"'","num_speculative_tokens":'"$spec_tokens"'}'
    start_server "fp8_spec_n${spec_tokens}" "$FP8_MODEL" "$FP8_MODEL" "$spec_config"
    run_sweep "fp8_spec_n${spec_tokens}" "$FP8_MODEL"
    stop_server
  done
fi

python "$SCRIPT_DIR/collect_metrics.py" \
  --bench-dir "$BENCH_DIR" \
  --metrics-dir "$METRICS_DIR" \
  --results-md "$METRICS_DIR/RESULTS_TUNED.md"
