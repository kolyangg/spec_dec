#!/usr/bin/env bash

set -euo pipefail

RERUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$RERUN_SCRIPT_DIR/../.." && pwd)"

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/lib/env_common.sh"
load_project_env "$ROOT_DIR"

RUN_ID="${RUN_ID:-tuned_latest}"

# Keep rerun outputs isolated even if the user's shell still has homework
# variables such as BENCH_DIR or LOG_DIR exported from manual runbook steps.
# Use RERUN_* variables for intentional overrides.
RUN_DIR="${RERUN_RUN_DIR:-$ROOT_DIR/analysis_runs/$RUN_ID}"
LOG_DIR="${RERUN_LOG_DIR:-$RUN_DIR/logs}"
METRICS_DIR="${RERUN_METRICS_DIR:-$RUN_DIR/metrics}"
BENCH_DIR="${RERUN_BENCH_DIR:-$RUN_DIR/benchmarks/results}"

ensure_run_dirs() {
  mkdir -p "$RUN_DIR" "$LOG_DIR" "$METRICS_DIR" "$BENCH_DIR"
}

activate_named_venv() {
  local name="$1"
  local venv_dir

  case "$name" in
    speculators)
      venv_dir="${SPECULATORS_VENV:-$ROOT_DIR/.venvs/speculators_venv}"
      ;;
    vllm)
      venv_dir="${VLLM_VENV:-$ROOT_DIR/.venvs/vllm_venv}"
      ;;
    comp)
      venv_dir="${COMP_VENV:-$ROOT_DIR/.venvs/comp_venv}"
      ;;
    *)
      echo "Unknown venv name: $name" >&2
      exit 1
      ;;
  esac

  if [[ ! -f "$venv_dir/bin/activate" || ! -x "$venv_dir/bin/python" ]]; then
    echo "Missing venv: $venv_dir" >&2
    echo "Run scripts/setup_all_envs.sh first." >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  source "$venv_dir/bin/activate"
}

wait_for_http() {
  local url="$1"
  local timeout_seconds="${2:-900}"
  local started

  started="$(date +%s)"
  until curl -fsS "$url" >/dev/null 2>&1; do
    if (( "$(date +%s)" - started > timeout_seconds )); then
      echo "Timed out waiting for $url" >&2
      return 1
    fi
    sleep 5
  done
}

print_run_context() {
  ensure_run_dirs
  cat <<EOF
Project root: $ROOT_DIR
Run id:       $RUN_ID
Run dir:      $RUN_DIR
Logs:         $LOG_DIR
Metrics:      $METRICS_DIR
Benchmarks:   $BENCH_DIR
EOF
}
