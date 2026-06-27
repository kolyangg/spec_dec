#!/usr/bin/env bash

set -euo pipefail

project_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
}

load_project_env() {
  local root_dir="$1"
  if [[ -f "$root_dir/.env" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$root_dir/.env"
    set +a
  fi
}

create_venv() {
  local python_bin="$1"
  local venv_dir="$2"

  require_command "$python_bin"
  mkdir -p "$(dirname "$venv_dir")"

  if [[ ! -x "$venv_dir/bin/python" ]]; then
    "$python_bin" -m venv "$venv_dir"
  fi

  # shellcheck disable=SC1091
  source "$venv_dir/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
}

register_kernel_if_requested() {
  local kernel_name="$1"
  local display_name="$2"

  if [[ "${REGISTER_KERNEL:-0}" == "1" ]]; then
    python -m pip install --upgrade ipykernel
    python -m ipykernel install --user --name "$kernel_name" --display-name "$display_name"
  fi
}
