#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/env_common.sh"

ROOT_DIR="$(project_root)"
load_project_env "$ROOT_DIR"
PYTHON_BIN="${SPECULATORS_PYTHON_BIN:-${PYTHON_BIN:-python3.12}}"
VENV_ROOT="${VENV_ROOT:-$ROOT_DIR/.venvs}"
VENV_DIR="${SPECULATORS_VENV:-$VENV_ROOT/speculators_venv}"
EXTERNAL_DIR="${EXTERNAL_DIR:-$ROOT_DIR/external}"
SPECULATORS_REPO="${SPECULATORS_REPO:-https://github.com/vllm-project/speculators.git}"
SPECULATORS_TAG="${SPECULATORS_TAG:-v0.5.0}"

require_command git

echo "Creating Speculators environment at $VENV_DIR"
create_venv "$PYTHON_BIN" "$VENV_DIR"

mkdir -p "$EXTERNAL_DIR"
if [[ ! -d "$EXTERNAL_DIR/speculators/.git" ]]; then
  git clone "$SPECULATORS_REPO" "$EXTERNAL_DIR/speculators"
fi

git -C "$EXTERNAL_DIR/speculators" fetch --tags
git -C "$EXTERNAL_DIR/speculators" checkout "$SPECULATORS_TAG"

python -m pip install --upgrade packaging ninja
python -m pip install -e "$EXTERNAL_DIR/speculators"
python -m pip install --upgrade huggingface_hub datasets
register_kernel_if_requested speculators_venv "Speculators EAGLE-3"

echo
echo "Speculators environment is ready."
echo "Activate with: source $VENV_DIR/bin/activate"
