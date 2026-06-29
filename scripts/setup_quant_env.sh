#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/env_common.sh"

ROOT_DIR="$(project_root)"
load_project_env "$ROOT_DIR"
PYTHON_BIN="${COMP_PYTHON_BIN:-${PYTHON_BIN:-python3.12}}"
VENV_ROOT="${VENV_ROOT:-$ROOT_DIR/.venvs}"
VENV_DIR="${COMP_VENV:-$VENV_ROOT/comp_venv}"

echo "Creating quantization environment at $VENV_DIR"
create_venv "$PYTHON_BIN" "$VENV_DIR"

python -m pip install --upgrade \
  "llmcompressor==0.12.0" \
  "accelerate>=1.6.0,<=1.13.0" \
  "transformers>=5.9.0,<=5.10.1" \
  safetensors \
  huggingface_hub
python -m pip check
register_kernel_if_requested comp_venv "LLM Compressor FP8"

echo
echo "Quantization environment is ready."
echo "Activate with: source $VENV_DIR/bin/activate"
