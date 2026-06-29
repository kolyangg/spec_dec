#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/env_common.sh"

ROOT_DIR="$(project_root)"
load_project_env "$ROOT_DIR"
PYTHON_BIN="${VLLM_PYTHON_BIN:-${PYTHON_BIN:-python3.12}}"
VENV_ROOT="${VENV_ROOT:-$ROOT_DIR/.venvs}"
VENV_DIR="${VLLM_VENV:-$VENV_ROOT/vllm_venv}"

echo "Creating vLLM environment at $VENV_DIR"
create_venv "$PYTHON_BIN" "$VENV_DIR"

python -m pip install "vllm==0.20.0" "fastapi<0.137"
python -m pip install --upgrade huggingface_hub datasets
python - <<'PY'
import sys
import sysconfig
from pathlib import Path

header = Path(sysconfig.get_path("include")) / "Python.h"
if not header.exists():
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print()
    print("WARNING: Python development headers were not found.")
    print(f"Missing: {header}")
    print("vLLM/Torch may fail at runtime while compiling CUDA helpers.")
    print("On Ubuntu/Debian, install:")
    print(f"  sudo apt-get install -y python{version}-dev build-essential")
PY
register_kernel_if_requested vllm_venv "vLLM Benchmark"

echo
echo "vLLM environment is ready."
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Check vLLM with: vllm --version"
