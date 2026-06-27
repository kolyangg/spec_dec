#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/setup_speculators_env.sh"
"$SCRIPT_DIR/setup_vllm_env.sh"
"$SCRIPT_DIR/setup_quant_env.sh"

echo
echo "All homework environments are ready."
