#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

ensure_run_dirs

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
FP8_MODEL_DIR="${FP8_MODEL_DIR:-$ROOT_DIR/models/Qwen3-8B-FP8-Dynamic}"
REBUILD_FP8="${REBUILD_FP8:-0}"
RUN_VLLM_SMOKE="${RUN_VLLM_SMOKE:-1}"
QUANT_LOG="$LOG_DIR/task3_quantize.log"
VERIFY_LOG="$LOG_DIR/task3_verify_fp8.log"
SMOKE_LOG="$LOG_DIR/task3_vllm_smoke.log"
export MODEL_ID FP8_MODEL_DIR

print_run_context

if [[ "$REBUILD_FP8" == "1" || ! -f "$FP8_MODEL_DIR/config.json" ]]; then
  activate_named_venv comp
  mkdir -p "$(dirname "$FP8_MODEL_DIR")"
  echo "Building FP8 dynamic model at $FP8_MODEL_DIR"
  python - <<'PY' 2>&1 | tee "$QUANT_LOG"
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

model_id = os.environ["MODEL_ID"]
save_dir = os.environ["FP8_MODEL_DIR"]

print(f"Loading {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Applying FP8 dynamic quantization")
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
)
oneshot(model=model, recipe=recipe)

print(f"Saving quantized model to {save_dir}")
model.save_pretrained(save_dir, safe_serialization=True)
tokenizer.save_pretrained(save_dir)
PY
else
  echo "Reusing existing FP8 model: $FP8_MODEL_DIR" | tee "$QUANT_LOG"
fi

activate_named_venv comp
python - "$FP8_MODEL_DIR" "$METRICS_DIR/task3_fp8_summary.json" "$METRICS_DIR/task3_fp8_summary.md" <<'PY' 2>&1 | tee "$VERIFY_LOG"
import json
import subprocess
import sys
from pathlib import Path
from importlib.metadata import version

model_dir = Path(sys.argv[1])
json_out = Path(sys.argv[2])
md_out = Path(sys.argv[3])
cfg = json.loads((model_dir / "config.json").read_text())
quant_cfg = (
    cfg.get("quantization_config")
    or cfg.get("compression_config")
    or cfg.get("quantization")
)
if not quant_cfg:
    raise SystemExit("No quantization metadata found in config.json")

files = sorted(path.name for path in model_dir.iterdir() if path.is_file())
du = subprocess.check_output(["du", "-sh", str(model_dir)], text=True).split()[0]
summary = {
    "model_dir": str(model_dir),
    "disk_usage": du,
    "files": files,
    "packages": {
        "llmcompressor": version("llmcompressor"),
        "transformers": version("transformers"),
        "torch": version("torch"),
        "safetensors": version("safetensors"),
    },
    "quantization_config": quant_cfg,
}
json_out.write_text(json.dumps(summary, indent=2) + "\n")

group = next(iter(quant_cfg.get("config_groups", {}).values()))
weights = group.get("weights", {})
acts = group.get("input_activations", {})
lines = [
    "# Task 3 FP8 Summary",
    "",
    f"- FP8 model: `{model_dir}`",
    f"- Disk usage: `{du}`",
    f"- Quantization method: `{quant_cfg.get('quant_method')}`",
    f"- Quantization status: `{quant_cfg.get('quantization_status')}`",
    f"- Format: `{quant_cfg.get('format')}`",
    f"- Targets: `{', '.join(group.get('targets', []))}`",
    f"- Ignored modules: `{', '.join(quant_cfg.get('ignore', []))}`",
    f"- Weight bits/type: `{weights.get('num_bits')}` / `{weights.get('type')}`",
    f"- Activation bits/type/dynamic: `{acts.get('num_bits')}` / `{acts.get('type')}` / `{acts.get('dynamic')}`",
    "",
    "Saved files:",
]
lines.extend(f"- `{name}`" for name in files)
md_out.write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=2)[:4000])
PY

if [[ "$RUN_VLLM_SMOKE" == "1" ]]; then
  activate_named_venv vllm
  echo "Smoke-loading FP8 model with vLLM"
  python - <<'PY' 2>&1 | tee "$SMOKE_LOG"
from vllm import LLM, SamplingParams
import os

model_dir = os.environ["FP8_MODEL_DIR"]
llm = LLM(model=model_dir, max_num_seqs=1)
out = llm.generate(
    ["Say hello in one short sentence."],
    SamplingParams(max_tokens=16, temperature=0),
)
print(out[0].outputs[0].text)
PY
fi
