# Speculative Decoding and FP8 Quantization Homework

This repo contains the notebook and runnable instructions for the LLM performance engineering homework on EAGLE-3 speculative decoding and FP8 quantization for `Qwen/Qwen3-8B`.

Target hardware:

```text
1x NVIDIA H100 80GB
```

## Quick Start

Clone the repo on the H100 server, then create a local `.env` if needed:

```bash
cp configs/project.env.example .env
```

Common `.env` values:

```bash
export PYTHON_BIN=python3.12
export HF_HOME=/path/to/large/disk/hf
export TRANSFORMERS_CACHE=/path/to/large/disk/hf/transformers
export HF_DATASETS_CACHE=/path/to/large/disk/hf/datasets
export HF_TOKEN=hf_xxx
```

`HF_TOKEN` is only needed if Hugging Face downloads require authentication.

Create the three environments:

```bash
bash scripts/setup_all_envs.sh
```

Expected environments:

```text
.venvs/speculators_venv
.venvs/vllm_venv
.venvs/comp_venv
```

## Follow The Instructions In Order

1. [Environment and Data Orchestration](_instructions/1.environment_and_data_orchestration.md)

   Create or verify the environments, preprocess ShareGPT-style data, launch the hidden-state vLLM server, and generate offline hidden states.

2. [Train The EAGLE-3 Draft Head](_instructions/2.train_eagle3_draft_head.md)

   Train the draft head from the generated hidden states and choose the best checkpoint.

3. [Quantize Qwen3-8B To FP8](_instructions/3.fp8_quantization.md)

   Quantize the verifier with `llmcompressor`, keep `lm_head` unquantized, and save a separate FP8 model directory.

4. [Serve and Benchmark With vLLM](_instructions/4.serve_and_benchmark_with_vllm.md)

   Benchmark baseline, speculative decoding, FP8, and FP8 + speculative decoding with consistent settings.

5. [Final Notebook Work](_instructions/5.final_notebook_work.md)

   Collect benchmark blocks, training metrics, quantization evidence, and written answers for the final notebook submission.

## Final Notebook Work

Follow [step 5](_instructions/5.final_notebook_work.md) to fill `spec_dec+quantization_homework.ipynb` with:

- speculative decoding benchmark output;
- FP8 quantization benchmark output;
- FP8 + speculative decoding benchmark output;
- short explanations for the written questions.

Passing output-token-throughput thresholds:

| Configuration | Threshold |
| --- | ---: |
| EAGLE-3 speculative decoding | `> 1250 tok/s` |
| FP8 dynamic quantization | `> 1550 tok/s` |
| FP8 + EAGLE-3 speculative decoding | `> 1750 tok/s` |

## Generated Artifacts

These are intentionally ignored by git:

```text
.venvs/
external/
data/raw/
data/processed/
data/hidden_states/
models/
output/checkpoints/
benchmarks/results/
```
