# Speculative Decoding + FP8 Quantization Homework TODO

## Goal

Build a reproducible workflow for `Qwen/Qwen3-8B` on one NVIDIA H100 80GB:

1. Train an EAGLE-3 speculative decoding draft head from offline hidden states.
2. Quantize the verifier model with FP8 dynamic quantization.
3. Benchmark baseline, speculative decoding, FP8, and FP8 + speculative decoding.
4. Fill the notebook with benchmark output and short technical explanations.

Main answer to support: speculative decoding training should be done before verifier quantization, then the quantized serving configuration should be benchmarked and retuned.

## Project Layout

- `scripts/`: setup and run scripts.
- `scripts/lib/`: shared shell helpers.
- `src/`: project Python utilities to add later.
- `configs/`: environment examples and future training/benchmark configs.
- `data/raw/`: source datasets, not committed.
- `data/processed/`: tokenized/prepared data, not committed.
- `data/hidden_states/`: generated offline hidden states, not committed.
- `models/`: local model outputs such as `Qwen3-8B-FP8-Dynamic`, not committed.
- `output/checkpoints/`: EAGLE-3 draft-head checkpoints, not committed.
- `benchmarks/results/`: copied `vllm bench serve` outputs, not committed.
- `external/`: cloned third-party repos such as `vllm-project/speculators`, not committed.
- `.venvs/`: local virtual environments, not committed.

## Setup TODO

- Clone this repo on the H100 server.
- Ensure Python 3.12 is available, or set `PYTHON_BIN` to another compatible Python executable.
- If needed, set Hugging Face/cache locations before running setup:
  - `HF_HOME`
  - `HF_TOKEN`
  - `TRANSFORMERS_CACHE`
  - `HF_DATASETS_CACHE`
- Run `bash scripts/setup_all_envs.sh`.
- Confirm these environments exist:
  - `.venvs/speculators_venv`
  - `.venvs/vllm_venv`
  - `.venvs/comp_venv`

## EAGLE-3 Training TODO

- Activate `.venvs/speculators_venv`.
- Follow the offline EAGLE-3 Speculators tutorial for `Qwen/Qwen3-8B`.
- Prepare ShareGPT-style training data.
- Start with approximately:
  - `max-samples=3000`
  - sequence length `2048`
- Generate offline hidden states into `data/hidden_states/`.
- Watch disk usage; hidden states can reach roughly 140GB for a few thousand samples.
- Clear stale `/tmp/hidden_states/*` if generation fails with missing temporary files.
- Train the EAGLE-3 draft head.
- Save checkpoints under `output/checkpoints/`.
- Record validation metrics:
  - `val/loss_*`
  - `val/full_acc_*`
  - `val/cond_acc_*`
- Select the best checkpoint for vLLM serving.

## Quantization TODO

- Activate `.venvs/comp_venv`.
- Use `llmcompressor==0.12.0`.
- Quantize `Qwen/Qwen3-8B` with FP8 dynamic quantization:
  - target linear layers
  - keep `lm_head` unquantized
  - save to `models/Qwen3-8B-FP8-Dynamic`
- Do not overwrite the original Hugging Face model.
- Verify the saved `config.json` includes quantization metadata.

## Benchmark TODO

- Activate `.venvs/vllm_venv`.
- Use the same benchmark settings for all runs:
  - dataset: `philschmid/mt-bench`
  - prompts: `80`
  - max concurrency: `8`
  - fixed seed where possible
  - prefix caching disabled unless intentionally tested
- Benchmark:
  - baseline `Qwen/Qwen3-8B`
  - BF16 verifier + EAGLE-3 draft head
  - FP8 dynamic verifier
  - FP8 dynamic verifier + EAGLE-3 draft head
- Tune speculative draft tokens independently for:
  - BF16 speculative decoding
  - FP8 + speculative decoding
- Compare output token throughput first, then explain TTFT, TPOT, acceptance rate, and acceptance length.

Passing thresholds:

| Configuration | Output token throughput threshold |
| --- | ---: |
| Speculative decoding | `> 1250 tok/s` |
| FP8 quantization | `> 1550 tok/s` |
| FP8 + speculative decoding | `> 1750 tok/s` |

## Notebook Submission TODO

Fill these sections in `spec_dec+quantization_homework.ipynb`:

- Speculative decoding benchmark results.
- FP8 quantization benchmark results.
- FP8 + speculative decoding benchmark results.

Also answer:

- Why hidden states require much more disk than raw text.
- What `full_acc` and `cond_acc` measure.
- Why accuracy decreases for later speculative positions.
- What to inspect if first-position accuracy is very low.
- Why FP8 dynamic quantization is useful on H100.
- Why `lm_head` may be excluded from quantization.
- How quantization can affect speculative decoding acceptance rate.
- Why speculative decoding can improve throughput even without near-100% acceptance.
- Which draft-token count was optimal and why.
