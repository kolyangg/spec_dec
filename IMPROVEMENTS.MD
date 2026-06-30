# Improvements From Initial To Final Runs

This file summarizes what changed between the initial benchmark run in `benchmarks/results/` and the final passing run in `analysis_runs/tuned_20260629_2319/`.

## Executive Summary

The biggest improvement was not a new model checkpoint or a new quantization recipe. The largest gain came from fixing the benchmark and serving orchestration so the H100 was driven with enough concurrent work.

The initial run used only `80` prompts with max concurrency `8`. That underloaded the server and made every configuration look slower than the homework thresholds. The final tuned run swept concurrency `8`, `16`, `24`, and `32`, used `20` prompts per concurrency level, and selected the best completed run at concurrency `32`.

## Before And After

| Configuration | Initial best run | Initial output tok/s | Initial threshold result | Final best run | Final output tok/s | Final threshold result | Improvement |
| --- | --- | ---: | --- | --- | ---: | --- | ---: |
| Baseline BF16 | `baseline_bf16` | `838.44` | N/A | `baseline_bf16_c32` | `4215.67` | N/A | `5.03x` |
| EAGLE-3 BF16 | `spec_bf16_n1` | `1201.13` | below `1250` | `spec_bf16_n1_c32` | `5040.63` | pass | `4.20x` |
| FP8 | `fp8` | `1069.66` | below `1550` | `fp8_c32` | `5704.46` | pass | `5.33x` |
| FP8 + EAGLE-3 | `fp8_spec_n2` | `1628.42` | below `1750` | `fp8_spec_n1_c32` | `6569.96` | pass | `4.03x` |

Threshold margins improved from slightly or substantially negative to very comfortably positive:

| Configuration | Initial margin | Final margin | Final / threshold |
| --- | ---: | ---: | ---: |
| EAGLE-3 BF16 | `-48.87 tok/s` | `+3790.63 tok/s` | `4.03x` |
| FP8 | `-480.34 tok/s` | `+4154.46 tok/s` | `3.68x` |
| FP8 + EAGLE-3 | `-121.58 tok/s` | `+4819.96 tok/s` | `3.75x` |

## What Helped Most

### 1. Higher-Concurrency Benchmarking

This was the main unlock. The initial run measured each configuration at:

```text
completed requests: 80
max concurrency:    8
```

The final selected runs measured:

```text
completed requests: 640
max concurrency:    32
```

That change alone better matched how vLLM batches work on an H100. The server needed enough simultaneous requests to fill batching capacity and amortize overhead. At low concurrency, TTFT and orchestration overhead dominated the measurement.

The evidence is visible even for the baseline:

| Run | Requests | Concurrency | Output tok/s |
| --- | ---: | ---: | ---: |
| `baseline_bf16` | `80` | `8` | `838.44` |
| `baseline_bf16_c8` | `160` | `8` | `1144.61` |
| `baseline_bf16_c16` | `320` | `16` | `2188.35` |
| `baseline_bf16_c24` | `480` | `24` | `3254.23` |
| `baseline_bf16_c32` | `640` | `32` | `4215.67` |

Throughput scaled strongly with concurrency, which explains why the initial low-load run missed thresholds despite the model artifacts being valid.

### 2. vLLM Server Capacity Settings

The final scripts used a more explicit serving configuration:

```text
gpu-memory-utilization: 0.92
max-num-seqs:          64
max-num-batched-tokens:32768
generation-config:     vllm
prefix caching:        disabled for consistent benchmark behavior
```

These settings gave vLLM room to batch the higher-concurrency benchmark. They were most useful together with the concurrency sweep; by themselves they would not help much if the benchmark still sent too little work.

### 3. Independent SPEC_TOKENS Sweep For BF16 And FP8

We swept `SPEC_TOKENS=1 2 3 4` separately for BF16 speculative serving and FP8 speculative serving. This mattered because more draft tokens did not mean better throughput.

Final BF16 speculative results at concurrency `32`:

| Run | SPEC_TOKENS | Output tok/s | Acceptance rate | Acceptance length |
| --- | ---: | ---: | ---: | ---: |
| `spec_bf16_n1_c32` | `1` | `5040.63` | `35.47%` | `1.35` |
| `spec_bf16_n2_c32` | `2` | `4967.04` | `21.75%` | `1.44` |
| `spec_bf16_n3_c32` | `3` | `4777.04` | `14.93%` | `1.45` |
| `spec_bf16_n4_c32` | `4` | `4388.77` | `11.27%` | `1.45` |

Final FP8 speculative results at concurrency `32`:

| Run | SPEC_TOKENS | Output tok/s | Acceptance rate | Acceptance length |
| --- | ---: | ---: | ---: | ---: |
| `fp8_spec_n1_c32` | `1` | `6569.96` | `35.41%` | `1.35` |
| `fp8_spec_n2_c32` | `2` | `5901.86` | `21.92%` | `1.44` |
| `fp8_spec_n3_c32` | `3` | `5765.15` | `14.98%` | `1.45` |
| `fp8_spec_n4_c32` | `4` | `4648.82` | `11.31%` | `1.45` |

The best final setting was `SPEC_TOKENS=1` for both BF16 speculative and FP8 speculative serving. Higher draft counts had lower acceptance rates and spent more work on tokens the verifier rejected.

This changed the FP8 speculative selection: the initial low-load run picked `fp8_spec_n2`, while the final high-load sweep picked `fp8_spec_n1_c32`.

### 4. FP8 Quantization Was Effective Once The Benchmark Was Properly Loaded

FP8 did not pass in the initial low-load run:

```text
initial fp8: 1069.66 tok/s, threshold 1550 tok/s
```

But under the final benchmark load:

```text
final fp8_c32: 5704.46 tok/s
```

FP8 also improved throughput relative to BF16 baseline in both runs:

| Comparison | Initial | Final |
| --- | ---: | ---: |
| FP8 / BF16 baseline output throughput | `1.28x` | `1.35x` |

So the quantized verifier was doing the right thing. The initial failure was mostly that the benchmark did not expose the available throughput.

### 5. Resumable And Retryable Benchmark Scripts

One final practical improvement was robustness. During the final sweep, one `vllm bench serve` call hit a transient HuggingFace `ReadTimeout` while reading `philschmid/mt-bench` metadata. We updated `scripts/rerun_analysis/task4_benchmark_sweep.sh` so that it:

- skips benchmark labels that already have a completed JSON result;
- retries failed benchmark commands;
- increases HuggingFace Hub metadata/download timeouts;
- resumes the same `RUN_ID` instead of forcing a full rerun.

This did not directly increase throughput, but it made the full sweep finish reliably and prevented losing already-completed runs.

## What Did Not Change Much

- The final run used the same draft model path: `output/checkpoints/eagle3_qwen3_8b_3k_seq2048/4`.
- The final run used the same FP8 model path: `models/Qwen3-8B-FP8-Dynamic`.
- The EAGLE validation metrics and FP8 quantization metadata were already reasonable.

So the main story is: the artifacts were good enough, but the initial benchmark was too small and underutilized the serving stack.

## Final Passing Configuration

Use these as the notebook-facing final benchmark choices:

| Notebook section | Final run | Why |
| --- | --- | --- |
| Speculative decoding benchmark results | `spec_bf16_n1_c32` | Best BF16 speculative throughput: `5040.63 tok/s` |
| FP8 quantization benchmark results | `fp8_c32` | Best FP8 throughput: `5704.46 tok/s` |
| FP8 + speculative decoding benchmark results | `fp8_spec_n1_c32` | Best combined throughput: `6569.96 tok/s` |

The final benchmark settings to preserve are:

```text
SWEEP_CONCURRENCY="8 16 24 32"
BF16_SPEC_TOKENS="1 2 3 4"
FP8_SPEC_TOKENS="1 2 3 4"
PROMPTS_PER_CONCURRENCY=20
GPU_MEM_UTIL=0.92
MAX_NUM_SEQS=64
MAX_NUM_BATCHED_TOKENS=32768
```

## Bottom Line

The most important improvement was increasing benchmark load and selecting from a real concurrency sweep. FP8 quantization and EAGLE-3 speculative decoding both helped, but their benefit only became visible once vLLM had enough concurrent requests to batch efficiently.
