# Results

## Task 1: Environment And Data Orchestration

- Model: `Qwen/Qwen3-8B`
- Source dataset: `sharegpt` / `share_gpt_vicuna_unfiltered`
- Preprocessed data path: `data/processed/qwen3_8b_sharegpt`
- Preprocessed data disk usage: `17M`
- Preprocessed data size: `3000` rows
- Preprocessed columns: `input_ids`, `loss_mask`, `seq_len`
- Sequence length cap: `2048`
- Observed sequence lengths: min `22`, max `2048`, first sample `1918`
- Preprocessed files:
  - `data-00000-of-00001.arrow`
  - `dataset_info.json`
  - `state.json`
  - `token_freq.pt`
  - `d2t.npy`
  - `t2d.npy`
- Hidden-state path: `data/hidden_states/qwen3_8b_sharegpt_3k_seq2048`
- Hidden-state files: `3000` `hs_*.safetensors` files
- Hidden-state disk usage: `122G`

## Task 2: Train The EAGLE-3 Draft Head

- Training data size: `3000` preprocessed samples
- Hidden-state path: `data/hidden_states/qwen3_8b_sharegpt_3k_seq2048`
- Hidden-state disk usage: `122G`
- Training run path: `output/checkpoints/eagle3_qwen3_8b_3k_seq2048`
- Best checkpoint path: `output/checkpoints/eagle3_qwen3_8b_3k_seq2048/4`
- Best checkpoint disk usage: `3.0G`
- Epoch selected: `4` (fifth epoch)
- Speculator type: `eagle3`
- Draft vocab size: `32000`
- Draft speculative tokens: `3`
- EAGLE auxiliary hidden-state layer ids: `2`, `18`, `33`
- Verifier: `Qwen/Qwen3-8B`

### Final Validation Metrics

| Metric | Value |
|---|---:|
| `val/loss_0_epoch` | `2.5099567186583007` |
| `val/full_acc_0_epoch` | `0.46552755526974476` |
| `val/cond_acc_0_epoch` | `0.46552755526974476` |
| `val/loss_1_epoch` | `3.7801205950425416` |
| `val/full_acc_1_epoch` | `0.18205723871552892` |
| `val/cond_acc_1_epoch` | `0.3652527761433769` |
| `val/loss_2_epoch` | `4.548701453842063` |
| `val/full_acc_2_epoch` | `0.06928455412323895` |
| `val/cond_acc_2_epoch` | `0.32223169854142636` |
| `val/loss_epoch` | `10.838778763433895` |

### Position-Wise Accuracy

| Draft position | `full_acc` | `cond_acc` | `loss` |
|---:|---:|---:|---:|
| 0 | `0.46552755526974476` | `0.46552755526974476` | `2.5099567186583007` |
| 1 | `0.18205723871552892` | `0.3652527761433769` | `3.7801205950425416` |
| 2 | `0.06928455412323895` | `0.32223169854142636` | `4.548701453842063` |

### Short Answers

- `full_acc_k` measures whether the draft prediction at position `k` is correct as part of the full speculative chain, so earlier mistakes reduce later-position full accuracy.
- `cond_acc_k` measures whether the draft prediction at position `k` is correct given that earlier speculative positions were already correct.
- Later positions are harder because the draft is predicting farther ahead with compounding uncertainty; this is visible here as `full_acc` drops from about `0.466` at position 0 to about `0.069` at position 2.
- If position-0 accuracy is very low, inspect the data pipeline first: confirm the same `MODEL_ID` and tokenizer were used, `SEQ_LENGTH` matches preprocessing, hidden-state generation, and training, EAGLE target layers match between generation and training, hidden-state files are complete and not stale, and a small regenerated batch passes output validation.

## Task 3: FP8 Quantization

- Base verifier model: `Qwen/Qwen3-8B`
- Quantized verifier path: `models/Qwen3-8B-FP8-Dynamic`
- Quantized model disk usage: `8.9G`
- Main weight file: `models/Qwen3-8B-FP8-Dynamic/model.safetensors`
- Main weight file size: `9438581928` bytes
- Saved files:
  - `chat_template.jinja`
  - `config.json`
  - `generation_config.json`
  - `model.safetensors`
  - `recipe.yaml`
  - `tokenizer.json`
  - `tokenizer_config.json`
- Quantization environment versions:
  - `llmcompressor`: `0.12.0`
  - `transformers`: `5.10.1`
  - `torch`: `2.12.0`
  - `safetensors`: `0.8.0`

### Quantization Configuration

- Quantization method: `compressed-tensors`
- Quantization status: `compressed`
- Overall format: `float-quantized`
- Recipe scheme: `FP8_DYNAMIC`
- Target modules: `Linear`
- Ignored module: `lm_head`
- Weight quantization: 8-bit float, static, channel-wise, symmetric, `memoryless_minmax` observer
- Input activation quantization: 8-bit float, dynamic, token-wise, symmetric

### Notebook Notes

- FP8 dynamic quantization is useful on H100 because Hopper GPUs have strong FP8 support, and the verifier becomes less memory-bandwidth-heavy.
- `lm_head` is often excluded because it is large, numerically sensitive, and directly controls logits. Keeping it unquantized can reduce quality and acceptance-rate regressions.
- Quantization can change verifier logits slightly. Speculative decoding is lossless relative to the verifier it actually runs, but changed verifier numerics can alter acceptance behavior and the best `num_speculative_tokens`.

## Task 4: Serve And Benchmark With vLLM

- Base model: `Qwen/Qwen3-8B`
- FP8 model: `models/Qwen3-8B-FP8-Dynamic`
- Draft model: `output/checkpoints/eagle3_qwen3_8b_3k_seq2048/4`
- Benchmark result directory: `benchmarks/results`
- Benchmark settings:
  - prompts: `80`
  - max concurrency: `8`
  - seed: `0`
  - dataset: `philschmid/mt-bench`

### Step 9 Comparison Table

| Configuration | Draft tokens | Output tok/s | Mean TTFT ms | Mean TPOT ms | Acceptance rate | Acceptance length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline BF16 | N/A | `838.44` | `659.98` | `6.99` | N/A | N/A |
| EAGLE-3 BF16 | `1` | `1201.13` | `126.04` | `5.84` | `37.03%` | `1.37` |
| FP8 | N/A | `1069.66` | `593.03` | `5.18` | N/A | N/A |
| FP8 + EAGLE-3 | `2` | `1628.42` | `60.09` | `4.48` | `22.67%` | `1.45` |

### Threshold Check

| Configuration | Threshold | Observed | Result |
| --- | ---: | ---: | --- |
| EAGLE-3 BF16 | `> 1250 tok/s` | `1201.13 tok/s` | below threshold |
| FP8 | `> 1550 tok/s` | `1069.66 tok/s` | below threshold |
| FP8 + EAGLE-3 | `> 1750 tok/s` | `1628.42 tok/s` | below threshold |

### Speculative Token Selection

- Best BF16 speculative run by output throughput: `SPEC_TOKENS=1`, from `benchmarks/results/spec_bf16_n1.json`.
- Best FP8 speculative run by output throughput: `SPEC_TOKENS=2`, from `benchmarks/results/fp8_spec_n2.json`.
- Larger BF16 draft counts reduced acceptance rate and did not improve throughput. `SPEC_TOKENS=1` had the best BF16 throughput and the highest acceptance rate.
- For FP8, `SPEC_TOKENS=2` was best: it gave the highest output throughput while keeping TPOT low. `SPEC_TOKENS=3` and `4` accepted slightly more total tokens but spent more work on rejected draft tokens, so throughput was lower.

### Speculative Run Details

| Run | Draft tokens | Output tok/s | Mean TTFT ms | Mean TPOT ms | Acceptance rate | Acceptance length | Drafts | Draft tokens | Accepted tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `spec_bf16_n1` | `1` | `1201.13` | `126.04` | `5.84` | `37.03%` | `1.37` | `14865` | `14865` | `5504` |
| `spec_bf16_n2` | `2` | `834.04` | `815.22` | `5.86` | `21.61%` | `1.43` | `14253` | `28506` | `6161` |
| `spec_bf16_n3` | `3` | `1183.20` | `138.17` | `5.97` | `15.32%` | `1.46` | `13999` | `41997` | `6432` |
| `spec_bf16_n4` | `4` | `1170.54` | `77.12` | `6.26` | `11.57%` | `1.46` | `13965` | `55860` | `6464` |
| `fp8_spec_n1` | `1` | `1032.13` | `746.79` | `4.40` | `35.51%` | `1.36` | `15035` | `15035` | `5339` |
| `fp8_spec_n2` | `2` | `1628.42` | `60.09` | `4.48` | `22.67%` | `1.45` | `14054` | `28108` | `6371` |
| `fp8_spec_n3` | `3` | `1569.86` | `54.43` | `4.64` | `15.43%` | `1.46` | `13972` | `41916` | `6466` |
| `fp8_spec_n4` | `4` | `1562.29` | `63.47` | `4.65` | `15.26%` | `1.46` | `13980` | `41940` | `6401` |

### Step 10 Benchmark Text Blocks

#### Speculative Decoding Benchmark Results

```text
============ Serving Benchmark Result ============
Successful requests:                     80
Failed requests:                         0
Maximum request concurrency:             8
Benchmark duration (s):                  17.01
Total input tokens:                      6078
Total generated tokens:                  20431
Request throughput (req/s):              4.70
Output token throughput (tok/s):         1201.13
Peak output token throughput (tok/s):    1014.00
Peak concurrent requests:                16.00
Total token throughput (tok/s):          1558.45
---------------Time to First Token----------------
Mean TTFT (ms):                          126.04
Median TTFT (ms):                        24.87
P99 TTFT (ms):                           1036.24
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          5.84
Median TPOT (ms):                        5.81
P99 TPOT (ms):                           7.90
---------------Inter-token Latency----------------
Mean ITL (ms):                           8.00
Median ITL (ms):                         7.88
P99 ITL (ms):                            8.82
---------------Speculative Decoding---------------
Acceptance rate (%):                     37.03
Acceptance length:                       1.37
Drafts:                                  14865
Draft tokens:                            14865
Accepted tokens:                         5504
Per-position acceptance (%):
  Position 0:                            37.03
==================================================
```

#### FP8 Quantization Benchmark Results

```text
============ Serving Benchmark Result ============
Successful requests:                     80
Failed requests:                         0
Maximum request concurrency:             8
Benchmark duration (s):                  19.15
Total input tokens:                      6078
Total generated tokens:                  20480
Request throughput (req/s):              4.18
Output token throughput (tok/s):         1069.66
Peak output token throughput (tok/s):    1640.00
Peak concurrent requests:                16.00
Total token throughput (tok/s):          1387.11
---------------Time to First Token----------------
Mean TTFT (ms):                          593.03
Median TTFT (ms):                        28.44
P99 TTFT (ms):                           6054.03
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          5.18
Median TPOT (ms):                        4.89
P99 TPOT (ms):                           9.88
---------------Inter-token Latency----------------
Mean ITL (ms):                           5.18
Median ITL (ms):                         4.88
P99 ITL (ms):                            5.49
==================================================
```

#### FP8 + Speculative Decoding Benchmark Results

```text
============ Serving Benchmark Result ============
Successful requests:                     80
Failed requests:                         0
Maximum request concurrency:             8
Benchmark duration (s):                  12.58
Total input tokens:                      6078
Total generated tokens:                  20480
Request throughput (req/s):              6.36
Output token throughput (tok/s):         1628.42
Peak output token throughput (tok/s):    1234.00
Peak concurrent requests:                16.00
Total token throughput (tok/s):          2111.70
---------------Time to First Token----------------
Mean TTFT (ms):                          60.09
Median TTFT (ms):                        19.94
P99 TTFT (ms):                           429.56
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          4.48
Median TPOT (ms):                        4.53
P99 TPOT (ms):                           5.08
---------------Inter-token Latency----------------
Mean ITL (ms):                           6.50
Median ITL (ms):                         6.44
P99 ITL (ms):                            8.28
---------------Speculative Decoding---------------
Acceptance rate (%):                     22.67
Acceptance length:                       1.45
Drafts:                                  14054
Draft tokens:                            28108
Accepted tokens:                         6371
Per-position acceptance (%):
  Position 0:                            36.45
  Position 1:                            8.89
==================================================
```

### Step 10 Explanation

- Speculative decoding helps because the verifier checks multiple proposed tokens in one pass, so accepted tokens reduce the number of expensive verifier steps.
- Acceptance rate does not need to be close to 100% because the verifier pass can still amortize multiple tokens when enough proposals are accepted.
- FP8 reduces verifier memory bandwidth and compute pressure on H100.
- FP8 can change verifier numerics, so the optimal speculative token count and acceptance rate can differ from BF16.

### Metrics Completeness

- All Step 9 table fields were available from `benchmarks/results/*.json` and `benchmarks/results/*.txt`.
- All Step 10 requested benchmark text blocks were available from the saved `.txt` files.
- Acceptance rate, acceptance length, drafts, draft tokens, accepted tokens, and per-position acceptance were available in the benchmark JSON/text outputs for every speculative run.
- I did not find a saved Prometheus `/metrics` endpoint snapshot, but it is not needed here because the benchmark outputs already include the speculative acceptance metrics requested by the runbook.
