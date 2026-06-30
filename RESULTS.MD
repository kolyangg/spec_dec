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

- Final tuned run id: `tuned_20260629_2319`
- Base model: `Qwen/Qwen3-8B`
- FP8 model: `models/Qwen3-8B-FP8-Dynamic`
- Draft model: `output/checkpoints/eagle3_qwen3_8b_3k_seq2048/4`
- Benchmark result directory: `analysis_runs/tuned_20260629_2319/benchmarks/results`
- Tuned summary file: `analysis_runs/tuned_20260629_2319/metrics/RESULTS_TUNED.md`
- Benchmark JSON summary: `analysis_runs/tuned_20260629_2319/metrics/task4_metrics.json`
- Benchmark CSV summary: `analysis_runs/tuned_20260629_2319/metrics/task4_metrics.csv`
- Completed benchmark files: `50` JSON files and `50` text files
- Measured benchmark rows: `40`; warmup rows: `10`
- Benchmark settings:
  - dataset: `philschmid/mt-bench`
  - seed: `0`
  - `--ignore-eos`: enabled
  - concurrency sweep: `8`, `16`, `24`, `32`
  - prompts per concurrency: `20`
  - measured prompts: `160`, `320`, `480`, `640`
  - GPU memory utilization: `0.92`
  - `max_num_seqs`: `64`
  - `max_num_batched_tokens`: `32768`

### Step 9 Comparison Table

| Configuration | Source run | Draft tokens | Requests | Output tok/s | Total tok/s | Req/s | Mean TTFT ms | Mean TPOT ms | Acceptance rate | Acceptance length |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline BF16 | `baseline_bf16_c32` | N/A | `640` | `4215.67` | `5508.26` | `16.47` | `65.67` | `7.36` | N/A | N/A |
| EAGLE-3 BF16 | `spec_bf16_n1_c32` | `1` | `640` | `5040.63` | `6586.17` | `19.69` | `30.98` | `6.14` | `35.47%` | `1.35` |
| FP8 | `fp8_c32` | N/A | `640` | `5704.46` | `7453.55` | `22.28` | `52.86` | `5.41` | N/A | N/A |
| FP8 + EAGLE-3 | `fp8_spec_n1_c32` | `1` | `640` | `6569.96` | `8584.41` | `25.66` | `31.18` | `4.68` | `35.41%` | `1.35` |

### Threshold Check

| Configuration | Threshold | Observed | Result |
| --- | ---: | ---: | --- |
| EAGLE-3 BF16 | `> 1250 tok/s` | `5040.63 tok/s` | pass |
| FP8 | `> 1550 tok/s` | `5704.46 tok/s` | pass |
| FP8 + EAGLE-3 | `> 1750 tok/s` | `6569.96 tok/s` | pass |

### Speculative Token Selection

- Best BF16 speculative run by output throughput: `SPEC_TOKENS=1`, from `spec_bf16_n1_c32`.
- Best FP8 speculative run by output throughput: `SPEC_TOKENS=1`, from `fp8_spec_n1_c32`.
- In this completed sweep, the highest-throughput point for all final configurations was concurrency `32`.
- Larger speculative counts (`2`, `3`, `4`) had lower acceptance rates and did not beat `SPEC_TOKENS=1` for either BF16 or FP8 speculative serving.

### All Measured Runs

| Run | Group | Draft tokens | Concurrency | Requests | Output tok/s | Total tok/s | Req/s | Mean TTFT ms | P99 TTFT ms | Mean TPOT ms | Acceptance rate | Acceptance length |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_bf16_c8` | `baseline_bf16` | N/A | `8` | `160` | `1144.61` | `1503.66` | `4.47` | `39.19` | `119.92` | `6.86` | N/A | N/A |
| `baseline_bf16_c16` | `baseline_bf16` | N/A | `16` | `320` | `2188.35` | `2887.49` | `8.55` | `50.71` | `72.48` | `7.13` | N/A | N/A |
| `baseline_bf16_c24` | `baseline_bf16` | N/A | `24` | `480` | `3254.23` | `4273.06` | `12.71` | `51.59` | `88.16` | `7.19` | N/A | N/A |
| `baseline_bf16_c32` | `baseline_bf16` | N/A | `32` | `640` | `4215.67` | `5508.26` | `16.47` | `65.67` | `105.27` | `7.36` | N/A | N/A |
| `spec_bf16_n1_c8` | `spec_bf16` | `1` | `8` | `160` | `1354.67` | `1779.62` | `5.29` | `56.63` | `694.23` | `5.58` | `35.48%` | `1.35` |
| `spec_bf16_n1_c16` | `spec_bf16` | `1` | `16` | `320` | `2646.82` | `3492.44` | `10.34` | `31.65` | `168.09` | `5.79` | `35.54%` | `1.36` |
| `spec_bf16_n1_c24` | `spec_bf16` | `1` | `24` | `480` | `3867.77` | `5078.68` | `15.11` | `29.08` | `91.53` | `6.00` | `35.28%` | `1.35` |
| `spec_bf16_n1_c32` | `spec_bf16` | `1` | `32` | `640` | `5040.63` | `6586.17` | `19.69` | `30.98` | `105.59` | `6.14` | `35.47%` | `1.35` |
| `spec_bf16_n2_c8` | `spec_bf16` | `2` | `8` | `160` | `1391.34` | `1827.80` | `5.43` | `57.75` | `689.96` | `5.47` | `21.95%` | `1.44` |
| `spec_bf16_n2_c16` | `spec_bf16` | `2` | `16` | `320` | `2646.16` | `3491.57` | `10.34` | `33.19` | `179.66` | `5.79` | `21.85%` | `1.44` |
| `spec_bf16_n2_c24` | `spec_bf16` | `2` | `24` | `480` | `3863.14` | `5072.60` | `15.09` | `29.53` | `90.26` | `6.00` | `21.73%` | `1.43` |
| `spec_bf16_n2_c32` | `spec_bf16` | `2` | `32` | `640` | `4967.04` | `6490.01` | `19.40` | `31.94` | `109.68` | `6.20` | `21.75%` | `1.44` |
| `spec_bf16_n3_c8` | `spec_bf16` | `3` | `8` | `160` | `1328.80` | `1745.64` | `5.19` | `58.46` | `694.13` | `5.73` | `14.75%` | `1.44` |
| `spec_bf16_n3_c16` | `spec_bf16` | `3` | `16` | `320` | `2559.88` | `3377.71` | `10.00` | `34.43` | `181.40` | `6.02` | `14.98%` | `1.45` |
| `spec_bf16_n3_c24` | `spec_bf16` | `3` | `24` | `480` | `3666.06` | `4813.83` | `14.32` | `30.95` | `90.22` | `6.31` | `14.87%` | `1.45` |
| `spec_bf16_n3_c32` | `spec_bf16` | `3` | `32` | `640` | `4777.04` | `6241.75` | `18.66` | `32.84` | `106.05` | `6.44` | `14.93%` | `1.45` |
| `spec_bf16_n4_c8` | `spec_bf16` | `4` | `8` | `160` | `1302.62` | `1711.25` | `5.09` | `27.80` | `57.83` | `5.99` | `11.23%` | `1.45` |
| `spec_bf16_n4_c16` | `spec_bf16` | `4` | `16` | `320` | `2413.92` | `3185.13` | `9.43` | `37.41` | `208.31` | `6.36` | `11.32%` | `1.45` |
| `spec_bf16_n4_c24` | `spec_bf16` | `4` | `24` | `480` | `3462.52` | `4546.55` | `13.53` | `32.58` | `89.94` | `6.68` | `11.18%` | `1.45` |
| `spec_bf16_n4_c32` | `spec_bf16` | `4` | `32` | `640` | `4388.77` | `5734.43` | `17.14` | `35.00` | `109.46` | `7.02` | `11.27%` | `1.45` |
| `fp8_c8` | `fp8` | N/A | `8` | `160` | `1644.21` | `2160.00` | `6.42` | `32.91` | `114.09` | `4.75` | N/A | N/A |
| `fp8_c16` | `fp8` | N/A | `16` | `320` | `3108.20` | `4101.21` | `12.14` | `41.42` | `60.95` | `5.00` | N/A | N/A |
| `fp8_c24` | `fp8` | N/A | `24` | `480` | `4327.63` | `5682.52` | `16.90` | `46.69` | `78.60` | `5.38` | N/A | N/A |
| `fp8_c32` | `fp8` | N/A | `32` | `640` | `5704.46` | `7453.55` | `22.28` | `52.86` | `93.31` | `5.41` | N/A | N/A |
| `fp8_spec_n1_c8` | `fp8_spec` | `1` | `8` | `160` | `1861.93` | `2446.01` | `7.27` | `62.32` | `821.33` | `4.00` | `35.58%` | `1.36` |
| `fp8_spec_n1_c16` | `fp8_spec` | `1` | `16` | `320` | `3530.14` | `4657.96` | `13.79` | `30.87` | `158.42` | `4.37` | `35.48%` | `1.35` |
| `fp8_spec_n1_c24` | `fp8_spec` | `1` | `24` | `480` | `5133.22` | `6740.32` | `20.05` | `29.12` | `86.29` | `4.50` | `35.19%` | `1.35` |
| `fp8_spec_n1_c32` | `fp8_spec` | `1` | `32` | `640` | `6569.96` | `8584.41` | `25.66` | `31.18` | `93.33` | `4.68` | `35.41%` | `1.35` |
| `fp8_spec_n2_c8` | `fp8_spec` | `2` | `8` | `160` | `1802.59` | `2368.05` | `7.04` | `25.07` | `55.34` | `4.27` | `21.45%` | `1.43` |
| `fp8_spec_n2_c16` | `fp8_spec` | `2` | `16` | `320` | `3487.39` | `4601.55` | `13.62` | `32.44` | `185.05` | `4.39` | `21.78%` | `1.44` |
| `fp8_spec_n2_c24` | `fp8_spec` | `2` | `24` | `480` | `4708.51` | `6182.64` | `18.39` | `28.28` | `92.18` | `4.90` | `21.64%` | `1.43` |
| `fp8_spec_n2_c32` | `fp8_spec` | `2` | `32` | `640` | `5901.86` | `7711.46` | `23.05` | `54.32` | `561.28` | `5.08` | `21.92%` | `1.44` |
| `fp8_spec_n3_c8` | `fp8_spec` | `3` | `8` | `160` | `1701.16` | `2234.81` | `6.65` | `57.56` | `717.64` | `4.41` | `14.64%` | `1.44` |
| `fp8_spec_n3_c16` | `fp8_spec` | `3` | `16` | `320` | `3304.63` | `4360.40` | `12.91` | `32.20` | `184.38` | `4.62` | `14.90%` | `1.45` |
| `fp8_spec_n3_c24` | `fp8_spec` | `3` | `24` | `480` | `4523.71` | `5939.98` | `17.67` | `28.63` | `87.18` | `5.13` | `14.81%` | `1.44` |
| `fp8_spec_n3_c32` | `fp8_spec` | `3` | `32` | `640` | `5765.15` | `7532.84` | `22.52` | `30.35` | `104.37` | `5.33` | `14.98%` | `1.45` |
| `fp8_spec_n4_c8` | `fp8_spec` | `4` | `8` | `160` | `1668.99` | `2192.55` | `6.52` | `31.10` | `192.27` | `4.60` | `11.01%` | `1.44` |
| `fp8_spec_n4_c16` | `fp8_spec` | `4` | `16` | `320` | `2955.71` | `3900.00` | `11.55` | `30.06` | `128.75` | `5.20` | `11.16%` | `1.45` |
| `fp8_spec_n4_c24` | `fp8_spec` | `4` | `24` | `480` | `4243.75` | `5572.38` | `16.58` | `28.84` | `82.36` | `5.46` | `11.18%` | `1.45` |
| `fp8_spec_n4_c32` | `fp8_spec` | `4` | `32` | `640` | `4648.82` | `6074.22` | `18.16` | `33.39` | `98.30` | `6.65` | `11.31%` | `1.45` |

### Step 10 Benchmark Text Blocks

#### Speculative Decoding Benchmark Results

```text
============ Serving Benchmark Result ============
Successful requests:                     640
Failed requests:                         0
Maximum request concurrency:             32
Benchmark duration (s):                  32.50
Total input tokens:                      50236
Total generated tokens:                  163840
Request throughput (req/s):              19.69
Output token throughput (tok/s):         5040.63
Peak output token throughput (tok/s):    3936.00
Peak concurrent requests:                64.00
Total token throughput (tok/s):          6586.17
---------------Time to First Token----------------
Mean TTFT (ms):                          30.98
Median TTFT (ms):                        24.99
P99 TTFT (ms):                           105.59
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          6.14
Median TPOT (ms):                        6.17
P99 TPOT (ms):                           6.75
---------------Inter-token Latency----------------
Mean ITL (ms):                           8.30
Median ITL (ms):                         8.16
P99 ITL (ms):                            17.58
---------------Speculative Decoding---------------
Acceptance rate (%):                     35.47
Acceptance length:                       1.35
Drafts:                                  120577
Draft tokens:                            120577
Accepted tokens:                         42766
Per-position acceptance (%):
  Position 0:                            35.47
==================================================
```

#### FP8 Quantization Benchmark Results

```text
============ Serving Benchmark Result ============
Successful requests:                     640
Failed requests:                         0
Maximum request concurrency:             32
Benchmark duration (s):                  28.72
Total input tokens:                      50236
Total generated tokens:                  163840
Request throughput (req/s):              22.28
Output token throughput (tok/s):         5704.46
Peak output token throughput (tok/s):    6061.00
Peak concurrent requests:                64.00
Total token throughput (tok/s):          7453.55
---------------Time to First Token----------------
Mean TTFT (ms):                          52.86
Median TTFT (ms):                        51.42
P99 TTFT (ms):                           93.31
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          5.41
Median TPOT (ms):                        5.41
P99 TPOT (ms):                           5.56
---------------Inter-token Latency----------------
Mean ITL (ms):                           5.41
Median ITL (ms):                         5.33
P99 ITL (ms):                            7.81
==================================================
```

#### FP8 + Speculative Decoding Benchmark Results

```text
============ Serving Benchmark Result ============
Successful requests:                     640
Failed requests:                         0
Maximum request concurrency:             32
Benchmark duration (s):                  24.94
Total input tokens:                      50236
Total generated tokens:                  163840
Request throughput (req/s):              25.66
Output token throughput (tok/s):         6569.96
Peak output token throughput (tok/s):    5127.00
Peak concurrent requests:                64.00
Total token throughput (tok/s):          8584.41
---------------Time to First Token----------------
Mean TTFT (ms):                          31.18
Median TTFT (ms):                        26.89
P99 TTFT (ms):                           93.33
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          4.68
Median TPOT (ms):                        4.69
P99 TPOT (ms):                           5.21
---------------Inter-token Latency----------------
Mean ITL (ms):                           6.33
Median ITL (ms):                         6.06
P99 ITL (ms):                            16.20
---------------Speculative Decoding---------------
Acceptance rate (%):                     35.41
Acceptance length:                       1.35
Drafts:                                  120655
Draft tokens:                            120655
Accepted tokens:                         42718
Per-position acceptance (%):
  Position 0:                            35.41
==================================================
```

### Step 10 Explanation

- Speculative decoding helps because the verifier checks multiple proposed tokens in one pass, so accepted tokens reduce the number of expensive verifier steps.
- Acceptance rate does not need to be close to 100% because the verifier pass can still amortize multiple tokens when enough proposals are accepted.
- FP8 reduces verifier memory bandwidth and compute pressure on H100.
- FP8 can change verifier numerics, so the optimal speculative token count and acceptance rate can differ from BF16.
- In this run, `SPEC_TOKENS=1` was best for both BF16 speculative serving and FP8 speculative serving. It had the highest acceptance rate and avoided extra draft work that did not pay back at `SPEC_TOKENS=2`, `3`, or `4`.

### Metrics Completeness

- All Step 9 table fields were available from `analysis_runs/tuned_20260629_2319/benchmarks/results/*.json`.
- All Step 10 requested benchmark text blocks were available from the matching `.txt` files.
- Acceptance rate, acceptance length, drafts, draft tokens, accepted tokens, and per-position acceptance were available for every speculative run.
- The saved root benchmark directory from the earlier low-concurrency attempt is superseded by the tuned run above.

## Task 5: Final Notebook Work

- Final collection summary: `analysis_runs/tuned_20260629_2319/metrics/FINAL_COLLECTION.md`
- Artifact sizes: `analysis_runs/tuned_20260629_2319/metrics/artifact_sizes.txt`
- Training validation extract: `analysis_runs/tuned_20260629_2319/metrics/training_validation_metrics.txt`
- Notebook TODO check: `analysis_runs/tuned_20260629_2319/metrics/notebook_todo_check.txt`
- Notebook TODO cells currently reported: `6`, `7`, `8`

### Final Artifact Sizes

| Artifact | Path | Size |
| --- | --- | ---: |
| Processed data | `data/processed/qwen3_8b_sharegpt` | `17M` |
| Hidden states | `data/hidden_states/qwen3_8b_sharegpt_3k_seq2048` | `122G` |
| FP8 model | `models/Qwen3-8B-FP8-Dynamic` | `8.9G` |
| Draft checkpoint run | `output/checkpoints/eagle3_qwen3_8b_3k_seq2048` | `3.0G` |

### Notebook Paste Targets

- Speculative decoding benchmark result: use `spec_bf16_n1_c32`.
- FP8 quantization benchmark result: use `fp8_c32`.
- FP8 + speculative decoding benchmark result: use `fp8_spec_n1_c32`.
- Written-answer material is included in the Task 2, Task 3, and Task 4 notebook notes above.
