# Tuned Benchmark Results

## Best-Run Comparison

| Configuration | Source run | Draft tokens | Output tok/s | Mean TTFT ms | Mean TPOT ms | Acceptance rate | Acceptance length |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline BF16 | `baseline_bf16_c32` | N/A | `4215.67` | `65.67` | `7.36` | N/A | `N/A` |
| EAGLE-3 BF16 | `spec_bf16_n1_c32` | 1 | `5040.63` | `30.98` | `6.14` | 35.47% | `1.35` |
| FP8 | `fp8_c32` | N/A | `5704.46` | `52.86` | `5.41` | N/A | `N/A` |
| FP8 + EAGLE-3 | `fp8_spec_n1_c32` | 1 | `6569.96` | `31.18` | `4.68` | 35.41% | `1.35` |

## Threshold Check

| Configuration | Threshold | Observed | Result |
| --- | ---: | ---: | --- |
| EAGLE-3 BF16 | `> 1250 tok/s` | `5040.63 tok/s` | pass |
| FP8 | `> 1550 tok/s` | `5704.46 tok/s` | pass |
| FP8 + EAGLE-3 | `> 1750 tok/s` | `6569.96 tok/s` | pass |

## All Measured Runs

| Run | Group | Draft tokens | Output tok/s | Mean TTFT ms | P99 TTFT ms | Mean TPOT ms | Acceptance rate | Acceptance length |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_bf16_c16` | `baseline_bf16` | N/A | `2188.35` | `50.71` | `72.48` | `7.13` | N/A | `N/A` |
| `baseline_bf16_c24` | `baseline_bf16` | N/A | `3254.23` | `51.59` | `88.16` | `7.19` | N/A | `N/A` |
| `baseline_bf16_c32` | `baseline_bf16` | N/A | `4215.67` | `65.67` | `105.27` | `7.36` | N/A | `N/A` |
| `baseline_bf16_c8` | `baseline_bf16` | N/A | `1144.61` | `39.19` | `119.92` | `6.86` | N/A | `N/A` |
| `fp8_c16` | `fp8` | N/A | `3108.20` | `41.42` | `60.95` | `5.00` | N/A | `N/A` |
| `fp8_c24` | `fp8` | N/A | `4327.63` | `46.69` | `78.60` | `5.38` | N/A | `N/A` |
| `fp8_c32` | `fp8` | N/A | `5704.46` | `52.86` | `93.31` | `5.41` | N/A | `N/A` |
| `fp8_c8` | `fp8` | N/A | `1644.21` | `32.91` | `114.09` | `4.75` | N/A | `N/A` |
| `fp8_spec_n1_c16` | `fp8_spec` | 1 | `3530.14` | `30.87` | `158.42` | `4.37` | 35.48% | `1.35` |
| `fp8_spec_n1_c24` | `fp8_spec` | 1 | `5133.22` | `29.12` | `86.29` | `4.50` | 35.19% | `1.35` |
| `fp8_spec_n1_c32` | `fp8_spec` | 1 | `6569.96` | `31.18` | `93.33` | `4.68` | 35.41% | `1.35` |
| `fp8_spec_n1_c8` | `fp8_spec` | 1 | `1861.93` | `62.32` | `821.33` | `4.00` | 35.58% | `1.36` |
| `fp8_spec_n2_c16` | `fp8_spec` | 2 | `3487.39` | `32.44` | `185.05` | `4.39` | 21.78% | `1.44` |
| `fp8_spec_n2_c24` | `fp8_spec` | 2 | `4708.51` | `28.28` | `92.18` | `4.90` | 21.64% | `1.43` |
| `fp8_spec_n2_c32` | `fp8_spec` | 2 | `5901.86` | `54.32` | `561.28` | `5.08` | 21.92% | `1.44` |
| `fp8_spec_n2_c8` | `fp8_spec` | 2 | `1802.59` | `25.07` | `55.34` | `4.27` | 21.45% | `1.43` |
| `fp8_spec_n3_c16` | `fp8_spec` | 3 | `3304.63` | `32.20` | `184.38` | `4.62` | 14.90% | `1.45` |
| `fp8_spec_n3_c24` | `fp8_spec` | 3 | `4523.71` | `28.63` | `87.18` | `5.13` | 14.81% | `1.44` |
| `fp8_spec_n3_c32` | `fp8_spec` | 3 | `5765.15` | `30.35` | `104.37` | `5.33` | 14.98% | `1.45` |
| `fp8_spec_n3_c8` | `fp8_spec` | 3 | `1701.16` | `57.56` | `717.64` | `4.41` | 14.64% | `1.44` |
| `fp8_spec_n4_c16` | `fp8_spec` | 4 | `2955.71` | `30.06` | `128.75` | `5.20` | 11.16% | `1.45` |
| `fp8_spec_n4_c24` | `fp8_spec` | 4 | `4243.75` | `28.84` | `82.36` | `5.46` | 11.18% | `1.45` |
| `fp8_spec_n4_c32` | `fp8_spec` | 4 | `4648.82` | `33.39` | `98.30` | `6.65` | 11.31% | `1.45` |
| `fp8_spec_n4_c8` | `fp8_spec` | 4 | `1668.99` | `31.10` | `192.27` | `4.60` | 11.01% | `1.44` |
| `spec_bf16_n1_c16` | `spec_bf16` | 1 | `2646.82` | `31.65` | `168.09` | `5.79` | 35.54% | `1.36` |
| `spec_bf16_n1_c24` | `spec_bf16` | 1 | `3867.77` | `29.08` | `91.53` | `6.00` | 35.28% | `1.35` |
| `spec_bf16_n1_c32` | `spec_bf16` | 1 | `5040.63` | `30.98` | `105.59` | `6.14` | 35.47% | `1.35` |
| `spec_bf16_n1_c8` | `spec_bf16` | 1 | `1354.67` | `56.63` | `694.23` | `5.58` | 35.48% | `1.35` |
| `spec_bf16_n2_c16` | `spec_bf16` | 2 | `2646.16` | `33.19` | `179.66` | `5.79` | 21.85% | `1.44` |
| `spec_bf16_n2_c24` | `spec_bf16` | 2 | `3863.14` | `29.53` | `90.26` | `6.00` | 21.73% | `1.43` |
| `spec_bf16_n2_c32` | `spec_bf16` | 2 | `4967.04` | `31.94` | `109.68` | `6.20` | 21.75% | `1.44` |
| `spec_bf16_n2_c8` | `spec_bf16` | 2 | `1391.34` | `57.75` | `689.96` | `5.47` | 21.95% | `1.44` |
| `spec_bf16_n3_c16` | `spec_bf16` | 3 | `2559.88` | `34.43` | `181.40` | `6.02` | 14.98% | `1.45` |
| `spec_bf16_n3_c24` | `spec_bf16` | 3 | `3666.06` | `30.95` | `90.22` | `6.31` | 14.87% | `1.45` |
| `spec_bf16_n3_c32` | `spec_bf16` | 3 | `4777.04` | `32.84` | `106.05` | `6.44` | 14.93% | `1.45` |
| `spec_bf16_n3_c8` | `spec_bf16` | 3 | `1328.80` | `58.46` | `694.13` | `5.73` | 14.75% | `1.44` |
| `spec_bf16_n4_c16` | `spec_bf16` | 4 | `2413.92` | `37.41` | `208.31` | `6.36` | 11.32% | `1.45` |
| `spec_bf16_n4_c24` | `spec_bf16` | 4 | `3462.52` | `32.58` | `89.94` | `6.68` | 11.18% | `1.45` |
| `spec_bf16_n4_c32` | `spec_bf16` | 4 | `4388.77` | `35.00` | `109.46` | `7.02` | 11.27% | `1.45` |
| `spec_bf16_n4_c8` | `spec_bf16` | 4 | `1302.62` | `27.80` | `57.83` | `5.99` | 11.23% | `1.45` |

## Notebook Text Blocks

### Speculative Decoding Benchmark Results

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

### FP8 Quantization Benchmark Results

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

### FP8 + Speculative Decoding Benchmark Results

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

## Metrics Completeness

- Step 9 table fields come from benchmark JSON files.
- Step 10 benchmark text blocks come from matching TXT files.
- Speculative acceptance fields are present when vLLM emits `spec_decode_*` metrics.
