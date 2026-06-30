# Tuned Benchmark Results

## Best-Run Comparison

| Configuration | Source run | Draft tokens | Output tok/s | Mean TTFT ms | Mean TPOT ms | Acceptance rate | Acceptance length |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline BF16 | `baseline_bf16` | N/A | `838.44` | `659.98` | `6.99` | N/A | `N/A` |
| EAGLE-3 BF16 | `spec_bf16_n1` | 1 | `1201.13` | `126.04` | `5.84` | 37.03% | `1.37` |
| FP8 | `fp8` | N/A | `1069.66` | `593.03` | `5.18` | N/A | `N/A` |
| FP8 + EAGLE-3 | `fp8_spec_n2` | 2 | `1628.42` | `60.09` | `4.48` | 22.67% | `1.45` |

## Threshold Check

| Configuration | Threshold | Observed | Result |
| --- | ---: | ---: | --- |
| EAGLE-3 BF16 | `> 1250 tok/s` | `1201.13 tok/s` | below threshold |
| FP8 | `> 1550 tok/s` | `1069.66 tok/s` | below threshold |
| FP8 + EAGLE-3 | `> 1750 tok/s` | `1628.42 tok/s` | below threshold |

## All Measured Runs

| Run | Group | Draft tokens | Output tok/s | Mean TTFT ms | P99 TTFT ms | Mean TPOT ms | Acceptance rate | Acceptance length |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_bf16` | `baseline_bf16` | N/A | `838.44` | `659.98` | `5993.71` | `6.99` | N/A | `N/A` |
| `fp8` | `fp8` | N/A | `1069.66` | `593.03` | `6054.03` | `5.18` | N/A | `N/A` |
| `fp8_spec_n1` | `fp8_spec` | 1 | `1032.13` | `746.79` | `7344.81` | `4.40` | 35.51% | `1.36` |
| `fp8_spec_n2` | `fp8_spec` | 2 | `1628.42` | `60.09` | `429.56` | `4.48` | 22.67% | `1.45` |
| `fp8_spec_n3` | `fp8_spec` | 3 | `1569.86` | `54.43` | `349.62` | `4.64` | 15.43% | `1.46` |
| `fp8_spec_n4` | `fp8_spec` | 4 | `1562.29` | `63.47` | `449.01` | `4.65` | 15.26% | `1.46` |
| `spec_bf16_n1` | `spec_bf16` | 1 | `1201.13` | `126.04` | `1036.24` | `5.84` | 37.03% | `1.37` |
| `spec_bf16_n2` | `spec_bf16` | 2 | `834.04` | `815.22` | `7962.39` | `5.86` | 21.61% | `1.43` |
| `spec_bf16_n3` | `spec_bf16` | 3 | `1183.20` | `138.17` | `1155.94` | `5.97` | 15.32% | `1.46` |
| `spec_bf16_n4` | `spec_bf16` | 4 | `1170.54` | `77.12` | `533.01` | `6.26` | 11.57% | `1.46` |

## Notebook Text Blocks

### Speculative Decoding Benchmark Results

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

### FP8 Quantization Benchmark Results

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

### FP8 + Speculative Decoding Benchmark Results

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

## Metrics Completeness

- Step 9 table fields come from benchmark JSON files.
- Step 10 benchmark text blocks come from matching TXT files.
- Speculative acceptance fields are present when vLLM emits `spec_decode_*` metrics.
