# Rerun Analysis Scripts

These scripts automate the tuned rerun path from the updated instructions and save new outputs under:

```text
analysis_runs/$RUN_ID/
```

If `RUN_ID` is not set, the scripts use:

```text
analysis_runs/tuned_latest/
```

Set a run id once in every terminal that participates in the rerun:

```bash
export RUN_ID="tuned_$(date +%Y%m%d_%H%M)"
```

## Fast Path: Benchmark Tuning First

Use this before regenerating hidden states or retraining. It checks the FP8 model and reruns the serving benchmark with higher load, warmups, fixed output lengths, and concurrency sweeps.

```bash
cd /home/niko/neb/spec_dec
export RUN_ID="tuned_$(date +%Y%m%d_%H%M)"

bash scripts/rerun_analysis/task3_quantize_or_verify_fp8.sh
bash scripts/rerun_analysis/task4_benchmark_sweep.sh
bash scripts/rerun_analysis/task5_collect_final_artifacts.sh
```

Main outputs:

```text
analysis_runs/$RUN_ID/benchmarks/results/
analysis_runs/$RUN_ID/metrics/RESULTS_TUNED.md
analysis_runs/$RUN_ID/metrics/task4_metrics.json
analysis_runs/$RUN_ID/metrics/task4_metrics.csv
```

To shorten the benchmark while experimenting:

```bash
export SWEEP_CONCURRENCY="16 24"
export BF16_SPEC_TOKENS="1 3"
export FP8_SPEC_TOKENS="2 3"
bash scripts/rerun_analysis/task4_benchmark_sweep.sh
```

To run only FP8 + speculative:

```bash
RUN_BASELINE=0 RUN_BF16_SPEC=0 RUN_FP8=0 RUN_FP8_SPEC=1 \
  bash scripts/rerun_analysis/task4_benchmark_sweep.sh
```

## Full Path: Stronger Draft Head

Use this if benchmark tuning still misses the BF16 speculative target and the acceptance metrics are low.

Terminal A, start the hidden-state server:

```bash
cd /home/niko/neb/spec_dec
export RUN_ID="tuned_$(date +%Y%m%d_%H%M)"
bash scripts/rerun_analysis/task1_start_hidden_state_server.sh
```

Terminal B, generate a larger 5k-sample hidden-state cache:

```bash
cd /home/niko/neb/spec_dec
export RUN_ID="<same value as Terminal A>"
bash scripts/rerun_analysis/task1_generate_hidden_states.sh
```

Stop Terminal A after hidden-state generation finishes.

Then train the stronger draft head and rerun benchmarks:

```bash
bash scripts/rerun_analysis/task2_train_stronger_eagle3.sh
bash scripts/rerun_analysis/task3_quantize_or_verify_fp8.sh
bash scripts/rerun_analysis/task4_benchmark_sweep.sh
bash scripts/rerun_analysis/task5_collect_final_artifacts.sh
```

`task4_benchmark_sweep.sh` automatically uses the best checkpoint from:

```text
analysis_runs/$RUN_ID/metrics/task2_best_checkpoint.json
```

You can override any default:

```bash
export MAX_SAMPLES=5000
export SEQ_LENGTH=2048
export TRAIN_EPOCHS=8
export DRAFT_VOCAB_SIZE=32000
export SWEEP_CONCURRENCY="8 16 24 32"
```

## Script Map

- `task1_start_hidden_state_server.sh`: start vLLM with hidden-state extraction.
- `task1_generate_hidden_states.sh`: preprocess ShareGPT data and generate offline hidden states.
- `task2_train_stronger_eagle3.sh`: train a stronger EAGLE-3 draft head and summarize best checkpoint.
- `task3_quantize_or_verify_fp8.sh`: verify existing FP8 model, or rebuild with `REBUILD_FP8=1`.
- `task4_benchmark_sweep.sh`: run baseline, BF16 speculative, FP8, and FP8 speculative benchmark sweeps.
- `task5_collect_final_artifacts.sh`: aggregate metrics and notebook-ready benchmark blocks.
- `collect_metrics.py`: parse benchmark JSON/TXT files into markdown, JSON, and CSV summaries.
