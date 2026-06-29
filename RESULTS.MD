# Results For Tasks 1-2

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
