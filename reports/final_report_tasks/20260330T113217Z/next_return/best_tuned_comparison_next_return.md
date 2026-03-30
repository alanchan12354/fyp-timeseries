# Best Tuned Model Comparison

- Config source: `tuning_winners.csv`
- Source note: Uses the final frozen staged winners from sequential tuning for each model.
- Baseline: shared flattened-sequence linear regression on the same split

## Summary

- Best model by validation MSE: **LSTM** (0.000242161872)
- Best model by test MSE: **LSTM** (0.000090147878)
- Ranking by validation MSE:
  1. LSTM — val MSE 0.000242161872, test MSE 0.000090147878
  2. GRU — val MSE 0.000245732549, test MSE 0.000092643757
  3. Transformer — val MSE 0.000251746807, test MSE 0.000098342680
  4. RNN — val MSE 0.000253167786, test MSE 0.000109641114
  5. Baseline-LR — val MSE 0.000372732022, test MSE 0.000121832247
- Note: This comparison uses the final frozen staged winners from sequential tuning for each model.

## Results

| Model | Hyperparameters | Train MSE | Validation MSE | Test MSE | MAE | Directional Accuracy | Run ID | Config source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| LSTM | `{"batch_size": 32, "data_source": "spy", "hidden": 128, "horizon": 1, "layers": 3, "lr": 0.0005, "target_mode": "next_return", "target_smooth_window": 1, "task_id": "next_return"}` | 0.000200634191 | 0.000242161872 | 0.000090147878 | 0.006518285364 | 0.554717 | `best_tuned_lstm_comparison-20260330T115620Z` | `tuning_winners.csv` |
| GRU | `{"batch_size": 64, "data_source": "spy", "hidden": 64, "horizon": 1, "layers": 3, "lr": 0.0001, "target_mode": "next_return", "target_smooth_window": 1, "task_id": "next_return"}` | 0.000135778153 | 0.000245732549 | 0.000092643757 | 0.006751110894 | 0.503145 | `best_tuned_gru_comparison-20260330T115621Z` | `tuning_winners.csv` |
| Transformer | `{"batch_size": 32, "d_model": 64, "data_source": "spy", "horizon": 1, "lr": 0.001, "nhead": 8, "num_layers": 2, "target_mode": "next_return", "target_smooth_window": 1, "task_id": "next_return"}` | 0.000201245450 | 0.000251746807 | 0.000098342680 | 0.006955591990 | 0.509434 | `best_tuned_transformer_comparison-20260330T115623Z` | `tuning_winners.csv` |
| RNN | `{"batch_size": 32, "data_source": "spy", "hidden": 32, "horizon": 1, "layers": 3, "lr": 0.0001, "target_mode": "next_return", "target_smooth_window": 1, "task_id": "next_return"}` | 0.000127583902 | 0.000253167786 | 0.000109641114 | 0.006968794618 | 0.527044 | `best_tuned_rnn_comparison-20260330T115622Z` | `tuning_winners.csv` |
| Baseline-LR | `{"flattened_sequence": true, "model": "LinearRegression"}` | 0.000106202342 | 0.000372732022 | 0.000121832247 | 0.007672862981 | 0.489308 | `best_tuned_lstm_comparison-20260330T115620Z-baseline-lr` | `tuning_winners.csv` |
