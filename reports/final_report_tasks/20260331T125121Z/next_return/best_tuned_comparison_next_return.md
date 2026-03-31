# Best Tuned Model Comparison

- Config source: `tuning_winners.csv`
- Source note: Uses the final frozen staged winners from sequential tuning for each model.
- Baseline: shared flattened-sequence linear regression on the same split

## Summary

- Best model by validation MSE: **LSTM** (0.000243976632)
- Best model by test MSE: **LSTM** (0.000091979055)
- Ranking by validation MSE:
  1. LSTM — val MSE 0.000243976632, test MSE 0.000091979055
  2. GRU — val MSE 0.000248083432, test MSE 0.000092194379
  3. RNN — val MSE 0.000269335408, test MSE 0.000112670023
  4. Transformer — val MSE 0.000281779324, test MSE 0.000122205849
  5. Baseline-LR — val MSE 0.000372807303, test MSE 0.000121759335
- Note: This comparison uses the final frozen staged winners from sequential tuning for each model.

## Results

| Model | Hyperparameters | Train MSE | Validation MSE | Test MSE | MAE | Directional Accuracy | Run ID | Config source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| LSTM | `{"batch_size": 32, "data_source": "spy", "hidden": 128, "horizon": 1, "layers": 3, "lr": 0.0005, "seq_len": 30, "target_mode": "next_return", "target_smooth_window": 1, "task_id": "next_return"}` | 0.000143519229 | 0.000243976632 | 0.000091979055 | 0.006618291718 | 0.527044 | `best_tuned_lstm_comparison-20260331T132130Z` | `tuning_winners.csv` |
| GRU | `{"batch_size": 64, "data_source": "spy", "hidden": 64, "horizon": 1, "layers": 3, "lr": 0.0001, "seq_len": 20, "target_mode": "next_return", "target_smooth_window": 1, "task_id": "next_return"}` | 0.000136734883 | 0.000248083432 | 0.000092194379 | 0.006668725653 | 0.515075 | `best_tuned_gru_comparison-20260331T132131Z` | `tuning_winners.csv` |
| RNN | `{"batch_size": 128, "data_source": "spy", "hidden": 128, "horizon": 1, "layers": 1, "lr": 0.001, "seq_len": 30, "target_mode": "next_return", "target_smooth_window": 1, "task_id": "next_return"}` | 0.000126425098 | 0.000269335408 | 0.000112670023 | 0.007116724006 | 0.477987 | `best_tuned_rnn_comparison-20260331T132131Z` | `tuning_winners.csv` |
| Transformer | `{"batch_size": 32, "d_model": 64, "data_source": "spy", "horizon": 1, "lr": 0.001, "nhead": 4, "num_layers": 2, "seq_len": 30, "target_mode": "next_return", "target_smooth_window": 1, "task_id": "next_return"}` | 0.000303577044 | 0.000281779324 | 0.000122205849 | 0.008317824865 | 0.464151 | `best_tuned_transformer_comparison-20260331T132132Z` | `tuning_winners.csv` |
| Baseline-LR | `{"flattened_sequence": true, "model": "LinearRegression", "seq_len": 30}` | 0.000106213016 | 0.000372807303 | 0.000121759335 | 0.007667290610 | 0.489308 | `best_tuned_lstm_comparison-20260331T132130Z-baseline-lr` | `tuning_winners.csv` |
