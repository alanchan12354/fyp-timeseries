# Best Tuned Model Comparison

- Config source: `tuning_winners.csv`
- Source note: Uses the final frozen staged winners from sequential tuning for each model.
- Baseline: shared flattened-sequence linear regression on the same split

## Summary

- Best model by validation MSE: **GRU** (0.000040914940)
- Best model by test MSE: **GRU** (0.000014938316)
- Ranking by validation MSE:
  1. GRU — val MSE 0.000040914940, test MSE 0.000014938316
  2. LSTM — val MSE 0.000045580974, test MSE 0.000023013823
  3. RNN — val MSE 0.000050419014, test MSE 0.000016501981
  4. Baseline-LR — val MSE 0.000054934276, test MSE 0.000018860735
  5. Transformer — val MSE 0.000079514258, test MSE 0.000034854438
- Note: This comparison uses the final frozen staged winners from sequential tuning for each model.

## Results

| Model | Hyperparameters | Train MSE | Validation MSE | Test MSE | MAE | Directional Accuracy | Run ID | Config source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| GRU | `{"batch_size": 128, "data_source": "spy", "hidden": 128, "horizon": 1, "layers": 3, "lr": 0.001, "target_mode": "next_mean_return", "target_smooth_window": 5, "task_id": "next_mean_return"}` | 0.000021878412 | 0.000040914940 | 0.000014938316 | 0.002903890594 | 0.560453 | `best_tuned_gru_comparison-20260328T041144Z` | `tuning_winners.csv` |
| LSTM | `{"batch_size": 64, "data_source": "spy", "hidden": 128, "horizon": 1, "layers": 3, "lr": 0.0005, "target_mode": "next_mean_return", "target_smooth_window": 5, "task_id": "next_mean_return"}` | 0.000515024951 | 0.000045580974 | 0.000023013823 | 0.003955892446 | 0.385390 | `best_tuned_lstm_comparison-20260328T041143Z` | `tuning_winners.csv` |
| RNN | `{"batch_size": 32, "data_source": "spy", "hidden": 128, "horizon": 1, "layers": 2, "lr": 0.001, "target_mode": "next_mean_return", "target_smooth_window": 5, "task_id": "next_mean_return"}` | 0.000026862826 | 0.000050419014 | 0.000016501981 | 0.002992404367 | 0.539043 | `best_tuned_rnn_comparison-20260328T041145Z` | `tuning_winners.csv` |
| Baseline-LR | `{"flattened_sequence": true, "model": "LinearRegression"}` | 0.000017463835 | 0.000054934276 | 0.000018860735 | 0.003272528679 | 0.506297 | `best_tuned_lstm_comparison-20260328T041143Z-baseline-lr` | `tuning_winners.csv` |
| Transformer | `{"batch_size": 32, "d_model": 32, "data_source": "spy", "horizon": 1, "lr": 0.0005, "nhead": 8, "num_layers": 3, "target_mode": "next_mean_return", "target_smooth_window": 5, "task_id": "next_mean_return"}` | 0.000136094525 | 0.000079514258 | 0.000034854438 | 0.004860837062 | 0.384131 | `best_tuned_transformer_comparison-20260328T041146Z` | `tuning_winners.csv` |
