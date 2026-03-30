# Best Tuned Model Comparison

- Config source: `tuning_winners.csv`
- Source note: Uses the final frozen staged winners from sequential tuning for each model.
- Baseline: shared flattened-sequence linear regression on the same split

## Summary

- Best model by validation MSE: **LSTM** (0.000040735964)
- Best model by test MSE: **RNN** (0.000015500109)
- Ranking by validation MSE:
  1. LSTM — val MSE 0.000040735964, test MSE 0.000017453160
  2. RNN — val MSE 0.000045057106, test MSE 0.000015500109
  3. GRU — val MSE 0.000046934604, test MSE 0.000019459630
  4. Baseline-LR — val MSE 0.000054934316, test MSE 0.000018860773
  5. Transformer — val MSE 0.000055404870, test MSE 0.000021720074
- Note: This comparison uses the final frozen staged winners from sequential tuning for each model.

## Results

| Model | Hyperparameters | Train MSE | Validation MSE | Test MSE | MAE | Directional Accuracy | Run ID | Config source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| LSTM | `{"batch_size": 64, "data_source": "spy", "hidden": 128, "horizon": 1, "layers": 2, "lr": 0.001, "target_mode": "next_mean_return", "target_smooth_window": 5, "task_id": "next_mean_return"}` | 0.000357732724 | 0.000040735964 | 0.000017453160 | 0.003108814819 | 0.591940 | `best_tuned_lstm_comparison-20260330T122945Z` | `tuning_winners.csv` |
| RNN | `{"batch_size": 64, "data_source": "spy", "hidden": 128, "horizon": 1, "layers": 3, "lr": 0.001, "target_mode": "next_mean_return", "target_smooth_window": 5, "task_id": "next_mean_return"}` | 0.000019904772 | 0.000045057106 | 0.000015500109 | 0.002952458373 | 0.523929 | `best_tuned_rnn_comparison-20260330T122947Z` | `tuning_winners.csv` |
| GRU | `{"batch_size": 32, "data_source": "spy", "hidden": 128, "horizon": 1, "layers": 3, "lr": 0.0005, "target_mode": "next_mean_return", "target_smooth_window": 5, "task_id": "next_mean_return"}` | 0.000189130039 | 0.000046934604 | 0.000019459630 | 0.003542164189 | 0.411839 | `best_tuned_gru_comparison-20260330T122946Z` | `tuning_winners.csv` |
| Baseline-LR | `{"flattened_sequence": true, "model": "LinearRegression"}` | 0.000017463843 | 0.000054934316 | 0.000018860773 | 0.003272527533 | 0.506297 | `best_tuned_lstm_comparison-20260330T122945Z-baseline-lr` | `tuning_winners.csv` |
| Transformer | `{"batch_size": 32, "d_model": 64, "data_source": "spy", "horizon": 1, "lr": 0.0005, "nhead": 4, "num_layers": 3, "target_mode": "next_mean_return", "target_smooth_window": 5, "task_id": "next_mean_return"}` | 0.000101774252 | 0.000055404870 | 0.000021720074 | 0.003435013883 | 0.574307 | `best_tuned_transformer_comparison-20260330T122948Z` | `tuning_winners.csv` |
