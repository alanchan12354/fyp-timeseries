# Best Tuned Model Comparison

- Config source: `tuning_winners.csv`
- Source note: Uses the final frozen staged winners from sequential tuning for each model.
- Baseline: shared flattened-sequence linear regression on the same split

## Summary

- Best model by validation MSE: **LSTM** (0.000039658958)
- Best model by test MSE: **GRU** (0.000015519119)
- Ranking by validation MSE:
  1. LSTM — val MSE 0.000039658958, test MSE 0.000016129959
  2. GRU — val MSE 0.000044157480, test MSE 0.000015519119
  3. RNN — val MSE 0.000050111856, test MSE 0.000016559785
  4. Transformer — val MSE 0.000057541246, test MSE 0.000025485584
  5. Baseline-LR — val MSE 0.000061308113, test MSE 0.000021937077
- Note: This comparison uses the final frozen staged winners from sequential tuning for each model.

## Results

| Model | Hyperparameters | Train MSE | Validation MSE | Test MSE | MAE | Directional Accuracy | Run ID | Config source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| LSTM | `{"batch_size": 128, "data_source": "spy", "hidden": 64, "horizon": 1, "layers": 3, "lr": 0.001, "seq_len": 60, "target_mode": "next_mean_return", "target_smooth_window": 5, "task_id": "next_mean_return"}` | 0.000021139361 | 0.000039658958 | 0.000016129959 | 0.002939243683 | 0.611392 | `best_tuned_lstm_comparison-20260331T135921Z` | `tuning_winners.csv` |
| GRU | `{"batch_size": 128, "data_source": "spy", "hidden": 128, "horizon": 1, "layers": 3, "lr": 0.001, "seq_len": 30, "target_mode": "next_mean_return", "target_smooth_window": 5, "task_id": "next_mean_return"}` | 0.000022154046 | 0.000044157480 | 0.000015519119 | 0.002935526963 | 0.568553 | `best_tuned_gru_comparison-20260331T135921Z` | `tuning_winners.csv` |
| RNN | `{"batch_size": 32, "data_source": "spy", "hidden": 128, "horizon": 1, "layers": 3, "lr": 0.0005, "seq_len": 60, "target_mode": "next_mean_return", "target_smooth_window": 5, "task_id": "next_mean_return"}` | 0.000023703562 | 0.000050111856 | 0.000016559785 | 0.003004477720 | 0.620253 | `best_tuned_rnn_comparison-20260331T135922Z` | `tuning_winners.csv` |
| Transformer | `{"batch_size": 32, "d_model": 64, "data_source": "spy", "horizon": 1, "lr": 0.001, "nhead": 8, "num_layers": 1, "seq_len": 30, "target_mode": "next_mean_return", "target_smooth_window": 5, "task_id": "next_mean_return"}` | 0.000059135441 | 0.000057541246 | 0.000025485584 | 0.003595452549 | 0.542138 | `best_tuned_transformer_comparison-20260331T135922Z` | `tuning_winners.csv` |
| Baseline-LR | `{"flattened_sequence": true, "model": "LinearRegression", "seq_len": 60}` | 0.000015308183 | 0.000061308113 | 0.000021937077 | 0.003475729084 | 0.525316 | `best_tuned_lstm_comparison-20260331T135921Z-baseline-lr` | `tuning_winners.csv` |
