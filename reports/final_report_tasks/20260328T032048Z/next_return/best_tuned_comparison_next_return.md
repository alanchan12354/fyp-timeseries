# Best Tuned Model Comparison

- Config source: `tuning_winners.csv`
- Source note: Uses the final frozen staged winners from sequential tuning for each model.
- Baseline: shared flattened-sequence linear regression on the same split

## Summary

- Best model by validation MSE: **LSTM** (0.000244726972)
- Best model by test MSE: **LSTM** (0.000092038579)
- Ranking by validation MSE:
  1. LSTM — val MSE 0.000244726972, test MSE 0.000092038579
  2. GRU — val MSE 0.000250613964, test MSE 0.000093188864
  3. RNN — val MSE 0.000271178657, test MSE 0.000107602798
  4. Transformer — val MSE 0.000312326921, test MSE 0.000166873288
  5. Baseline-LR — val MSE 0.000372732450, test MSE 0.000121832244
- Note: This comparison uses the final frozen staged winners from sequential tuning for each model.

## Results

| Model | Hyperparameters | Train MSE | Validation MSE | Test MSE | MAE | Directional Accuracy | Run ID | Config source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| LSTM | `{"batch_size": 32, "data_source": "spy", "hidden": 32, "horizon": 1, "layers": 3, "lr": 0.001, "target_mode": "next_return", "target_smooth_window": 1, "task_id": "next_return"}` | 0.000394720304 | 0.000244726972 | 0.000092038579 | 0.006588046858 | 0.534591 | `best_tuned_lstm_comparison-20260328T034106Z` | `tuning_winners.csv` |
| GRU | `{"batch_size": 64, "data_source": "spy", "hidden": 128, "horizon": 1, "layers": 3, "lr": 0.001, "target_mode": "next_return", "target_smooth_window": 1, "task_id": "next_return"}` | 0.000599381747 | 0.000250613964 | 0.000093188864 | 0.006742425480 | 0.479245 | `best_tuned_gru_comparison-20260328T034107Z` | `tuning_winners.csv` |
| RNN | `{"batch_size": 64, "data_source": "spy", "hidden": 128, "horizon": 1, "layers": 3, "lr": 0.0005, "target_mode": "next_return", "target_smooth_window": 1, "task_id": "next_return"}` | 0.002351175232 | 0.000271178657 | 0.000107602798 | 0.007438386082 | 0.500629 | `best_tuned_rnn_comparison-20260328T034108Z` | `tuning_winners.csv` |
| Transformer | `{"batch_size": 128, "d_model": 32, "data_source": "spy", "horizon": 1, "lr": 0.0005, "nhead": 4, "num_layers": 1, "target_mode": "next_return", "target_smooth_window": 1, "task_id": "next_return"}` | 0.000304143632 | 0.000312326921 | 0.000166873288 | 0.007900764138 | 0.489308 | `best_tuned_transformer_comparison-20260328T034109Z` | `tuning_winners.csv` |
| Baseline-LR | `{"flattened_sequence": true, "model": "LinearRegression"}` | 0.000106202225 | 0.000372732450 | 0.000121832244 | 0.007672863970 | 0.489308 | `best_tuned_lstm_comparison-20260328T034106Z-baseline-lr` | `tuning_winners.csv` |
