# Best Tuned Model Comparison

- Config source: `tuning_winners.csv`
- Source note: Uses the final frozen staged winners from sequential tuning for each model.
- Baseline: shared flattened-sequence linear regression on the same split

## Summary

- Best model by validation MSE: **LSTM** (0.000038705595)
- Best model by test MSE: **Baseline-LR** (0.000018526396)
- Ranking by validation MSE:
  1. LSTM — val MSE 0.000038705595, test MSE 0.000019608429
  2. Baseline-LR — val MSE 0.000043313757, test MSE 0.000018526396
  3. RNN — val MSE 0.000045628437, test MSE 0.000019026347
  4. Transformer — val MSE 0.000048823909, test MSE 0.000019625531
  5. GRU — val MSE 0.000055873064, test MSE 0.000019715399
- Note: This comparison uses the final frozen staged winners from sequential tuning for each model.

## Results

| Model | Hyperparameters | Train MSE | Validation MSE | Test MSE | MAE | Directional Accuracy | Run ID | Config source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| LSTM | `{"batch_size": 32, "data_source": "spy", "hidden": 32, "horizon": 1, "layers": 3, "lr": 0.0005, "target_mode": "next_volatility", "target_smooth_window": 5, "task_id": "next_volatility"}` | 0.000147468283 | 0.000038705595 | 0.000019608429 | 0.002580440852 | 1.000000 | `best_tuned_lstm_comparison-20260328T035502Z` | `tuning_winners.csv` |
| Baseline-LR | `{"flattened_sequence": true, "model": "LinearRegression"}` | 0.000013149738 | 0.000043313757 | 0.000018526396 | 0.002670905284 | 0.997481 | `best_tuned_lstm_comparison-20260328T035502Z-baseline-lr` | `tuning_winners.csv` |
| RNN | `{"batch_size": 128, "data_source": "spy", "hidden": 128, "horizon": 1, "layers": 3, "lr": 0.001, "target_mode": "next_volatility", "target_smooth_window": 5, "task_id": "next_volatility"}` | 0.000019639436 | 0.000045628437 | 0.000019026347 | 0.002689088056 | 1.000000 | `best_tuned_rnn_comparison-20260328T035505Z` | `tuning_winners.csv` |
| Transformer | `{"batch_size": 32, "d_model": 32, "data_source": "spy", "horizon": 1, "lr": 0.001, "nhead": 8, "num_layers": 1, "target_mode": "next_volatility", "target_smooth_window": 5, "task_id": "next_volatility"}` | 0.000030075058 | 0.000048823909 | 0.000019625531 | 0.002876725842 | 0.997481 | `best_tuned_transformer_comparison-20260328T035505Z` | `tuning_winners.csv` |
| GRU | `{"batch_size": 64, "data_source": "spy", "hidden": 32, "horizon": 1, "layers": 2, "lr": 0.001, "target_mode": "next_volatility", "target_smooth_window": 5, "task_id": "next_volatility"}` | 0.000016940588 | 0.000055873064 | 0.000019715399 | 0.002840734856 | 0.998741 | `best_tuned_gru_comparison-20260328T035504Z` | `tuning_winners.csv` |
