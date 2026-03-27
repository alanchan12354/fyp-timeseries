# Best Tuned Model Comparison

- Config source: `tuning_winners.csv`
- Source note: Uses the final frozen staged winners from sequential tuning for each model.
- Baseline: shared flattened-sequence linear regression on the same split

## Summary

- Best model by validation MSE: **GRU** (0.000253892693)
- Best model by test MSE: **GRU** (0.000094600728)
- Ranking by validation MSE:
  1. GRU — val MSE 0.000253892693, test MSE 0.000094600728
  2. LSTM — val MSE 0.000259542162, test MSE 0.000097805569
  3. RNN — val MSE 0.000270551936, test MSE 0.000103296050
  4. Transformer — val MSE 0.000292985925, test MSE 0.000151831584
  5. Baseline-LR — val MSE 0.000372732762, test MSE 0.000121767021
- Note: This comparison uses the final frozen staged winners from sequential tuning for each model.

## Results

| Model | Hyperparameters | Train MSE | Validation MSE | Test MSE | MAE | Directional Accuracy | Run ID | Config source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| GRU | `{"batch_size": 32, "hidden": 128, "layers": 2, "lr": 0.0001}` | 0.000178405963 | 0.000253892693 | 0.000094600728 | 0.007047767094 | 0.472956 | `best_tuned_gru_comparison-20260327T185634Z` | `tuning_winners.csv` |
| LSTM | `{"batch_size": 128, "hidden": 32, "layers": 2, "lr": 0.0001}` | 0.000138846290 | 0.000259542162 | 0.000097805569 | 0.007005118473 | 0.493082 | `best_tuned_lstm_comparison-20260327T185633Z` | `tuning_winners.csv` |
| RNN | `{"batch_size": 128, "hidden": 128, "layers": 2, "lr": 0.001}` | 0.000157105655 | 0.000270551936 | 0.000103296050 | 0.007227437098 | 0.470440 | `best_tuned_rnn_comparison-20260327T185634Z` | `tuning_winners.csv` |
| Transformer | `{"batch_size": 32, "d_model": 64, "lr": 0.001, "nhead": 4, "num_layers": 1}` | 0.000247158592 | 0.000292985925 | 0.000151831584 | 0.008503559645 | 0.467925 | `best_tuned_transformer_comparison-20260327T185635Z` | `tuning_winners.csv` |
| Baseline-LR | `{"flattened_sequence": true, "model": "LinearRegression"}` | 0.000106202266 | 0.000372732762 | 0.000121767021 | 0.007670198590 | 0.489308 | `best_tuned_lstm_comparison-20260327T185633Z-baseline-lr` | `tuning_winners.csv` |
