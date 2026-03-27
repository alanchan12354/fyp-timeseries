# Best Tuned Model Comparison

- Config source: `tuning_winners.csv`
- Source note: Uses the final frozen staged winners from sequential tuning for each model.
- Baseline: shared flattened-sequence linear regression on the same split

## Summary

- Best model by validation MSE: **RNN** (0.000254363828)
- Best model by test MSE: **GRU** (0.000095761738)
- Ranking by validation MSE:
  1. RNN — val MSE 0.000254363828, test MSE 0.000099253142
  2. GRU — val MSE 0.000258048193, test MSE 0.000095761738
  3. LSTM — val MSE 0.000296664460, test MSE 0.000099745550
  4. Transformer — val MSE 0.000311149108, test MSE 0.000134892602
  5. Baseline-LR — val MSE 0.000372866961, test MSE 0.000121646610
- Note: This comparison uses the final frozen staged winners from sequential tuning for each model.

## Results

| Model | Hyperparameters | Train MSE | Validation MSE | Test MSE | MAE | Directional Accuracy | Run ID | Config source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| RNN | `{"batch_size": 128, "hidden": 64, "layers": 3, "lr": 0.0001}` | 0.000153695291 | 0.000254363828 | 0.000099253142 | 0.006865587359 | 0.538365 | `best_tuned_rnn_comparison-20260327T154333Z` | `tuning_winners.csv` |
| GRU | `{"batch_size": 128, "hidden": 128, "layers": 3, "lr": 0.0001}` | 0.000226289238 | 0.000258048193 | 0.000095761738 | 0.006817974501 | 0.520755 | `best_tuned_gru_comparison-20260327T154333Z` | `tuning_winners.csv` |
| LSTM | `{"batch_size": 64, "hidden": 32, "layers": 2, "lr": 0.0001}` | 0.000134950773 | 0.000296664460 | 0.000099745550 | 0.006830783612 | 0.489308 | `best_tuned_lstm_comparison-20260327T154332Z` | `tuning_winners.csv` |
| Transformer | `{"batch_size": 64, "d_model": 32, "lr": 0.001, "nhead": 4, "num_layers": 1}` | 0.000224466691 | 0.000311149108 | 0.000134892602 | 0.007759623363 | 0.503145 | `best_tuned_transformer_comparison-20260327T154334Z` | `tuning_winners.csv` |
| Baseline-LR | `{"flattened_sequence": true, "model": "LinearRegression"}` | 0.000106217107 | 0.000372866961 | 0.000121646610 | 0.007659596800 | 0.489308 | `best_tuned_lstm_comparison-20260327T154332Z-baseline-lr` | `tuning_winners.csv` |
