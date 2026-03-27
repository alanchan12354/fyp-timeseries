# Best Tuned Model Comparison

- Config source: `tuning_winners.csv`
- Source note: Uses the final frozen staged winners from sequential tuning for each model.
- Baseline: shared flattened-sequence linear regression on the same split

## Summary

- Best model by validation MSE: **LSTM** (0.000242262342)
- Best model by test MSE: **LSTM** (0.000090180009)
- Ranking by validation MSE:
  1. LSTM — val MSE 0.000242262342, test MSE 0.000090180009
  2. RNN — val MSE 0.000268744276, test MSE 0.000106678133
  3. GRU — val MSE 0.000274819442, test MSE 0.000102688777
  4. Transformer — val MSE 0.000343755856, test MSE 0.000135015143
  5. Baseline-LR — val MSE 0.000372732378, test MSE 0.000121688135
- Note: This comparison uses the final frozen staged winners from sequential tuning for each model.

## Results

| Model | Hyperparameters | Train MSE | Validation MSE | Test MSE | MAE | Directional Accuracy | Run ID | Config source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| LSTM | `{"batch_size": 32, "hidden": 64, "layers": 3, "lr": 0.001}` | 0.000141297149 | 0.000242262342 | 0.000090180009 | 0.006555008590 | 0.530818 | `best_tuned_lstm_comparison-20260327T172208Z` | `tuning_winners.csv` |
| RNN | `{"batch_size": 32, "hidden": 128, "layers": 3, "lr": 0.001}` | 0.000157064406 | 0.000268744276 | 0.000106678133 | 0.007024721417 | 0.523270 | `best_tuned_rnn_comparison-20260327T172209Z` | `tuning_winners.csv` |
| GRU | `{"batch_size": 64, "hidden": 128, "layers": 2, "lr": 0.001}` | 0.000133031097 | 0.000274819442 | 0.000102688777 | 0.007129939119 | 0.527044 | `best_tuned_gru_comparison-20260327T172209Z` | `tuning_winners.csv` |
| Transformer | `{"batch_size": 32, "d_model": 32, "lr": 0.0005, "nhead": 8, "num_layers": 1}` | 0.000141574063 | 0.000343755856 | 0.000135015143 | 0.007856514909 | 0.545912 | `best_tuned_transformer_comparison-20260327T172210Z` | `tuning_winners.csv` |
| Baseline-LR | `{"flattened_sequence": true, "model": "LinearRegression"}` | 0.000106202369 | 0.000372732378 | 0.000121688135 | 0.007666126364 | 0.489308 | `best_tuned_lstm_comparison-20260327T172208Z-baseline-lr` | `tuning_winners.csv` |
