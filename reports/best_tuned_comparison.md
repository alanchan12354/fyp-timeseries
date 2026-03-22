# Best Tuned Model Comparison

- Config source: `tuning_winners.csv`
- Source note: Uses the final frozen staged winners from sequential tuning for each model.
- Baseline: shared flattened-sequence linear regression on the same split

## Summary

- Best model by validation MSE: **LSTM** (0.000135595361)
- Best model by test MSE: **LSTM** (0.000098718767)
- Ranking by validation MSE:
  1. LSTM — val MSE 0.000135595361, test MSE 0.000098718767
  2. Baseline-LR — val MSE 0.000137576658, test MSE 0.000100075212
  3. GRU — val MSE 0.000142753908, test MSE 0.000100299512
  4. RNN — val MSE 0.000145848553, test MSE 0.000102792619
  5. Transformer — val MSE 0.000157517732, test MSE 0.000133374386
- Note: This comparison uses the final frozen staged winners from sequential tuning for each model.

## Results

| Model | Hyperparameters | Train MSE | Validation MSE | Test MSE | MAE | Directional Accuracy | Run ID | Config source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| LSTM | `{"batch_size": 32, "hidden": 128, "layers": 1, "lr": 0.0005}` | 0.000311817693 | 0.000135595361 | 0.000098718767 | 0.006570027628 | 0.520593 | `best_tuned_lstm_comparison-20260320T093214Z` | `tuning_winners.csv` |
| Baseline-LR | `{"flattened_sequence": true, "model": "LinearRegression"}` | 0.000114152675 | 0.000137576658 | 0.000100075212 | 0.006655243618 | 0.527183 | `best_tuned_lstm_comparison-20260320T093214Z-baseline-lr` | `tuning_winners.csv` |
| GRU | `{"batch_size": 64, "hidden": 32, "layers": 3, "lr": 0.001}` | 0.000118986487 | 0.000142753908 | 0.000100299512 | 0.006599891473 | 0.570016 | `best_tuned_gru_comparison-20260320T093215Z` | `tuning_winners.csv` |
| RNN | `{"batch_size": 32, "hidden": 64, "layers": 2, "lr": 0.0001}` | 0.000254451150 | 0.000145848553 | 0.000102792619 | 0.006844040525 | 0.472817 | `best_tuned_rnn_comparison-20260320T093215Z` | `tuning_winners.csv` |
| Transformer | `{"batch_size": 32, "d_model": 32, "lr": 0.001, "nhead": 8, "num_layers": 1}` | 0.000258977049 | 0.000157517732 | 0.000133374386 | 0.007735318113 | 0.542010 | `best_tuned_transformer_comparison-20260320T093216Z` | `tuning_winners.csv` |
