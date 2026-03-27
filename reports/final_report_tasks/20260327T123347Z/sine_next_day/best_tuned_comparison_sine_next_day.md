# Best Tuned Model Comparison

- Config source: `tuning_winners.csv`
- Source note: Uses the final frozen staged winners from sequential tuning for each model.
- Baseline: shared flattened-sequence linear regression on the same split

## Summary

- Best model by validation MSE: **RNN** (0.000259016542)
- Best model by test MSE: **GRU** (0.000097735734)
- Ranking by validation MSE:
  1. RNN — val MSE 0.000259016542, test MSE 0.000099158839
  2. GRU — val MSE 0.000259292887, test MSE 0.000097735734
  3. LSTM — val MSE 0.000264411018, test MSE 0.000098558980
  4. Transformer — val MSE 0.000287104353, test MSE 0.000118950178
  5. Baseline-LR — val MSE 0.000372731954, test MSE 0.000121641229
- Note: This comparison uses the final frozen staged winners from sequential tuning for each model.

## Results

| Model | Hyperparameters | Train MSE | Validation MSE | Test MSE | MAE | Directional Accuracy | Run ID | Config source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| RNN | `{"batch_size": 32, "hidden": 64, "layers": 2, "lr": 0.0005}` | 0.000126671498 | 0.000259016542 | 0.000099158839 | 0.006949475726 | 0.522013 | `best_tuned_rnn_comparison-20260327T141401Z` | `tuning_winners.csv` |
| GRU | `{"batch_size": 32, "hidden": 64, "layers": 2, "lr": 0.001}` | 0.000126728809 | 0.000259292887 | 0.000097735734 | 0.006911346593 | 0.504403 | `best_tuned_gru_comparison-20260327T141400Z` | `tuning_winners.csv` |
| LSTM | `{"batch_size": 64, "hidden": 128, "layers": 1, "lr": 0.001}` | 0.000153091428 | 0.000264411018 | 0.000098558980 | 0.006878868678 | 0.511950 | `best_tuned_lstm_comparison-20260327T141400Z` | `tuning_winners.csv` |
| Transformer | `{"batch_size": 32, "d_model": 32, "lr": 0.0005, "nhead": 2, "num_layers": 1}` | 0.000109987875 | 0.000287104353 | 0.000118950178 | 0.007219530008 | 0.494340 | `best_tuned_transformer_comparison-20260327T141401Z` | `tuning_winners.csv` |
| Baseline-LR | `{"flattened_sequence": true, "model": "LinearRegression"}` | 0.000106202370 | 0.000372731954 | 0.000121641229 | 0.007662594471 | 0.489308 | `best_tuned_lstm_comparison-20260327T141400Z-baseline-lr` | `tuning_winners.csv` |
