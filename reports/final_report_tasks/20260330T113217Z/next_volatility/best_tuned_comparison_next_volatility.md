# Best Tuned Model Comparison

- Config source: `tuning_winners.csv`
- Source note: Uses the final frozen staged winners from sequential tuning for each model.
- Baseline: shared flattened-sequence linear regression on the same split

## Summary

- Best model by validation MSE: **LSTM** (0.000040374842)
- Best model by test MSE: **Baseline-LR** (0.000018526508)
- Ranking by validation MSE:
  1. LSTM — val MSE 0.000040374842, test MSE 0.000022292839
  2. RNN — val MSE 0.000041938150, test MSE 0.000022666456
  3. Baseline-LR — val MSE 0.000043313563, test MSE 0.000018526508
  4. GRU — val MSE 0.000044224099, test MSE 0.000022269160
  5. Transformer — val MSE 0.000054075228, test MSE 0.000026333999
- Note: This comparison uses the final frozen staged winners from sequential tuning for each model.

## Results

| Model | Hyperparameters | Train MSE | Validation MSE | Test MSE | MAE | Directional Accuracy | Run ID | Config source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| LSTM | `{"batch_size": 64, "data_source": "spy", "hidden": 32, "horizon": 1, "layers": 2, "lr": 0.0005, "target_mode": "next_volatility", "target_smooth_window": 5, "task_id": "next_volatility"}` | 0.000019763692 | 0.000040374842 | 0.000022292839 | 0.002757049405 | 1.000000 | `best_tuned_lstm_comparison-20260330T121153Z` | `tuning_winners.csv` |
| RNN | `{"batch_size": 32, "data_source": "spy", "hidden": 64, "horizon": 1, "layers": 2, "lr": 0.001, "target_mode": "next_volatility", "target_smooth_window": 5, "task_id": "next_volatility"}` | 0.000016571037 | 0.000041938150 | 0.000022666456 | 0.002918791129 | 1.000000 | `best_tuned_rnn_comparison-20260330T121155Z` | `tuning_winners.csv` |
| Baseline-LR | `{"flattened_sequence": true, "model": "LinearRegression"}` | 0.000013149623 | 0.000043313563 | 0.000018526508 | 0.002670911372 | 0.997481 | `best_tuned_lstm_comparison-20260330T121153Z-baseline-lr` | `tuning_winners.csv` |
| GRU | `{"batch_size": 64, "data_source": "spy", "hidden": 64, "horizon": 1, "layers": 2, "lr": 0.001, "target_mode": "next_volatility", "target_smooth_window": 5, "task_id": "next_volatility"}` | 0.000012784256 | 0.000044224099 | 0.000022269160 | 0.002828644341 | 0.998741 | `best_tuned_gru_comparison-20260330T121154Z` | `tuning_winners.csv` |
| Transformer | `{"batch_size": 32, "d_model": 32, "data_source": "spy", "horizon": 1, "lr": 0.0005, "nhead": 2, "num_layers": 1, "target_mode": "next_volatility", "target_smooth_window": 5, "task_id": "next_volatility"}` | 0.000052536109 | 0.000054075228 | 0.000026333999 | 0.003628298063 | 0.998741 | `best_tuned_transformer_comparison-20260330T121156Z` | `tuning_winners.csv` |
