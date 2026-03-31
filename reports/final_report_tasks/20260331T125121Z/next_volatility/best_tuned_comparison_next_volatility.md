# Best Tuned Model Comparison

- Config source: `tuning_winners.csv`
- Source note: Uses the final frozen staged winners from sequential tuning for each model.
- Baseline: shared flattened-sequence linear regression on the same split

## Summary

- Best model by validation MSE: **GRU** (0.000038570659)
- Best model by test MSE: **Baseline-LR** (0.000017733482)
- Ranking by validation MSE:
  1. GRU — val MSE 0.000038570659, test MSE 0.000019411440
  2. LSTM — val MSE 0.000038970613, test MSE 0.000019779773
  3. Baseline-LR — val MSE 0.000041327864, test MSE 0.000017733482
  4. RNN — val MSE 0.000043288692, test MSE 0.000021044572
  5. Transformer — val MSE 0.000051374823, test MSE 0.000021407383
- Note: This comparison uses the final frozen staged winners from sequential tuning for each model.

## Results

| Model | Hyperparameters | Train MSE | Validation MSE | Test MSE | MAE | Directional Accuracy | Run ID | Config source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| GRU | `{"batch_size": 64, "data_source": "spy", "hidden": 128, "horizon": 1, "layers": 3, "lr": 0.001, "seq_len": 30, "target_mode": "next_volatility", "target_smooth_window": 5, "task_id": "next_volatility"}` | 0.000363663101 | 0.000038570659 | 0.000019411440 | 0.002730968597 | 1.000000 | `best_tuned_gru_comparison-20260331T134031Z` | `tuning_winners.csv` |
| LSTM | `{"batch_size": 32, "data_source": "spy", "hidden": 64, "horizon": 1, "layers": 3, "lr": 0.0001, "seq_len": 20, "target_mode": "next_volatility", "target_smooth_window": 5, "task_id": "next_volatility"}` | 0.000020813818 | 0.000038970613 | 0.000019779773 | 0.002696220408 | 1.000000 | `best_tuned_lstm_comparison-20260331T134030Z` | `tuning_winners.csv` |
| Baseline-LR | `{"flattened_sequence": true, "model": "LinearRegression", "seq_len": 20}` | 0.000013972190 | 0.000041327864 | 0.000017733482 | 0.002563732058 | 1.000000 | `best_tuned_lstm_comparison-20260331T134030Z-baseline-lr` | `tuning_winners.csv` |
| RNN | `{"batch_size": 64, "data_source": "spy", "hidden": 64, "horizon": 1, "layers": 3, "lr": 0.001, "seq_len": 30, "target_mode": "next_volatility", "target_smooth_window": 5, "task_id": "next_volatility"}` | 0.000015293320 | 0.000043288692 | 0.000021044572 | 0.002881411629 | 1.000000 | `best_tuned_rnn_comparison-20260331T134031Z` | `tuning_winners.csv` |
| Transformer | `{"batch_size": 32, "d_model": 64, "data_source": "spy", "horizon": 1, "lr": 0.001, "nhead": 4, "num_layers": 1, "seq_len": 60, "target_mode": "next_volatility", "target_smooth_window": 5, "task_id": "next_volatility"}` | 0.000036315173 | 0.000051374823 | 0.000021407383 | 0.003165709433 | 0.996203 | `best_tuned_transformer_comparison-20260331T134032Z` | `tuning_winners.csv` |
