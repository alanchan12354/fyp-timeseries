# Best Tuned Model Comparison

- Config source: `tuning_winners.csv`
- Source note: Uses the final frozen staged winners from sequential tuning for each model.
- Baseline: shared flattened-sequence linear regression on the same split

## Summary

- Best model by validation MSE: **GRU** (0.000038570182)
- Best model by test MSE: **Baseline-LR** (0.000017797154)
- Ranking by validation MSE:
  1. GRU — val MSE 0.000038570182, test MSE 0.000019462334
  2. LSTM — val MSE 0.000038970993, test MSE 0.000019850624
  3. Baseline-LR — val MSE 0.000041327857, test MSE 0.000017797154
  4. RNN — val MSE 0.000043284226, test MSE 0.000021104879
  5. Transformer — val MSE 0.000049758833, test MSE 0.000021354819
- Note: This comparison uses the final frozen staged winners from sequential tuning for each model.

## Results

| Model | Hyperparameters | Train MSE | Validation MSE | Test MSE | MAE | Directional Accuracy | Run ID | Config source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| GRU | `{"batch_size": 64, "data_source": "spy", "hidden": 128, "horizon": 1, "layers": 3, "lr": 0.001, "seq_len": 30, "target_mode": "next_volatility", "target_smooth_window": 5, "task_id": "next_volatility"}` | 0.000363653080 | 0.000038570182 | 0.000019462334 | 0.002737188849 | 1.000000 | `best_tuned_gru_comparison-20260331T201253Z` | `tuning_winners.csv` |
| LSTM | `{"batch_size": 32, "data_source": "spy", "hidden": 64, "horizon": 1, "layers": 3, "lr": 0.0001, "seq_len": 20, "target_mode": "next_volatility", "target_smooth_window": 5, "task_id": "next_volatility"}` | 0.000020813744 | 0.000038970993 | 0.000019850624 | 0.002702610239 | 1.000000 | `best_tuned_lstm_comparison-20260331T201252Z` | `tuning_winners.csv` |
| Baseline-LR | `{"flattened_sequence": true, "model": "LinearRegression", "seq_len": 20}` | 0.000013972209 | 0.000041327857 | 0.000017797154 | 0.002570154766 | 1.000000 | `best_tuned_lstm_comparison-20260331T201252Z-baseline-lr` | `tuning_winners.csv` |
| RNN | `{"batch_size": 64, "data_source": "spy", "hidden": 64, "horizon": 1, "layers": 3, "lr": 0.001, "seq_len": 30, "target_mode": "next_volatility", "target_smooth_window": 5, "task_id": "next_volatility"}` | 0.000015292936 | 0.000043284226 | 0.000021104879 | 0.002889096064 | 1.000000 | `best_tuned_rnn_comparison-20260331T201253Z` | `tuning_winners.csv` |
| Transformer | `{"batch_size": 32, "d_model": 64, "data_source": "spy", "horizon": 1, "lr": 0.001, "nhead": 2, "num_layers": 1, "seq_len": 20, "target_mode": "next_volatility", "target_smooth_window": 5, "task_id": "next_volatility"}` | 0.000043689018 | 0.000049758833 | 0.000021354819 | 0.002929901167 | 0.998744 | `best_tuned_transformer_comparison-20260331T201254Z` | `tuning_winners.csv` |
