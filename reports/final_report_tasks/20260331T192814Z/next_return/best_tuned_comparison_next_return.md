# Best Tuned Model Comparison

- Config source: `tuning_winners.csv`
- Source note: Uses the final frozen staged winners from sequential tuning for each model.
- Baseline: shared flattened-sequence linear regression on the same split

## Summary

- Best model by validation MSE: **LSTM** (0.000243675326)
- Best model by test MSE: **GRU** (0.000092716793)
- Ranking by validation MSE:
  1. LSTM — val MSE 0.000243675326, test MSE 0.000092900540
  2. GRU — val MSE 0.000248084090, test MSE 0.000092716793
  3. RNN — val MSE 0.000269017184, test MSE 0.000113939677
  4. Transformer — val MSE 0.000305234497, test MSE 0.000124861108
  5. Baseline-LR — val MSE 0.000372516956, test MSE 0.000122597410
- Note: This comparison uses the final frozen staged winners from sequential tuning for each model.

## Results

| Model | Hyperparameters | Train MSE | Validation MSE | Test MSE | MAE | Directional Accuracy | Run ID | Config source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| LSTM | `{"batch_size": 32, "data_source": "spy", "hidden": 128, "horizon": 1, "layers": 3, "lr": 0.0005, "seq_len": 30, "target_mode": "next_return", "target_smooth_window": 1, "task_id": "next_return"}` | 0.000143519424 | 0.000243675326 | 0.000092900540 | 0.006650535659 | 0.527044 | `best_tuned_lstm_comparison-20260331T195507Z` | `tuning_winners.csv` |
| GRU | `{"batch_size": 64, "data_source": "spy", "hidden": 64, "horizon": 1, "layers": 3, "lr": 0.0001, "seq_len": 20, "target_mode": "next_return", "target_smooth_window": 1, "task_id": "next_return"}` | 0.000136737509 | 0.000248084090 | 0.000092716793 | 0.006688856538 | 0.516939 | `best_tuned_gru_comparison-20260331T195508Z` | `tuning_winners.csv` |
| RNN | `{"batch_size": 128, "data_source": "spy", "hidden": 128, "horizon": 1, "layers": 1, "lr": 0.001, "seq_len": 30, "target_mode": "next_return", "target_smooth_window": 1, "task_id": "next_return"}` | 0.000126424465 | 0.000269017184 | 0.000113939677 | 0.007150323007 | 0.477987 | `best_tuned_rnn_comparison-20260331T195508Z` | `tuning_winners.csv` |
| Transformer | `{"batch_size": 32, "d_model": 64, "data_source": "spy", "horizon": 1, "lr": 0.001, "nhead": 8, "num_layers": 1, "seq_len": 20, "target_mode": "next_return", "target_smooth_window": 1, "task_id": "next_return"}` | 0.000107226642 | 0.000305234497 | 0.000124861108 | 0.007316818106 | 0.490590 | `best_tuned_transformer_comparison-20260331T195509Z` | `tuning_winners.csv` |
| Baseline-LR | `{"flattened_sequence": true, "model": "LinearRegression", "seq_len": 30}` | 0.000106213001 | 0.000372516956 | 0.000122597410 | 0.007687991232 | 0.489308 | `best_tuned_lstm_comparison-20260331T195507Z-baseline-lr` | `tuning_winners.csv` |
