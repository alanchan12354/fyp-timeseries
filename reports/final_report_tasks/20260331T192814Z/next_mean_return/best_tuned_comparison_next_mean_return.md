# Best Tuned Model Comparison

- Config source: `tuning_winners.csv`
- Source note: Uses the final frozen staged winners from sequential tuning for each model.
- Baseline: shared flattened-sequence linear regression on the same split

## Summary

- Best model by validation MSE: **LSTM** (0.000039658262)
- Best model by test MSE: **GRU** (0.000015496840)
- Ranking by validation MSE:
  1. LSTM — val MSE 0.000039658262, test MSE 0.000016106107
  2. GRU — val MSE 0.000044156570, test MSE 0.000015496840
  3. RNN — val MSE 0.000050107314, test MSE 0.000016532178
  4. Baseline-LR — val MSE 0.000061308254, test MSE 0.000021942094
  5. Transformer — val MSE 0.000064074053, test MSE 0.000030383495
- Note: This comparison uses the final frozen staged winners from sequential tuning for each model.

## Results

| Model | Hyperparameters | Train MSE | Validation MSE | Test MSE | MAE | Directional Accuracy | Run ID | Config source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| LSTM | `{"batch_size": 128, "data_source": "spy", "hidden": 64, "horizon": 1, "layers": 3, "lr": 0.001, "seq_len": 60, "target_mode": "next_mean_return", "target_smooth_window": 5, "task_id": "next_mean_return"}` | 0.000021139145 | 0.000039658262 | 0.000016106107 | 0.002936071290 | 0.611392 | `best_tuned_lstm_comparison-20260331T203017Z` | `tuning_winners.csv` |
| GRU | `{"batch_size": 128, "data_source": "spy", "hidden": 128, "horizon": 1, "layers": 3, "lr": 0.001, "seq_len": 30, "target_mode": "next_mean_return", "target_smooth_window": 5, "task_id": "next_mean_return"}` | 0.000022153405 | 0.000044156570 | 0.000015496840 | 0.002932350612 | 0.569811 | `best_tuned_gru_comparison-20260331T203017Z` | `tuning_winners.csv` |
| RNN | `{"batch_size": 32, "data_source": "spy", "hidden": 128, "horizon": 1, "layers": 3, "lr": 0.0005, "seq_len": 60, "target_mode": "next_mean_return", "target_smooth_window": 5, "task_id": "next_mean_return"}` | 0.000023702505 | 0.000050107314 | 0.000016532178 | 0.003000947617 | 0.620253 | `best_tuned_rnn_comparison-20260331T203018Z` | `tuning_winners.csv` |
| Baseline-LR | `{"flattened_sequence": true, "model": "LinearRegression", "seq_len": 60}` | 0.000015308177 | 0.000061308254 | 0.000021942094 | 0.003477769234 | 0.525316 | `best_tuned_lstm_comparison-20260331T203017Z-baseline-lr` | `tuning_winners.csv` |
| Transformer | `{"batch_size": 32, "d_model": 64, "data_source": "spy", "horizon": 1, "lr": 0.001, "nhead": 8, "num_layers": 1, "seq_len": 30, "target_mode": "next_mean_return", "target_smooth_window": 5, "task_id": "next_mean_return"}` | 0.000044238672 | 0.000064074053 | 0.000030383495 | 0.003808957281 | 0.510692 | `best_tuned_transformer_comparison-20260331T203018Z` | `tuning_winners.csv` |
