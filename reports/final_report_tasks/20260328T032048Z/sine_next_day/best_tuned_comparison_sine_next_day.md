# Best Tuned Model Comparison

- Config source: `tuning_winners.csv`
- Source note: Uses the final frozen staged winners from sequential tuning for each model.
- Baseline: shared flattened-sequence linear regression on the same split

## Summary

- Best model by validation MSE: **Baseline-LR** (0.000000045926)
- Best model by test MSE: **Baseline-LR** (0.000000048070)
- Ranking by validation MSE:
  1. Baseline-LR — val MSE 0.000000045926, test MSE 0.000000048070
  2. RNN — val MSE 0.000003446677, test MSE 0.000003220928
  3. GRU — val MSE 0.000003909916, test MSE 0.000001410753
  4. LSTM — val MSE 0.000008232446, test MSE 0.000003530492
  5. Transformer — val MSE 0.000010844949, test MSE 0.000010177931
- Note: This comparison uses the final frozen staged winners from sequential tuning for each model.

## Results

| Model | Hyperparameters | Train MSE | Validation MSE | Test MSE | MAE | Directional Accuracy | Run ID | Config source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Baseline-LR | `{"flattened_sequence": true, "model": "LinearRegression"}` | 0.000000035837 | 0.000000045926 | 0.000000048070 | 0.000174059896 | 0.970190 | `best_tuned_lstm_comparison-20260328T032711Z-baseline-lr` | `tuning_winners.csv` |
| RNN | `{"batch_size": 32, "data_source": "sine", "hidden": 64, "horizon": 1, "layers": 1, "lr": 0.001, "target_mode": "sine_next_day", "target_smooth_window": 1, "task_id": "sine_next_day"}` | 0.000003467867 | 0.000003446677 | 0.000003220928 | 0.001421423454 | 0.956640 | `best_tuned_rnn_comparison-20260328T032711Z` | `tuning_winners.csv` |
| GRU | `{"batch_size": 128, "data_source": "sine", "hidden": 128, "horizon": 1, "layers": 2, "lr": 0.0005, "target_mode": "sine_next_day", "target_smooth_window": 1, "task_id": "sine_next_day"}` | 0.000001827494 | 0.000003909916 | 0.000001410753 | 0.000976799950 | 0.972900 | `best_tuned_gru_comparison-20260328T032711Z` | `tuning_winners.csv` |
| LSTM | `{"batch_size": 64, "data_source": "sine", "hidden": 64, "horizon": 1, "layers": 1, "lr": 0.0005, "target_mode": "sine_next_day", "target_smooth_window": 1, "task_id": "sine_next_day"}` | 0.000004276092 | 0.000008232446 | 0.000003530492 | 0.001451728142 | 0.945799 | `best_tuned_lstm_comparison-20260328T032711Z` | `tuning_winners.csv` |
| Transformer | `{"batch_size": 32, "d_model": 128, "data_source": "sine", "horizon": 1, "lr": 0.0005, "nhead": 2, "num_layers": 1, "target_mode": "sine_next_day", "target_smooth_window": 1, "task_id": "sine_next_day"}` | 0.000125987820 | 0.000010844949 | 0.000010177931 | 0.002472155373 | 0.932249 | `best_tuned_transformer_comparison-20260328T032711Z` | `tuning_winners.csv` |
