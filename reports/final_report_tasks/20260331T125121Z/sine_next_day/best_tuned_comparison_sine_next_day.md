# Best Tuned Model Comparison

- Config source: `tuning_winners.csv`
- Source note: Uses the final frozen staged winners from sequential tuning for each model.
- Baseline: shared flattened-sequence linear regression on the same split

## Summary

- Best model by validation MSE: **Baseline-LR** (0.000000045926)
- Best model by test MSE: **Baseline-LR** (0.000000048070)
- Ranking by validation MSE:
  1. Baseline-LR — val MSE 0.000000045926, test MSE 0.000000048070
  2. LSTM — val MSE 0.000001496048, test MSE 0.000000110954
  3. RNN — val MSE 0.000001723856, test MSE 0.000001601384
  4. GRU — val MSE 0.000003147241, test MSE 0.000002293541
  5. Transformer — val MSE 0.000011991270, test MSE 0.000010591856
- Note: This comparison uses the final frozen staged winners from sequential tuning for each model.

## Results

| Model | Hyperparameters | Train MSE | Validation MSE | Test MSE | MAE | Directional Accuracy | Run ID | Config source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Baseline-LR | `{"flattened_sequence": true, "model": "LinearRegression", "seq_len": 30}` | 0.000000035837 | 0.000000045926 | 0.000000048070 | 0.000174059896 | 0.970190 | `best_tuned_lstm_comparison-20260331T130042Z-baseline-lr` | `tuning_winners.csv` |
| LSTM | `{"batch_size": 64, "data_source": "sine", "hidden": 128, "horizon": 1, "layers": 2, "lr": 0.001, "seq_len": 30, "target_mode": "sine_next_day", "target_smooth_window": 1, "task_id": "sine_next_day"}` | 0.000000215730 | 0.000001496048 | 0.000000110954 | 0.000261716555 | 0.986450 | `best_tuned_lstm_comparison-20260331T130042Z` | `tuning_winners.csv` |
| RNN | `{"batch_size": 128, "data_source": "sine", "hidden": 64, "horizon": 1, "layers": 2, "lr": 0.001, "seq_len": 20, "target_mode": "sine_next_day", "target_smooth_window": 1, "task_id": "sine_next_day"}` | 0.000001675418 | 0.000001723856 | 0.000001601384 | 0.000966363485 | 0.948649 | `best_tuned_rnn_comparison-20260331T130042Z` | `tuning_winners.csv` |
| GRU | `{"batch_size": 64, "data_source": "sine", "hidden": 64, "horizon": 1, "layers": 2, "lr": 0.0001, "seq_len": 60, "target_mode": "sine_next_day", "target_smooth_window": 1, "task_id": "sine_next_day"}` | 0.000002362519 | 0.000003147241 | 0.000002293541 | 0.001189883193 | 0.956044 | `best_tuned_gru_comparison-20260331T130042Z` | `tuning_winners.csv` |
| Transformer | `{"batch_size": 32, "d_model": 64, "data_source": "sine", "horizon": 1, "lr": 0.0001, "nhead": 4, "num_layers": 2, "seq_len": 30, "target_mode": "sine_next_day", "target_smooth_window": 1, "task_id": "sine_next_day"}` | 0.000518754816 | 0.000011991270 | 0.000010591856 | 0.002501032546 | 0.888889 | `best_tuned_transformer_comparison-20260331T130042Z` | `tuning_winners.csv` |
