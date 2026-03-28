# Best Tuned Model Comparison

- Config source: `tuning_winners.csv`
- Source note: Uses the final frozen staged winners from sequential tuning for each model.
- Baseline: shared flattened-sequence linear regression on the same split

## Summary

- Best model by validation MSE: **Baseline-LR** (0.000000045926)
- Best model by test MSE: **Baseline-LR** (0.000000048070)
- Ranking by validation MSE:
  1. Baseline-LR — val MSE 0.000000045926, test MSE 0.000000048070
  2. LSTM — val MSE 0.000000826048, test MSE 0.000000408471
  3. GRU — val MSE 0.000005221039, test MSE 0.000005026586
  4. RNN — val MSE 0.000008548595, test MSE 0.000005361074
  5. Transformer — val MSE 0.000010260271, test MSE 0.000010966342
- Note: This comparison uses the final frozen staged winners from sequential tuning for each model.

## Results

| Model | Hyperparameters | Train MSE | Validation MSE | Test MSE | MAE | Directional Accuracy | Run ID | Config source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Baseline-LR | `{"flattened_sequence": true, "model": "LinearRegression"}` | 0.000000035837 | 0.000000045926 | 0.000000048070 | 0.000174059896 | 0.970190 | `best_tuned_lstm_comparison-20260327T211351Z-baseline-lr` | `tuning_winners.csv` |
| LSTM | `{"batch_size": 64, "data_source": "sine", "hidden": 128, "horizon": 1, "layers": 2, "lr": 0.0001, "target_mode": "sine_next_day", "target_smooth_window": 1, "task_id": "sine_next_day"}` | 0.000000419421 | 0.000000826048 | 0.000000408471 | 0.000508791535 | 0.978320 | `best_tuned_lstm_comparison-20260327T211351Z` | `tuning_winners.csv` |
| GRU | `{"batch_size": 32, "data_source": "sine", "hidden": 128, "horizon": 1, "layers": 3, "lr": 0.0005, "target_mode": "sine_next_day", "target_smooth_window": 1, "task_id": "sine_next_day"}` | 0.000181508925 | 0.000005221039 | 0.000005026586 | 0.001914761782 | 0.929539 | `best_tuned_gru_comparison-20260327T211351Z` | `tuning_winners.csv` |
| RNN | `{"batch_size": 64, "data_source": "sine", "hidden": 32, "horizon": 1, "layers": 3, "lr": 0.001, "target_mode": "sine_next_day", "target_smooth_window": 1, "task_id": "sine_next_day"}` | 0.000007289247 | 0.000008548595 | 0.000005361074 | 0.001814029638 | 0.929539 | `best_tuned_rnn_comparison-20260327T211351Z` | `tuning_winners.csv` |
| Transformer | `{"batch_size": 64, "d_model": 64, "data_source": "sine", "horizon": 1, "lr": 0.0005, "nhead": 4, "num_layers": 1, "target_mode": "sine_next_day", "target_smooth_window": 1, "task_id": "sine_next_day"}` | 0.000228354446 | 0.000010260271 | 0.000010966342 | 0.002655889390 | 0.864499 | `best_tuned_transformer_comparison-20260327T211351Z` | `tuning_winners.csv` |
