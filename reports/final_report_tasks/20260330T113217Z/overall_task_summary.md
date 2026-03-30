# Overall task performance summary

This table compares the best-tuned winner in each of the four core tasks.

| task_id | target_mode | horizon | target_smooth_window | best_model_by_test_mse | best_test_mse | best_val_mse | best_train_mse |
|---|---|---:|---:|---|---:|---:|---:|
| sine_next_day | sine_next_day | 1 | 1 | Baseline-LR | 4.807035774110167e-08 | 4.592560596418131e-08 | 3.5837217038958094e-08 |
| next_return | next_return | 1 | 1 | LSTM | 9.014787792693824e-05 | 0.00024216187244967038 | 0.00020063419092258492 |
| next_volatility | next_volatility | 1 | 5 | Baseline-LR | 1.8526507889821364e-05 | 4.331356270684067e-05 | 1.3149623132286978e-05 |
| next_mean_return | next_mean_return | 1 | 5 | RNN | 1.5500108929700218e-05 | 4.505710601411983e-05 | 1.9904771728573513e-05 |
