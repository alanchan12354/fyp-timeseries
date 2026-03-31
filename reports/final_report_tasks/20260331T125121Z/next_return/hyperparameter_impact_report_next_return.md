# Hyper-Parameter Impact Report

This report summarises how the tuning workflow changed model performance and compares the best tuned runs across models.

## Best tuned configuration by model

| Model | Best validation MSE | Best testing MSE | Best training MSE | MAE | DA | Hyperparameters | Run ID |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| LSTM | 0.000243977 | 9.19806e-05 | 0.00014352 | 0.00661839 | 52.70% | `{"hidden": 128, "input_size": 8, "layers": 3, "random_seed": 42}` | `lstm_experiment-20260331T130337Z` |
| GRU | 0.000248079 | 9.21956e-05 | 0.000136736 | 0.00666861 | 51.51% | `{"hidden": 64, "input_size": 8, "layers": 3, "random_seed": 42}` | `gru_experiment-20260331T130619Z` |
| RNN | 0.000269338 | 0.000112669 | 0.000126425 | 0.00711664 | 47.80% | `{"hidden": 128, "input_size": 8, "layers": 1, "random_seed": 42}` | `rnn_experiment-20260331T130833Z` |
| Transformer | 0.000280221 | 0.00013229 | 0.000172529 | 0.00821334 | 44.15% | `{"d_model": 64, "dropout": 0.1, "input_size": 8, "nhead": 4, "num_layers": 2, "random_seed": 42}` | `transformer_experiment-20260331T131540Z` |
| Baseline-LR | 0.000372807 | 0.000121759 | 0.000106213 | 0.00766729 | 48.93% | `{"details": "{\"flattened_sequence\": true, \"model\": \"LinearRegression\", \"seq_len\": 30}", "source": "best_tuned_comparison_csv"}` | `best_tuned_lstm_comparison-20260331T132130Z-baseline-lr` |

## Stage-by-stage hyper-parameter impact

The tuning workflow was sequential, so each stage winner was selected while earlier winners stayed frozen.

### Baseline-LR

- Stage 1 (`seq_len`): winner 20 with validation MSE 0.000347349; relative to the previous stage this n/a.

### GRU

- Stage 1 (`lr`): winner 0.0001 with validation MSE 0.000259715; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 64 with validation MSE 0.000259719; relative to the previous stage this worsened by 4.4676e-09.
- Stage 3 (`layers`): winner 3 with validation MSE 0.000248854; relative to the previous stage this improved by 1.08651e-05.
- Stage 4 (`batch_size`): winner 64 with validation MSE 0.000248857; relative to the previous stage this worsened by 2.7174e-09.
- Stage 5 (`seq_len`): winner 20 with validation MSE 0.000248079; relative to the previous stage this improved by 7.77816e-07.

### LSTM

- Stage 1 (`lr`): winner 0.0005 with validation MSE 0.000248197; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 0.000246867; relative to the previous stage this improved by 1.33035e-06.
- Stage 3 (`layers`): winner 3 with validation MSE 0.000246403; relative to the previous stage this improved by 4.64083e-07.
- Stage 4 (`batch_size`): winner 32 with validation MSE 0.000243978; relative to the previous stage this improved by 2.42531e-06.
- Stage 5 (`seq_len`): winner 30 with validation MSE 0.000243977; relative to the previous stage this improved by 3.53393e-10.

### RNN

- Stage 1 (`lr`): winner 0.001 with validation MSE 0.00028255; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 0.000275834; relative to the previous stage this improved by 6.71542e-06.
- Stage 3 (`layers`): winner 1 with validation MSE 0.000272939; relative to the previous stage this improved by 2.89515e-06.
- Stage 4 (`batch_size`): winner 128 with validation MSE 0.000269338; relative to the previous stage this improved by 3.60126e-06.
- Stage 5 (`seq_len`): winner 30 with validation MSE 0.000269338; relative to the previous stage this improved by 1.89236e-10.

### Transformer

- Stage 1 (`lr`): winner 0.001 with validation MSE 0.000295524; relative to the previous stage this n/a.
- Stage 2 (`d_model`): winner 64 with validation MSE 0.000310084; relative to the previous stage this worsened by 1.456e-05.
- Stage 3 (`num_layers`): winner 2 with validation MSE 0.000291201; relative to the previous stage this improved by 1.88831e-05.
- Stage 4 (`nhead`): winner 4 with validation MSE 0.000281001; relative to the previous stage this improved by 1.02009e-05.
- Stage 5 (`batch_size`): winner 32 with validation MSE 0.000280221; relative to the previous stage this improved by 7.79677e-07.
- Stage 6 (`seq_len`): winner 30 with validation MSE 0.000283101; relative to the previous stage this worsened by 2.8805e-06.

## Interpretation

- **Validation winner:** LSTM achieved the lowest validation MSE at 0.000243977.
- **Testing winner:** LSTM achieved the lowest testing MSE at 9.19806e-05.
- **Directional winner:** LSTM achieved the highest directional accuracy at 52.70%.
- Across the current tuning archive, recurrent models stayed tightly grouped, while the Transformer remained materially higher-loss than the recurrent models after tuning.

## Figure

![Best tuned model losses](figures/hyperparameter_model_loss_summary_next_return.svg)

The figure uses one shared y-axis across three subplots so the training, testing, and validation losses remain directly comparable.
