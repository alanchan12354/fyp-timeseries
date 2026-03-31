# Hyper-Parameter Impact Report

This report summarises how the tuning workflow changed model performance and compares the best tuned runs across models.

## Best tuned configuration by model

| Model | Best validation MSE | Best testing MSE | Best training MSE | MAE | DA | Hyperparameters | Run ID |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| Baseline-LR | 4.59256e-08 | 4.80704e-08 | 3.58372e-08 | 0.00017406 | 97.02% | `{"details": "{\"flattened_sequence\": true, \"model\": \"LinearRegression\", \"seq_len\": 30}", "source": "best_tuned_comparison_csv"}` | `best_tuned_lstm_comparison-20260331T130042Z-baseline-lr` |
| LSTM | 1.49605e-06 | 1.10954e-07 | 2.1573e-07 | 0.000261717 | 98.64% | `{"hidden": 128, "input_size": 8, "layers": 2, "random_seed": 42}` | `lstm_experiment-20260331T125144Z` |
| RNN | 1.72386e-06 | 1.60138e-06 | 1.67542e-06 | 0.000966363 | 94.86% | `{"hidden": 64, "input_size": 8, "layers": 2, "random_seed": 42}` | `rnn_experiment-20260331T125408Z` |
| GRU | 3.14724e-06 | 2.29354e-06 | 2.36252e-06 | 0.00118988 | 95.60% | `{"hidden": 64, "input_size": 8, "layers": 2, "random_seed": 42}` | `gru_experiment-20260331T125317Z` |
| Transformer | 1.19913e-05 | 1.05919e-05 | 0.000518755 | 0.00250103 | 88.89% | `{"d_model": 64, "dropout": 0.1, "input_size": 8, "nhead": 4, "num_layers": 2, "random_seed": 42}` | `transformer_experiment-20260331T125726Z` |

## Stage-by-stage hyper-parameter impact

The tuning workflow was sequential, so each stage winner was selected while earlier winners stayed frozen.

### Baseline-LR

- Stage 1 (`seq_len`): winner 20 with validation MSE 4.26212e-08; relative to the previous stage this n/a.

### GRU

- Stage 1 (`lr`): winner 0.0001 with validation MSE 4.09286e-06; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 64 with validation MSE 4.09286e-06; relative to the previous stage this matched by 0.
- Stage 3 (`layers`): winner 2 with validation MSE 4.09286e-06; relative to the previous stage this matched by 0.
- Stage 4 (`batch_size`): winner 64 with validation MSE 4.09286e-06; relative to the previous stage this matched by 0.
- Stage 5 (`seq_len`): winner 60 with validation MSE 3.14724e-06; relative to the previous stage this improved by 9.45615e-07.

### LSTM

- Stage 1 (`lr`): winner 0.001 with validation MSE 2.05896e-06; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 1.49605e-06; relative to the previous stage this improved by 5.62913e-07.
- Stage 3 (`layers`): winner 2 with validation MSE 1.49605e-06; relative to the previous stage this matched by 0.
- Stage 4 (`batch_size`): winner 64 with validation MSE 1.49605e-06; relative to the previous stage this matched by 0.
- Stage 5 (`seq_len`): winner 30 with validation MSE 1.49605e-06; relative to the previous stage this matched by 0.

### RNN

- Stage 1 (`lr`): winner 0.001 with validation MSE 3.5336e-06; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 64 with validation MSE 3.5336e-06; relative to the previous stage this matched by 0.
- Stage 3 (`layers`): winner 2 with validation MSE 3.5336e-06; relative to the previous stage this matched by 0.
- Stage 4 (`batch_size`): winner 128 with validation MSE 3.34421e-06; relative to the previous stage this improved by 1.89384e-07.
- Stage 5 (`seq_len`): winner 20 with validation MSE 1.72386e-06; relative to the previous stage this improved by 1.62036e-06.

### Transformer

- Stage 1 (`lr`): winner 0.0001 with validation MSE 2.0745e-05; relative to the previous stage this n/a.
- Stage 2 (`d_model`): winner 64 with validation MSE 2.0745e-05; relative to the previous stage this matched by 0.
- Stage 3 (`num_layers`): winner 2 with validation MSE 2.0745e-05; relative to the previous stage this matched by 0.
- Stage 4 (`nhead`): winner 4 with validation MSE 2.0745e-05; relative to the previous stage this matched by 0.
- Stage 5 (`batch_size`): winner 32 with validation MSE 1.19913e-05; relative to the previous stage this improved by 8.75371e-06.
- Stage 6 (`seq_len`): winner 30 with validation MSE 1.19913e-05; relative to the previous stage this matched by 0.

## Interpretation

- **Validation winner:** Baseline-LR achieved the lowest validation MSE at 4.59256e-08.
- **Testing winner:** Baseline-LR achieved the lowest testing MSE at 4.80704e-08.
- **Directional winner:** LSTM achieved the highest directional accuracy at 98.64%.
- Across the current tuning archive, recurrent models stayed tightly grouped, while the Transformer remained materially higher-loss than the recurrent models after tuning.

## Figure

![Best tuned model losses](figures/hyperparameter_model_loss_summary_sine_next_day.svg)

The figure uses one shared y-axis across three subplots so the training, testing, and validation losses remain directly comparable.
