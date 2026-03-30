# Hyper-Parameter Impact Report

This report summarises how the tuning workflow changed model performance and compares the best tuned runs across models.

## Best tuned configuration by model

| Model | Best validation MSE | Best testing MSE | Best training MSE | MAE | DA | Hyperparameters | Run ID |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| LSTM | 1.49605e-06 | 1.10954e-07 | 2.1573e-07 | 0.000261717 | 98.64% | `{"hidden": 128, "input_size": 8, "layers": 2, "random_seed": 42}` | `lstm_experiment-20260330T113310Z` |
| RNN | 3.34421e-06 | 3.47841e-06 | 3.51925e-06 | 0.00143773 | 92.95% | `{"hidden": 64, "input_size": 8, "layers": 2, "random_seed": 42}` | `rnn_experiment-20260330T113511Z` |
| GRU | 4.09286e-06 | 3.86169e-06 | 3.41991e-06 | 0.00150962 | 93.77% | `{"hidden": 64, "input_size": 8, "layers": 2, "random_seed": 42}` | `gru_experiment-20260330T113346Z` |
| Transformer | 1.19913e-05 | 1.05919e-05 | 0.000518755 | 0.00250103 | 88.89% | `{"d_model": 64, "dropout": 0.1, "input_size": 8, "nhead": 4, "num_layers": 2, "random_seed": 42}` | `transformer_experiment-20260330T113831Z` |

## Stage-by-stage hyper-parameter impact

The tuning workflow was sequential, so each stage winner was selected while earlier winners stayed frozen.

### GRU

- Stage 1 (`lr`): winner 0.0001 with validation MSE 4.09286e-06; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 64 with validation MSE 4.09286e-06; relative to the previous stage this matched by 0.
- Stage 3 (`layers`): winner 2 with validation MSE 4.09286e-06; relative to the previous stage this matched by 0.
- Stage 4 (`batch_size`): winner 64 with validation MSE 4.09286e-06; relative to the previous stage this matched by 0.

### LSTM

- Stage 1 (`lr`): winner 0.001 with validation MSE 2.05896e-06; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 1.49605e-06; relative to the previous stage this improved by 5.62913e-07.
- Stage 3 (`layers`): winner 2 with validation MSE 1.49605e-06; relative to the previous stage this matched by 0.
- Stage 4 (`batch_size`): winner 64 with validation MSE 1.49605e-06; relative to the previous stage this matched by 0.

### RNN

- Stage 1 (`lr`): winner 0.001 with validation MSE 3.5336e-06; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 64 with validation MSE 3.5336e-06; relative to the previous stage this matched by 0.
- Stage 3 (`layers`): winner 2 with validation MSE 3.5336e-06; relative to the previous stage this matched by 0.
- Stage 4 (`batch_size`): winner 128 with validation MSE 3.34421e-06; relative to the previous stage this improved by 1.89384e-07.

### Transformer

- Stage 1 (`lr`): winner 0.0001 with validation MSE 2.0745e-05; relative to the previous stage this n/a.
- Stage 2 (`d_model`): winner 64 with validation MSE 2.0745e-05; relative to the previous stage this matched by 0.
- Stage 3 (`num_layers`): winner 2 with validation MSE 2.0745e-05; relative to the previous stage this matched by 0.
- Stage 4 (`nhead`): winner 4 with validation MSE 2.0745e-05; relative to the previous stage this matched by 0.
- Stage 5 (`batch_size`): winner 32 with validation MSE 1.19913e-05; relative to the previous stage this improved by 8.75371e-06.

## Interpretation

- **Validation winner:** LSTM achieved the lowest validation MSE at 1.49605e-06.
- **Testing winner:** LSTM achieved the lowest testing MSE at 1.10954e-07.
- **Directional winner:** LSTM achieved the highest directional accuracy at 98.64%.
- Across the current tuning archive, recurrent models stayed tightly grouped, while the Transformer remained materially higher-loss than the recurrent models after tuning.

## Figure

![Best tuned model losses](figures\hyperparameter_model_loss_summary_sine_next_day.svg)

The figure uses one shared y-axis across three subplots so the training, testing, and validation losses remain directly comparable.
