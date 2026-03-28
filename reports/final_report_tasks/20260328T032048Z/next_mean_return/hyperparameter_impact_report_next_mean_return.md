# Hyper-Parameter Impact Report

This report summarises how the tuning workflow changed model performance and compares the best tuned runs across models.

## Best tuned configuration by model

| Model | Best validation MSE | Best testing MSE | Best training MSE | MAE | DA | Hyperparameters | Run ID |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| GRU | 3.95987e-05 | 1.50653e-05 | 2.08897e-05 | 0.00288549 | 60.71% | `{"hidden": 128, "input_size": 8, "layers": 3}` | `gru_experiment-20260328T035940Z` |
| LSTM | 4.11088e-05 | 1.6134e-05 | 0.000143336 | 0.00309041 | 41.18% | `{"hidden": 128, "input_size": 8, "layers": 3}` | `lstm_experiment-20260328T035724Z` |
| Transformer | 4.73474e-05 | 1.82144e-05 | 5.25141e-05 | 0.00300375 | 60.58% | `{"d_model": 32, "dropout": 0.1, "input_size": 8, "nhead": 8, "num_layers": 3}` | `transformer_experiment-20260328T040914Z` |
| RNN | 4.77045e-05 | 1.7107e-05 | 2.79367e-05 | 0.00312715 | 52.02% | `{"hidden": 128, "input_size": 8, "layers": 2}` | `rnn_experiment-20260328T040103Z` |

## Stage-by-stage hyper-parameter impact

The tuning workflow was sequential, so each stage winner was selected while earlier winners stayed frozen.

### GRU

- Stage 1 (`lr`): winner 0.001 with validation MSE 5.1665e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 5.9631e-05; relative to the previous stage this worsened by 7.96608e-06.
- Stage 3 (`layers`): winner 3 with validation MSE 4.23019e-05; relative to the previous stage this improved by 1.73291e-05.
- Stage 4 (`batch_size`): winner 128 with validation MSE 3.95987e-05; relative to the previous stage this improved by 2.70327e-06.

### LSTM

- Stage 1 (`lr`): winner 0.0005 with validation MSE 4.66165e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 4.17934e-05; relative to the previous stage this improved by 4.82311e-06.
- Stage 3 (`layers`): winner 3 with validation MSE 4.11088e-05; relative to the previous stage this improved by 6.84538e-07.
- Stage 4 (`batch_size`): winner 64 with validation MSE 4.12397e-05; relative to the previous stage this worsened by 1.30921e-07.

### RNN

- Stage 1 (`lr`): winner 0.001 with validation MSE 5.14397e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 4.95833e-05; relative to the previous stage this improved by 1.85641e-06.
- Stage 3 (`layers`): winner 2 with validation MSE 4.8879e-05; relative to the previous stage this improved by 7.04325e-07.
- Stage 4 (`batch_size`): winner 32 with validation MSE 4.77045e-05; relative to the previous stage this improved by 1.17444e-06.

### Transformer

- Stage 1 (`lr`): winner 0.0005 with validation MSE 0.000105389; relative to the previous stage this n/a.
- Stage 2 (`d_model`): winner 32 with validation MSE 7.35783e-05; relative to the previous stage this improved by 3.18105e-05.
- Stage 3 (`num_layers`): winner 3 with validation MSE 6.74436e-05; relative to the previous stage this improved by 6.13463e-06.
- Stage 4 (`nhead`): winner 8 with validation MSE 8.0987e-05; relative to the previous stage this worsened by 1.35433e-05.
- Stage 5 (`batch_size`): winner 32 with validation MSE 4.73474e-05; relative to the previous stage this improved by 3.36396e-05.

## Interpretation

- **Validation winner:** GRU achieved the lowest validation MSE at 3.95987e-05.
- **Testing winner:** GRU achieved the lowest testing MSE at 1.50653e-05.
- **Directional winner:** GRU achieved the highest directional accuracy at 60.71%.
- Across the current tuning archive, recurrent models stayed tightly grouped, while the Transformer remained materially higher-loss than the recurrent models after tuning.

## Figure

![Best tuned model losses](figures\hyperparameter_model_loss_summary_next_mean_return.svg)

The figure uses one shared y-axis across three subplots so the training, testing, and validation losses remain directly comparable.
