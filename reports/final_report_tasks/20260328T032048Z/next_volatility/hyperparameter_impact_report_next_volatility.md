# Hyper-Parameter Impact Report

This report summarises how the tuning workflow changed model performance and compares the best tuned runs across models.

## Best tuned configuration by model

| Model | Best validation MSE | Best testing MSE | Best training MSE | MAE | DA | Hyperparameters | Run ID |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| GRU | 4.23955e-05 | 2.86114e-05 | 1.92773e-05 | 0.00299733 | 99.75% | `{"hidden": 32, "input_size": 8, "layers": 2}` | `gru_experiment-20260328T034523Z` |
| LSTM | 4.30292e-05 | 2.57028e-05 | 0.00112384 | 0.00307122 | 100.00% | `{"hidden": 64, "input_size": 8, "layers": 2}` | `lstm_experiment-20260328T034212Z` |
| RNN | 4.44331e-05 | 2.10213e-05 | 1.78575e-05 | 0.00304841 | 100.00% | `{"hidden": 128, "input_size": 8, "layers": 3}` | `rnn_experiment-20260328T034651Z` |
| Transformer | 5.37459e-05 | 1.92828e-05 | 3.72795e-05 | 0.00263675 | 100.00% | `{"d_model": 32, "dropout": 0.1, "input_size": 8, "nhead": 4, "num_layers": 1}` | `transformer_experiment-20260328T035107Z` |

## Stage-by-stage hyper-parameter impact

The tuning workflow was sequential, so each stage winner was selected while earlier winners stayed frozen.

### GRU

- Stage 1 (`lr`): winner 0.001 with validation MSE 5.02432e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 32 with validation MSE 4.61501e-05; relative to the previous stage this improved by 4.0931e-06.
- Stage 3 (`layers`): winner 2 with validation MSE 4.87494e-05; relative to the previous stage this worsened by 2.5993e-06.
- Stage 4 (`batch_size`): winner 64 with validation MSE 4.23955e-05; relative to the previous stage this improved by 6.35393e-06.

### LSTM

- Stage 1 (`lr`): winner 0.0005 with validation MSE 4.30292e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 32 with validation MSE 4.34112e-05; relative to the previous stage this worsened by 3.82055e-07.
- Stage 3 (`layers`): winner 3 with validation MSE 4.58046e-05; relative to the previous stage this worsened by 2.39336e-06.
- Stage 4 (`batch_size`): winner 32 with validation MSE 4.4334e-05; relative to the previous stage this improved by 1.4706e-06.

### RNN

- Stage 1 (`lr`): winner 0.001 with validation MSE 5.08758e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 4.58825e-05; relative to the previous stage this improved by 4.99334e-06.
- Stage 3 (`layers`): winner 3 with validation MSE 4.44331e-05; relative to the previous stage this improved by 1.4494e-06.
- Stage 4 (`batch_size`): winner 128 with validation MSE 4.59749e-05; relative to the previous stage this worsened by 1.54186e-06.

### Transformer

- Stage 1 (`lr`): winner 0.001 with validation MSE 9.33932e-05; relative to the previous stage this n/a.
- Stage 2 (`d_model`): winner 32 with validation MSE 8.25363e-05; relative to the previous stage this improved by 1.08569e-05.
- Stage 3 (`num_layers`): winner 1 with validation MSE 5.37459e-05; relative to the previous stage this improved by 2.87904e-05.
- Stage 4 (`nhead`): winner 8 with validation MSE 6.09028e-05; relative to the previous stage this worsened by 7.1569e-06.
- Stage 5 (`batch_size`): winner 32 with validation MSE 6.42651e-05; relative to the previous stage this worsened by 3.3623e-06.

## Interpretation

- **Validation winner:** GRU achieved the lowest validation MSE at 4.23955e-05.
- **Testing winner:** Transformer achieved the lowest testing MSE at 1.92828e-05.
- **Directional winner:** LSTM achieved the highest directional accuracy at 100.00%.
- Across the current tuning archive, recurrent models stayed tightly grouped, while the Transformer remained materially higher-loss than the recurrent models after tuning.

## Figure

![Best tuned model losses](figures\hyperparameter_model_loss_summary_next_volatility.svg)

The figure uses one shared y-axis across three subplots so the training, testing, and validation losses remain directly comparable.
