# Hyper-Parameter Impact Report

This report summarises how the tuning workflow changed model performance and compares the best tuned runs across models.

## Best tuned configuration by model

| Model | Best validation MSE | Best testing MSE | Best training MSE | MAE | DA | Hyperparameters | Run ID |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| LSTM | 5.98754e-07 | 3.0106e-07 | 2.70514e-07 | 0.000431412 | 97.83% | `{"hidden": 64, "input_size": 8, "layers": 2}` | `lstm_experiment-20260328T032120Z` |
| RNN | 8.35394e-07 | 5.85591e-07 | 6.23568e-07 | 0.000599302 | 97.29% | `{"hidden": 64, "input_size": 8, "layers": 2}` | `rnn_experiment-20260328T032308Z` |
| GRU | 8.68073e-07 | 5.57398e-07 | 5.59281e-07 | 0.000586072 | 97.29% | `{"hidden": 128, "input_size": 8, "layers": 2}` | `gru_experiment-20260328T032223Z` |
| Transformer | 7.29116e-06 | 4.94446e-06 | 0.000121149 | 0.00180194 | 92.68% | `{"d_model": 128, "dropout": 0.1, "input_size": 8, "nhead": 2, "num_layers": 1}` | `transformer_experiment-20260328T032626Z` |

## Stage-by-stage hyper-parameter impact

The tuning workflow was sequential, so each stage winner was selected while earlier winners stayed frozen.

### GRU

- Stage 1 (`lr`): winner 0.0005 with validation MSE 2.00416e-06; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 8.68073e-07; relative to the previous stage this improved by 1.13609e-06.
- Stage 3 (`layers`): winner 2 with validation MSE 1.83761e-06; relative to the previous stage this worsened by 9.69541e-07.
- Stage 4 (`batch_size`): winner 128 with validation MSE 5.77958e-06; relative to the previous stage this worsened by 3.94196e-06.

### LSTM

- Stage 1 (`lr`): winner 0.0005 with validation MSE 5.98754e-07; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 64 with validation MSE 1.48791e-06; relative to the previous stage this worsened by 8.89154e-07.
- Stage 3 (`layers`): winner 1 with validation MSE 5.05776e-06; relative to the previous stage this worsened by 3.56985e-06.
- Stage 4 (`batch_size`): winner 64 with validation MSE 9.37328e-07; relative to the previous stage this improved by 4.12043e-06.

### RNN

- Stage 1 (`lr`): winner 0.001 with validation MSE 3.70435e-06; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 64 with validation MSE 8.35394e-07; relative to the previous stage this improved by 2.86896e-06.
- Stage 3 (`layers`): winner 1 with validation MSE 3.04794e-06; relative to the previous stage this worsened by 2.21255e-06.
- Stage 4 (`batch_size`): winner 32 with validation MSE 2.67128e-06; relative to the previous stage this improved by 3.76666e-07.

### Transformer

- Stage 1 (`lr`): winner 0.0005 with validation MSE 2.60953e-05; relative to the previous stage this n/a.
- Stage 2 (`d_model`): winner 128 with validation MSE 2.40177e-05; relative to the previous stage this improved by 2.07751e-06.
- Stage 3 (`num_layers`): winner 1 with validation MSE 1.07657e-05; relative to the previous stage this improved by 1.32521e-05.
- Stage 4 (`nhead`): winner 2 with validation MSE 1.21654e-05; relative to the previous stage this worsened by 1.39976e-06.
- Stage 5 (`batch_size`): winner 32 with validation MSE 7.29116e-06; relative to the previous stage this improved by 4.87429e-06.

## Interpretation

- **Validation winner:** LSTM achieved the lowest validation MSE at 5.98754e-07.
- **Testing winner:** LSTM achieved the lowest testing MSE at 3.0106e-07.
- **Directional winner:** LSTM achieved the highest directional accuracy at 97.83%.
- Across the current tuning archive, recurrent models stayed tightly grouped, while the Transformer remained materially higher-loss than the recurrent models after tuning.

## Figure

![Best tuned model losses](figures\hyperparameter_model_loss_summary_sine_next_day.svg)

The figure uses one shared y-axis across three subplots so the training, testing, and validation losses remain directly comparable.
