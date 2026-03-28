# Hyper-Parameter Impact Report

This report summarises how the tuning workflow changed model performance and compares the best tuned runs across models.

## Best tuned configuration by model

| Model | Best validation MSE | Best testing MSE | Best training MSE | MAE | DA | Hyperparameters | Run ID |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| LSTM | 5.6441e-07 | 3.35821e-07 | 4.24404e-07 | 0.00045882 | 97.02% | `{"hidden": 128, "input_size": 8, "layers": 2}` | `lstm_experiment-20260327T210831Z` |
| GRU | 8.20602e-07 | 1.45372e-07 | 2.10281e-07 | 0.000302072 | 97.56% | `{"hidden": 128, "input_size": 8, "layers": 3}` | `gru_experiment-20260327T210912Z` |
| RNN | 1.0374e-06 | 9.80425e-07 | 8.78578e-07 | 0.000768548 | 97.29% | `{"hidden": 32, "input_size": 8, "layers": 3}` | `rnn_experiment-20260327T211003Z` |
| Transformer | 7.97378e-06 | 1.07923e-05 | 0.000259138 | 0.00216207 | 93.77% | `{"d_model": 64, "dropout": 0.1, "input_size": 8, "nhead": 4, "num_layers": 1}` | `transformer_experiment-20260327T211253Z` |

## Stage-by-stage hyper-parameter impact

The tuning workflow was sequential, so each stage winner was selected while earlier winners stayed frozen.

### GRU

- Stage 1 (`lr`): winner 0.0005 with validation MSE 1.15407e-06; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 2.85358e-06; relative to the previous stage this worsened by 1.69952e-06.
- Stage 3 (`layers`): winner 3 with validation MSE 8.20602e-07; relative to the previous stage this improved by 2.03298e-06.
- Stage 4 (`batch_size`): winner 32 with validation MSE 4.27365e-06; relative to the previous stage this worsened by 3.45305e-06.

### LSTM

- Stage 1 (`lr`): winner 0.0001 with validation MSE 9.27288e-07; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 3.59483e-06; relative to the previous stage this worsened by 2.66754e-06.
- Stage 3 (`layers`): winner 2 with validation MSE 7.20617e-07; relative to the previous stage this improved by 2.87421e-06.
- Stage 4 (`batch_size`): winner 64 with validation MSE 5.6441e-07; relative to the previous stage this improved by 1.56206e-07.

### RNN

- Stage 1 (`lr`): winner 0.001 with validation MSE 4.39366e-06; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 32 with validation MSE 8.44614e-06; relative to the previous stage this worsened by 4.05249e-06.
- Stage 3 (`layers`): winner 3 with validation MSE 1.0374e-06; relative to the previous stage this improved by 7.40874e-06.
- Stage 4 (`batch_size`): winner 64 with validation MSE 4.86838e-06; relative to the previous stage this worsened by 3.83098e-06.

### Transformer

- Stage 1 (`lr`): winner 0.0005 with validation MSE 2.23175e-05; relative to the previous stage this n/a.
- Stage 2 (`d_model`): winner 64 with validation MSE 2.12581e-05; relative to the previous stage this improved by 1.05941e-06.
- Stage 3 (`num_layers`): winner 1 with validation MSE 1.74031e-05; relative to the previous stage this improved by 3.85505e-06.
- Stage 4 (`nhead`): winner 4 with validation MSE 7.97378e-06; relative to the previous stage this improved by 9.42928e-06.
- Stage 5 (`batch_size`): winner 64 with validation MSE 1.45042e-05; relative to the previous stage this worsened by 6.53042e-06.

## Interpretation

- **Validation winner:** LSTM achieved the lowest validation MSE at 5.6441e-07.
- **Testing winner:** GRU achieved the lowest testing MSE at 1.45372e-07.
- **Directional winner:** GRU achieved the highest directional accuracy at 97.56%.
- Across the current tuning archive, recurrent models stayed tightly grouped, while the Transformer remained materially higher-loss than the recurrent models after tuning.

## Figure

![Best tuned model losses](figures\hyperparameter_model_loss_summary_sine_next_day.svg)

The figure uses one shared y-axis across three subplots so the training, testing, and validation losses remain directly comparable.
