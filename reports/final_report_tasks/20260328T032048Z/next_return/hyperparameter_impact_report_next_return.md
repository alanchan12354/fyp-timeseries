# Hyper-Parameter Impact Report

This report summarises how the tuning workflow changed model performance and compares the best tuned runs across models.

## Best tuned configuration by model

| Model | Best validation MSE | Best testing MSE | Best training MSE | MAE | DA | Hyperparameters | Run ID |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| LSTM | 0.000240184 | 9.09098e-05 | 0.000134443 | 0.00665214 | 47.67% | `{"hidden": 32, "input_size": 8, "layers": 2}` | `lstm_experiment-20260328T032827Z` |
| GRU | 0.000246558 | 9.08801e-05 | 0.000131933 | 0.00659724 | 52.08% | `{"hidden": 128, "input_size": 8, "layers": 3}` | `gru_experiment-20260328T033126Z` |
| RNN | 0.000271846 | 0.000113632 | 0.000124255 | 0.00751132 | 55.35% | `{"hidden": 64, "input_size": 8, "layers": 2}` | `rnn_experiment-20260328T033148Z` |
| Transformer | 0.000279971 | 0.000117152 | 0.000196001 | 0.00749169 | 52.08% | `{"d_model": 32, "dropout": 0.1, "input_size": 8, "nhead": 4, "num_layers": 1}` | `transformer_experiment-20260328T033856Z` |

## Stage-by-stage hyper-parameter impact

The tuning workflow was sequential, so each stage winner was selected while earlier winners stayed frozen.

### GRU

- Stage 1 (`lr`): winner 0.001 with validation MSE 0.000267156; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 0.000260312; relative to the previous stage this improved by 6.8447e-06.
- Stage 3 (`layers`): winner 3 with validation MSE 0.000250018; relative to the previous stage this improved by 1.02934e-05.
- Stage 4 (`batch_size`): winner 64 with validation MSE 0.000246558; relative to the previous stage this improved by 3.46029e-06.

### LSTM

- Stage 1 (`lr`): winner 0.001 with validation MSE 0.000256037; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 32 with validation MSE 0.000240184; relative to the previous stage this improved by 1.58534e-05.
- Stage 3 (`layers`): winner 3 with validation MSE 0.000247934; relative to the previous stage this worsened by 7.74996e-06.
- Stage 4 (`batch_size`): winner 32 with validation MSE 0.00024096; relative to the previous stage this improved by 6.97432e-06.

### RNN

- Stage 1 (`lr`): winner 0.0005 with validation MSE 0.000271846; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 0.000293803; relative to the previous stage this worsened by 2.19571e-05.
- Stage 3 (`layers`): winner 3 with validation MSE 0.000281246; relative to the previous stage this improved by 1.25572e-05.
- Stage 4 (`batch_size`): winner 64 with validation MSE 0.000272309; relative to the previous stage this improved by 8.93661e-06.

### Transformer

- Stage 1 (`lr`): winner 0.0005 with validation MSE 0.000332648; relative to the previous stage this n/a.
- Stage 2 (`d_model`): winner 32 with validation MSE 0.000280709; relative to the previous stage this improved by 5.19388e-05.
- Stage 3 (`num_layers`): winner 1 with validation MSE 0.000341301; relative to the previous stage this worsened by 6.05921e-05.
- Stage 4 (`nhead`): winner 4 with validation MSE 0.000279971; relative to the previous stage this improved by 6.13307e-05.
- Stage 5 (`batch_size`): winner 128 with validation MSE 0.000281151; relative to the previous stage this worsened by 1.18064e-06.

## Interpretation

- **Validation winner:** LSTM achieved the lowest validation MSE at 0.000240184.
- **Testing winner:** GRU achieved the lowest testing MSE at 9.08801e-05.
- **Directional winner:** RNN achieved the highest directional accuracy at 55.35%.
- Across the current tuning archive, recurrent models stayed tightly grouped, while the Transformer remained materially higher-loss than the recurrent models after tuning.

## Figure

![Best tuned model losses](figures\hyperparameter_model_loss_summary_next_return.svg)

The figure uses one shared y-axis across three subplots so the training, testing, and validation losses remain directly comparable.
