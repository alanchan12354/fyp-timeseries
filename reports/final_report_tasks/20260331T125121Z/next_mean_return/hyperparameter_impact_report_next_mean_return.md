# Hyper-Parameter Impact Report

This report summarises how the tuning workflow changed model performance and compares the best tuned runs across models.

## Best tuned configuration by model

| Model | Best validation MSE | Best testing MSE | Best training MSE | MAE | DA | Hyperparameters | Run ID |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| LSTM | 3.96583e-05 | 1.61397e-05 | 2.11392e-05 | 0.0029402 | 61.14% | `{"hidden": 64, "input_size": 8, "layers": 3, "random_seed": 42}` | `lstm_experiment-20260331T134353Z` |
| GRU | 4.41552e-05 | 1.55259e-05 | 2.21528e-05 | 0.00293636 | 56.98% | `{"hidden": 128, "input_size": 8, "layers": 3, "random_seed": 42}` | `gru_experiment-20260331T134538Z` |
| RNN | 5.01031e-05 | 1.65583e-05 | 2.37028e-05 | 0.00300434 | 62.03% | `{"hidden": 128, "input_size": 8, "layers": 3, "random_seed": 42}` | `rnn_experiment-20260331T134839Z` |
| Transformer | 5.90441e-05 | 2.60511e-05 | 5.90861e-05 | 0.00360681 | 52.83% | `{"d_model": 64, "dropout": 0.1, "input_size": 8, "nhead": 8, "num_layers": 1, "random_seed": 42}` | `transformer_experiment-20260331T135723Z` |

## Stage-by-stage hyper-parameter impact

The tuning workflow was sequential, so each stage winner was selected while earlier winners stayed frozen.

### Baseline-LR

- Stage 1 (`seq_len`): winner 20 with validation MSE 5.12073e-05; relative to the previous stage this n/a.

### GRU

- Stage 1 (`lr`): winner 0.001 with validation MSE 5.5674e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 5.21542e-05; relative to the previous stage this improved by 3.51984e-06.
- Stage 3 (`layers`): winner 3 with validation MSE 4.98347e-05; relative to the previous stage this improved by 2.31945e-06.
- Stage 4 (`batch_size`): winner 128 with validation MSE 4.41552e-05; relative to the previous stage this improved by 5.67947e-06.
- Stage 5 (`seq_len`): winner 30 with validation MSE 4.41561e-05; relative to the previous stage this worsened by 8.80156e-10.

### LSTM

- Stage 1 (`lr`): winner 0.001 with validation MSE 4.48323e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 64 with validation MSE 4.48306e-05; relative to the previous stage this improved by 1.68606e-09.
- Stage 3 (`layers`): winner 3 with validation MSE 4.14555e-05; relative to the previous stage this improved by 3.37514e-06.
- Stage 4 (`batch_size`): winner 128 with validation MSE 3.97703e-05; relative to the previous stage this improved by 1.68514e-06.
- Stage 5 (`seq_len`): winner 60 with validation MSE 3.96583e-05; relative to the previous stage this improved by 1.12038e-07.

### RNN

- Stage 1 (`lr`): winner 0.0005 with validation MSE 5.53243e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 5.16367e-05; relative to the previous stage this improved by 3.68766e-06.
- Stage 3 (`layers`): winner 3 with validation MSE 5.15545e-05; relative to the previous stage this improved by 8.22076e-08.
- Stage 4 (`batch_size`): winner 32 with validation MSE 5.04195e-05; relative to the previous stage this improved by 1.13495e-06.
- Stage 5 (`seq_len`): winner 60 with validation MSE 5.01077e-05; relative to the previous stage this improved by 3.11813e-07.

### Transformer

- Stage 1 (`lr`): winner 0.001 with validation MSE 0.000103458; relative to the previous stage this n/a.
- Stage 2 (`d_model`): winner 64 with validation MSE 0.000125086; relative to the previous stage this worsened by 2.16283e-05.
- Stage 3 (`num_layers`): winner 1 with validation MSE 7.93217e-05; relative to the previous stage this improved by 4.57646e-05.
- Stage 4 (`nhead`): winner 8 with validation MSE 8.07071e-05; relative to the previous stage this worsened by 1.38536e-06.
- Stage 5 (`batch_size`): winner 32 with validation MSE 6.03668e-05; relative to the previous stage this improved by 2.03402e-05.
- Stage 6 (`seq_len`): winner 30 with validation MSE 5.90441e-05; relative to the previous stage this improved by 1.32275e-06.

## Interpretation

- **Validation winner:** LSTM achieved the lowest validation MSE at 3.96583e-05.
- **Testing winner:** GRU achieved the lowest testing MSE at 1.55259e-05.
- **Directional winner:** RNN achieved the highest directional accuracy at 62.03%.
- Across the current tuning archive, recurrent models stayed tightly grouped, while the Transformer remained materially higher-loss than the recurrent models after tuning.

## Figure

![Best tuned model losses](figures\hyperparameter_model_loss_summary_next_mean_return.svg)

The figure uses one shared y-axis across three subplots so the training, testing, and validation losses remain directly comparable.
