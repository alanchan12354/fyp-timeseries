# Hyper-Parameter Impact Report

This report summarises how the tuning workflow changed model performance and compares the best tuned runs across models.

## Best tuned configuration by model

| Model | Best validation MSE | Best testing MSE | Best training MSE | MAE | DA | Hyperparameters | Run ID |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| LSTM | 4.03748e-05 | 2.22916e-05 | 1.97636e-05 | 0.00275702 | 100.00% | `{"hidden": 32, "input_size": 8, "layers": 2, "random_seed": 42}` | `lstm_experiment-20260330T115955Z` |
| RNN | 4.19394e-05 | 2.26662e-05 | 1.6569e-05 | 0.00291918 | 100.00% | `{"hidden": 64, "input_size": 8, "layers": 2, "random_seed": 42}` | `rnn_experiment-20260330T120319Z` |
| GRU | 4.42316e-05 | 2.22739e-05 | 1.27832e-05 | 0.00282795 | 99.87% | `{"hidden": 64, "input_size": 8, "layers": 2, "random_seed": 42}` | `gru_experiment-20260330T120107Z` |
| Transformer | 5.40284e-05 | 2.70178e-05 | 5.23603e-05 | 0.00370071 | 99.87% | `{"d_model": 32, "dropout": 0.1, "input_size": 8, "nhead": 2, "num_layers": 1, "random_seed": 42}` | `transformer_experiment-20260330T121025Z` |

## Stage-by-stage hyper-parameter impact

The tuning workflow was sequential, so each stage winner was selected while earlier winners stayed frozen.

### GRU

- Stage 1 (`lr`): winner 0.001 with validation MSE 4.42355e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 64 with validation MSE 4.42363e-05; relative to the previous stage this worsened by 7.3612e-10.
- Stage 3 (`layers`): winner 2 with validation MSE 4.42316e-05; relative to the previous stage this improved by 4.67549e-09.
- Stage 4 (`batch_size`): winner 64 with validation MSE 4.42341e-05; relative to the previous stage this worsened by 2.47583e-09.

### LSTM

- Stage 1 (`lr`): winner 0.0005 with validation MSE 4.21845e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 32 with validation MSE 4.03759e-05; relative to the previous stage this improved by 1.80862e-06.
- Stage 3 (`layers`): winner 2 with validation MSE 4.03757e-05; relative to the previous stage this improved by 1.26671e-10.
- Stage 4 (`batch_size`): winner 64 with validation MSE 4.03748e-05; relative to the previous stage this improved by 9.41032e-10.

### RNN

- Stage 1 (`lr`): winner 0.001 with validation MSE 4.31696e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 64 with validation MSE 4.31698e-05; relative to the previous stage this worsened by 1.78049e-10.
- Stage 3 (`layers`): winner 2 with validation MSE 4.31748e-05; relative to the previous stage this worsened by 4.95585e-09.
- Stage 4 (`batch_size`): winner 32 with validation MSE 4.19394e-05; relative to the previous stage this improved by 1.23537e-06.

### Transformer

- Stage 1 (`lr`): winner 0.0005 with validation MSE 0.000166521; relative to the previous stage this n/a.
- Stage 2 (`d_model`): winner 32 with validation MSE 0.000109644; relative to the previous stage this improved by 5.6877e-05.
- Stage 3 (`num_layers`): winner 1 with validation MSE 8.50742e-05; relative to the previous stage this improved by 2.45697e-05.
- Stage 4 (`nhead`): winner 2 with validation MSE 6.6401e-05; relative to the previous stage this improved by 1.86732e-05.
- Stage 5 (`batch_size`): winner 32 with validation MSE 5.40284e-05; relative to the previous stage this improved by 1.23726e-05.

## Interpretation

- **Validation winner:** LSTM achieved the lowest validation MSE at 4.03748e-05.
- **Testing winner:** GRU achieved the lowest testing MSE at 2.22739e-05.
- **Directional winner:** LSTM achieved the highest directional accuracy at 100.00%.
- Across the current tuning archive, recurrent models stayed tightly grouped, while the Transformer remained materially higher-loss than the recurrent models after tuning.

## Figure

![Best tuned model losses](figures\hyperparameter_model_loss_summary_next_volatility.svg)

The figure uses one shared y-axis across three subplots so the training, testing, and validation losses remain directly comparable.
