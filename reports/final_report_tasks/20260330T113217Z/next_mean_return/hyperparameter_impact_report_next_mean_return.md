# Hyper-Parameter Impact Report

This report summarises how the tuning workflow changed model performance and compares the best tuned runs across models.

## Best tuned configuration by model

| Model | Best validation MSE | Best testing MSE | Best training MSE | MAE | DA | Hyperparameters | Run ID |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| LSTM | 4.07353e-05 | 1.74526e-05 | 0.000357733 | 0.00310874 | 59.19% | `{"hidden": 128, "input_size": 8, "layers": 2, "random_seed": 42}` | `lstm_experiment-20260330T121450Z` |
| RNN | 4.50626e-05 | 1.5502e-05 | 1.99052e-05 | 0.00295216 | 52.64% | `{"hidden": 128, "input_size": 8, "layers": 3, "random_seed": 42}` | `rnn_experiment-20260330T121820Z` |
| GRU | 4.69386e-05 | 1.94561e-05 | 0.00018913 | 0.0035416 | 41.31% | `{"hidden": 128, "input_size": 8, "layers": 3, "random_seed": 42}` | `gru_experiment-20260330T121617Z` |
| Transformer | 4.77619e-05 | 1.99666e-05 | 7.28211e-05 | 0.00318436 | 52.64% | `{"d_model": 64, "dropout": 0.1, "input_size": 8, "nhead": 4, "num_layers": 3, "random_seed": 42}` | `transformer_experiment-20260330T122638Z` |

## Stage-by-stage hyper-parameter impact

The tuning workflow was sequential, so each stage winner was selected while earlier winners stayed frozen.

### GRU

- Stage 1 (`lr`): winner 0.0005 with validation MSE 5.59779e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 5.50289e-05; relative to the previous stage this improved by 9.48992e-07.
- Stage 3 (`layers`): winner 3 with validation MSE 4.95617e-05; relative to the previous stage this improved by 5.46726e-06.
- Stage 4 (`batch_size`): winner 32 with validation MSE 4.69386e-05; relative to the previous stage this improved by 2.62307e-06.

### LSTM

- Stage 1 (`lr`): winner 0.001 with validation MSE 4.59944e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 4.07355e-05; relative to the previous stage this improved by 5.25897e-06.
- Stage 3 (`layers`): winner 2 with validation MSE 4.07367e-05; relative to the previous stage this worsened by 1.19655e-09.
- Stage 4 (`batch_size`): winner 64 with validation MSE 4.07353e-05; relative to the previous stage this improved by 1.3178e-09.

### RNN

- Stage 1 (`lr`): winner 0.001 with validation MSE 5.36911e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 5.10984e-05; relative to the previous stage this improved by 2.59278e-06.
- Stage 3 (`layers`): winner 3 with validation MSE 4.50635e-05; relative to the previous stage this improved by 6.03485e-06.
- Stage 4 (`batch_size`): winner 64 with validation MSE 4.50626e-05; relative to the previous stage this improved by 8.76925e-10.

### Transformer

- Stage 1 (`lr`): winner 0.0005 with validation MSE 0.000120771; relative to the previous stage this n/a.
- Stage 2 (`d_model`): winner 64 with validation MSE 0.000108386; relative to the previous stage this improved by 1.23847e-05.
- Stage 3 (`num_layers`): winner 3 with validation MSE 9.41178e-05; relative to the previous stage this improved by 1.42684e-05.
- Stage 4 (`nhead`): winner 4 with validation MSE 9.42428e-05; relative to the previous stage this worsened by 1.25067e-07.
- Stage 5 (`batch_size`): winner 32 with validation MSE 4.77619e-05; relative to the previous stage this improved by 4.64809e-05.

## Interpretation

- **Validation winner:** LSTM achieved the lowest validation MSE at 4.07353e-05.
- **Testing winner:** RNN achieved the lowest testing MSE at 1.5502e-05.
- **Directional winner:** LSTM achieved the highest directional accuracy at 59.19%.
- Across the current tuning archive, recurrent models stayed tightly grouped, while the Transformer remained materially higher-loss than the recurrent models after tuning.

## Figure

![Best tuned model losses](figures\hyperparameter_model_loss_summary_next_mean_return.svg)

The figure uses one shared y-axis across three subplots so the training, testing, and validation losses remain directly comparable.
