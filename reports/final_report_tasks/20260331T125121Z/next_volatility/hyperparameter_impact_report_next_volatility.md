# Hyper-Parameter Impact Report

This report summarises how the tuning workflow changed model performance and compares the best tuned runs across models.

## Best tuned configuration by model

| Model | Best validation MSE | Best testing MSE | Best training MSE | MAE | DA | Hyperparameters | Run ID |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| GRU | 3.85708e-05 | 1.94298e-05 | 0.00036366 | 0.00273193 | 100.00% | `{"hidden": 128, "input_size": 8, "layers": 3, "random_seed": 42}` | `gru_experiment-20260331T132703Z` |
| LSTM | 3.90142e-05 | 1.97734e-05 | 2.08138e-05 | 0.00269503 | 100.00% | `{"hidden": 64, "input_size": 8, "layers": 3, "random_seed": 42}` | `lstm_experiment-20260331T132545Z` |
| RNN | 4.32835e-05 | 2.10737e-05 | 1.52922e-05 | 0.00288465 | 100.00% | `{"hidden": 64, "input_size": 8, "layers": 3, "random_seed": 42}` | `rnn_experiment-20260331T132945Z` |
| Transformer | 4.90387e-05 | 1.74755e-05 | 3.0873e-05 | 0.00272577 | 100.00% | `{"d_model": 64, "dropout": 0.1, "input_size": 8, "nhead": 4, "num_layers": 1, "random_seed": 42}` | `transformer_experiment-20260331T133850Z` |

## Stage-by-stage hyper-parameter impact

The tuning workflow was sequential, so each stage winner was selected while earlier winners stayed frozen.

### Baseline-LR

- Stage 1 (`seq_len`): winner 20 with validation MSE 4.13278e-05; relative to the previous stage this n/a.

### GRU

- Stage 1 (`lr`): winner 0.001 with validation MSE 5.10322e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 4.6358e-05; relative to the previous stage this improved by 4.67425e-06.
- Stage 3 (`layers`): winner 3 with validation MSE 3.85708e-05; relative to the previous stage this improved by 7.78713e-06.
- Stage 4 (`batch_size`): winner 64 with validation MSE 3.85742e-05; relative to the previous stage this worsened by 3.36379e-09.
- Stage 5 (`seq_len`): winner 30 with validation MSE 3.85737e-05; relative to the previous stage this improved by 4.58825e-10.

### LSTM

- Stage 1 (`lr`): winner 0.0001 with validation MSE 4.31479e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 64 with validation MSE 4.3147e-05; relative to the previous stage this improved by 8.54184e-10.
- Stage 3 (`layers`): winner 3 with validation MSE 4.0258e-05; relative to the previous stage this improved by 2.88904e-06.
- Stage 4 (`batch_size`): winner 32 with validation MSE 3.92048e-05; relative to the previous stage this improved by 1.05323e-06.
- Stage 5 (`seq_len`): winner 20 with validation MSE 3.90146e-05; relative to the previous stage this improved by 1.90204e-07.

### RNN

- Stage 1 (`lr`): winner 0.001 with validation MSE 4.36518e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 64 with validation MSE 4.36636e-05; relative to the previous stage this worsened by 1.17837e-08.
- Stage 3 (`layers`): winner 3 with validation MSE 4.32908e-05; relative to the previous stage this improved by 3.72727e-07.
- Stage 4 (`batch_size`): winner 64 with validation MSE 4.32835e-05; relative to the previous stage this improved by 7.37671e-09.
- Stage 5 (`seq_len`): winner 30 with validation MSE 4.32884e-05; relative to the previous stage this worsened by 4.98025e-09.

### Transformer

- Stage 1 (`lr`): winner 0.001 with validation MSE 0.000121075; relative to the previous stage this n/a.
- Stage 2 (`d_model`): winner 64 with validation MSE 0.000131736; relative to the previous stage this worsened by 1.06605e-05.
- Stage 3 (`num_layers`): winner 1 with validation MSE 5.12317e-05; relative to the previous stage this improved by 8.0504e-05.
- Stage 4 (`nhead`): winner 4 with validation MSE 5.17059e-05; relative to the previous stage this worsened by 4.74216e-07.
- Stage 5 (`batch_size`): winner 32 with validation MSE 5.94099e-05; relative to the previous stage this worsened by 7.70403e-06.
- Stage 6 (`seq_len`): winner 60 with validation MSE 4.90387e-05; relative to the previous stage this improved by 1.03712e-05.

## Interpretation

- **Validation winner:** GRU achieved the lowest validation MSE at 3.85708e-05.
- **Testing winner:** Transformer achieved the lowest testing MSE at 1.74755e-05.
- **Directional winner:** GRU achieved the highest directional accuracy at 100.00%.
- Across the current tuning archive, recurrent models stayed tightly grouped, while the Transformer remained materially higher-loss than the recurrent models after tuning.

## Figure

![Best tuned model losses](figures\hyperparameter_model_loss_summary_next_volatility.svg)

The figure uses one shared y-axis across three subplots so the training, testing, and validation losses remain directly comparable.
