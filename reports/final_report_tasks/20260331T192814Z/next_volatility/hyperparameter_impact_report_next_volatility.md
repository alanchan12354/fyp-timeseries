# Hyper-Parameter Impact Report

This report summarises how the tuning workflow changed model performance and compares the best tuned runs across models.

## Best tuned configuration by model

| Model | Best validation MSE | Best testing MSE | Best training MSE | MAE | DA | Hyperparameters | Run ID |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| GRU | 3.85679e-05 | 1.94626e-05 | 0.000363659 | 0.00273702 | 100.00% | `{"hidden": 128, "input_size": 8, "layers": 3, "random_seed": 42}` | `gru_experiment-20260331T200029Z` |
| LSTM | 3.89705e-05 | 1.98461e-05 | 2.08138e-05 | 0.00270235 | 100.00% | `{"hidden": 64, "input_size": 8, "layers": 3, "random_seed": 42}` | `lstm_experiment-20260331T195809Z` |
| Baseline-LR | 4.13279e-05 | 1.77972e-05 | 1.39722e-05 | 0.00257015 | 100.00% | `{"details": "{\"flattened_sequence\": true, \"model\": \"LinearRegression\", \"seq_len\": 20}", "source": "best_tuned_comparison_csv"}` | `best_tuned_lstm_comparison-20260331T201252Z-baseline-lr` |
| RNN | 4.3291e-05 | 2.11042e-05 | 1.52907e-05 | 0.0028882 | 100.00% | `{"hidden": 64, "input_size": 8, "layers": 3, "random_seed": 42}` | `rnn_experiment-20260331T200240Z` |
| Transformer | 4.62823e-05 | 2.19549e-05 | 3.18639e-05 | 0.0030066 | 100.00% | `{"d_model": 64, "dropout": 0.1, "input_size": 8, "nhead": 2, "num_layers": 1, "random_seed": 42}` | `transformer_experiment-20260331T201206Z` |

## Stage-by-stage hyper-parameter impact

The tuning workflow was sequential, so each stage winner was selected while earlier winners stayed frozen.

### Baseline-LR

- Stage 1 (`seq_len`): winner 20 with validation MSE 4.13279e-05; relative to the previous stage this n/a.

### GRU

- Stage 1 (`lr`): winner 0.001 with validation MSE 5.10359e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 4.63555e-05; relative to the previous stage this improved by 4.6804e-06.
- Stage 3 (`layers`): winner 3 with validation MSE 3.85736e-05; relative to the previous stage this improved by 7.78188e-06.
- Stage 4 (`batch_size`): winner 64 with validation MSE 3.85679e-05; relative to the previous stage this improved by 5.67789e-09.
- Stage 5 (`seq_len`): winner 30 with validation MSE 3.85727e-05; relative to the previous stage this worsened by 4.74643e-09.

### LSTM

- Stage 1 (`lr`): winner 0.0001 with validation MSE 4.31481e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 64 with validation MSE 4.31479e-05; relative to the previous stage this improved by 2.08082e-10.
- Stage 3 (`layers`): winner 3 with validation MSE 4.02571e-05; relative to the previous stage this improved by 2.89071e-06.
- Stage 4 (`batch_size`): winner 32 with validation MSE 3.92043e-05; relative to the previous stage this improved by 1.05283e-06.
- Stage 5 (`seq_len`): winner 20 with validation MSE 3.89705e-05; relative to the previous stage this improved by 2.33775e-07.

### RNN

- Stage 1 (`lr`): winner 0.001 with validation MSE 4.36602e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 64 with validation MSE 4.36618e-05; relative to the previous stage this worsened by 1.59403e-09.
- Stage 3 (`layers`): winner 3 with validation MSE 4.33012e-05; relative to the previous stage this improved by 3.60632e-07.
- Stage 4 (`batch_size`): winner 64 with validation MSE 4.3291e-05; relative to the previous stage this improved by 1.02208e-08.
- Stage 5 (`seq_len`): winner 30 with validation MSE 4.32984e-05; relative to the previous stage this worsened by 7.46067e-09.

### Transformer

- Stage 1 (`lr`): winner 0.001 with validation MSE 0.0001142; relative to the previous stage this n/a.
- Stage 2 (`d_model`): winner 64 with validation MSE 0.000130595; relative to the previous stage this worsened by 1.63954e-05.
- Stage 3 (`num_layers`): winner 1 with validation MSE 6.67371e-05; relative to the previous stage this improved by 6.3858e-05.
- Stage 4 (`nhead`): winner 2 with validation MSE 6.25187e-05; relative to the previous stage this improved by 4.21843e-06.
- Stage 5 (`batch_size`): winner 32 with validation MSE 5.37243e-05; relative to the previous stage this improved by 8.79442e-06.
- Stage 6 (`seq_len`): winner 20 with validation MSE 4.66568e-05; relative to the previous stage this improved by 7.06742e-06.

## Interpretation

- **Validation winner:** GRU achieved the lowest validation MSE at 3.85679e-05.
- **Testing winner:** Baseline-LR achieved the lowest testing MSE at 1.77972e-05.
- **Directional winner:** GRU achieved the highest directional accuracy at 100.00%.
- Across the current tuning archive, recurrent models stayed tightly grouped, while the Transformer remained materially higher-loss than the recurrent models after tuning.

## Figure

![Best tuned model losses](figures\hyperparameter_model_loss_summary_next_volatility.svg)

The figure uses one shared y-axis across three subplots so the training, testing, and validation losses remain directly comparable.
