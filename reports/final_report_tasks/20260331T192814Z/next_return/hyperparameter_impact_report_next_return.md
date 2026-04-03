# Hyper-Parameter Impact Report

This report summarises how the tuning workflow changed model performance and compares the best tuned runs across models.

## Best tuned configuration by model

| Model | Best validation MSE | Best testing MSE | Best training MSE | MAE | DA | Hyperparameters | Run ID |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| LSTM | 0.000243674 | 9.29586e-05 | 0.000143519 | 0.00665155 | 52.70% | `{"hidden": 128, "input_size": 8, "layers": 3, "random_seed": 42}` | `lstm_experiment-20260331T194102Z` |
| GRU | 0.000248078 | 9.27157e-05 | 0.000136736 | 0.0066885 | 51.57% | `{"hidden": 64, "input_size": 8, "layers": 3, "random_seed": 42}` | `gru_experiment-20260331T194321Z` |
| RNN | 0.000269011 | 0.000113957 | 0.000126426 | 0.00715052 | 47.80% | `{"hidden": 128, "input_size": 8, "layers": 1, "random_seed": 42}` | `rnn_experiment-20260331T194453Z` |
| Transformer | 0.000279233 | 0.000125509 | 0.000350308 | 0.00768097 | 49.06% | `{"d_model": 64, "dropout": 0.1, "input_size": 8, "nhead": 4, "num_layers": 2, "random_seed": 42}` | `transformer_experiment-20260331T194512Z` |
| Baseline-LR | 0.000372517 | 0.000122597 | 0.000106213 | 0.00768799 | 48.93% | `{"details": "{\"flattened_sequence\": true, \"model\": \"LinearRegression\", \"seq_len\": 30}", "source": "best_tuned_comparison_csv"}` | `best_tuned_lstm_comparison-20260331T195507Z-baseline-lr` |

## Stage-by-stage hyper-parameter impact

The tuning workflow was sequential, so each stage winner was selected while earlier winners stayed frozen.

### Baseline-LR

- Stage 1 (`seq_len`): winner 20 with validation MSE 0.000347348; relative to the previous stage this n/a.

### GRU

- Stage 1 (`lr`): winner 0.0001 with validation MSE 0.000259431; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 64 with validation MSE 0.000259432; relative to the previous stage this worsened by 1.10913e-09.
- Stage 3 (`layers`): winner 3 with validation MSE 0.000248556; relative to the previous stage this improved by 1.08761e-05.
- Stage 4 (`batch_size`): winner 64 with validation MSE 0.000248554; relative to the previous stage this improved by 1.69276e-09.
- Stage 5 (`seq_len`): winner 20 with validation MSE 0.000248082; relative to the previous stage this improved by 4.72012e-07.

### LSTM

- Stage 1 (`lr`): winner 0.0005 with validation MSE 0.000247885; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 0.000246564; relative to the previous stage this improved by 1.32113e-06.
- Stage 3 (`layers`): winner 3 with validation MSE 0.0002461; relative to the previous stage this improved by 4.63741e-07.
- Stage 4 (`batch_size`): winner 32 with validation MSE 0.000243674; relative to the previous stage this improved by 2.42565e-06.
- Stage 5 (`seq_len`): winner 30 with validation MSE 0.000243675; relative to the previous stage this worsened by 7.21102e-10.

### RNN

- Stage 1 (`lr`): winner 0.001 with validation MSE 0.000282232; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 0.000275575; relative to the previous stage this improved by 6.65774e-06.
- Stage 3 (`layers`): winner 1 with validation MSE 0.000272681; relative to the previous stage this improved by 2.8936e-06.
- Stage 4 (`batch_size`): winner 128 with validation MSE 0.000269011; relative to the previous stage this improved by 3.67018e-06.
- Stage 5 (`seq_len`): winner 30 with validation MSE 0.000269015; relative to the previous stage this worsened by 3.95896e-09.

### Transformer

- Stage 1 (`lr`): winner 0.001 with validation MSE 0.000279233; relative to the previous stage this n/a.
- Stage 2 (`d_model`): winner 64 with validation MSE 0.000292235; relative to the previous stage this worsened by 1.3002e-05.
- Stage 3 (`num_layers`): winner 1 with validation MSE 0.000303735; relative to the previous stage this worsened by 1.14992e-05.
- Stage 4 (`nhead`): winner 8 with validation MSE 0.000296064; relative to the previous stage this improved by 7.67042e-06.
- Stage 5 (`batch_size`): winner 32 with validation MSE 0.00028853; relative to the previous stage this improved by 7.53397e-06.
- Stage 6 (`seq_len`): winner 20 with validation MSE 0.000298537; relative to the previous stage this worsened by 1.00069e-05.

## Interpretation

- **Validation winner:** LSTM achieved the lowest validation MSE at 0.000243674.
- **Testing winner:** GRU achieved the lowest testing MSE at 9.27157e-05.
- **Directional winner:** LSTM achieved the highest directional accuracy at 52.70%.
- Across the current tuning archive, recurrent models stayed tightly grouped, while the Transformer remained materially higher-loss than the recurrent models after tuning.

## Figure

![Best tuned model losses](figures\hyperparameter_model_loss_summary_next_return.svg)

The figure uses one shared y-axis across three subplots so the training, testing, and validation losses remain directly comparable.
