# Hyper-Parameter Impact Report

This report summarises how the tuning workflow changed model performance and compares the best tuned runs across models.

## Best tuned configuration by model

| Model | Best validation MSE | Best testing MSE | Best training MSE | MAE | DA | Hyperparameters | Run ID |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| LSTM | 3.96585e-05 | 1.61064e-05 | 2.11393e-05 | 0.00293612 | 61.14% | `{"hidden": 64, "input_size": 8, "layers": 3, "random_seed": 42}` | `lstm_experiment-20260331T201604Z` |
| GRU | 4.41561e-05 | 1.54978e-05 | 2.21533e-05 | 0.00293237 | 56.98% | `{"hidden": 128, "input_size": 8, "layers": 3, "random_seed": 42}` | `gru_experiment-20260331T201746Z` |
| RNN | 5.01139e-05 | 1.65341e-05 | 2.37031e-05 | 0.00300132 | 62.03% | `{"hidden": 128, "input_size": 8, "layers": 3, "random_seed": 42}` | `rnn_experiment-20260331T202002Z` |
| Transformer | 5.73275e-05 | 2.53651e-05 | 5.84686e-05 | 0.00355665 | 54.59% | `{"d_model": 64, "dropout": 0.1, "input_size": 8, "nhead": 8, "num_layers": 1, "random_seed": 42}` | `transformer_experiment-20260331T202651Z` |
| Baseline-LR | 6.13083e-05 | 2.19421e-05 | 1.53082e-05 | 0.00347777 | 52.53% | `{"details": "{\"flattened_sequence\": true, \"model\": \"LinearRegression\", \"seq_len\": 60}", "source": "best_tuned_comparison_csv"}` | `best_tuned_lstm_comparison-20260331T203017Z-baseline-lr` |

## Stage-by-stage hyper-parameter impact

The tuning workflow was sequential, so each stage winner was selected while earlier winners stayed frozen.

### Baseline-LR

- Stage 1 (`seq_len`): winner 20 with validation MSE 5.12073e-05; relative to the previous stage this n/a.

### GRU

- Stage 1 (`lr`): winner 0.001 with validation MSE 5.56769e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 5.21571e-05; relative to the previous stage this improved by 3.51975e-06.
- Stage 3 (`layers`): winner 3 with validation MSE 4.98392e-05; relative to the previous stage this improved by 2.31792e-06.
- Stage 4 (`batch_size`): winner 128 with validation MSE 4.41564e-05; relative to the previous stage this improved by 5.68276e-06.
- Stage 5 (`seq_len`): winner 30 with validation MSE 4.41561e-05; relative to the previous stage this improved by 3.34381e-10.

### LSTM

- Stage 1 (`lr`): winner 0.001 with validation MSE 4.48323e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 64 with validation MSE 4.48316e-05; relative to the previous stage this improved by 6.60664e-10.
- Stage 3 (`layers`): winner 3 with validation MSE 4.14578e-05; relative to the previous stage this improved by 3.37385e-06.
- Stage 4 (`batch_size`): winner 128 with validation MSE 3.97703e-05; relative to the previous stage this improved by 1.68743e-06.
- Stage 5 (`seq_len`): winner 60 with validation MSE 3.96586e-05; relative to the previous stage this improved by 1.11739e-07.

### RNN

- Stage 1 (`lr`): winner 0.0005 with validation MSE 5.53164e-05; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 5.16443e-05; relative to the previous stage this improved by 3.67215e-06.
- Stage 3 (`layers`): winner 3 with validation MSE 5.15566e-05; relative to the previous stage this improved by 8.76497e-08.
- Stage 4 (`batch_size`): winner 32 with validation MSE 5.04043e-05; relative to the previous stage this improved by 1.15235e-06.
- Stage 5 (`seq_len`): winner 60 with validation MSE 5.01139e-05; relative to the previous stage this improved by 2.9037e-07.

### Transformer

- Stage 1 (`lr`): winner 0.001 with validation MSE 0.000109667; relative to the previous stage this n/a.
- Stage 2 (`d_model`): winner 64 with validation MSE 0.000118929; relative to the previous stage this worsened by 9.26225e-06.
- Stage 3 (`num_layers`): winner 1 with validation MSE 8.11507e-05; relative to the previous stage this improved by 3.77786e-05.
- Stage 4 (`nhead`): winner 8 with validation MSE 7.92198e-05; relative to the previous stage this improved by 1.93088e-06.
- Stage 5 (`batch_size`): winner 32 with validation MSE 5.73275e-05; relative to the previous stage this improved by 2.18923e-05.
- Stage 6 (`seq_len`): winner 30 with validation MSE 6.12108e-05; relative to the previous stage this worsened by 3.88331e-06.

## Interpretation

- **Validation winner:** LSTM achieved the lowest validation MSE at 3.96585e-05.
- **Testing winner:** GRU achieved the lowest testing MSE at 1.54978e-05.
- **Directional winner:** RNN achieved the highest directional accuracy at 62.03%.
- Across the current tuning archive, recurrent models stayed tightly grouped, while the Transformer remained materially higher-loss than the recurrent models after tuning.

## Figure

![Best tuned model losses](figures\hyperparameter_model_loss_summary_next_mean_return.svg)

The figure uses one shared y-axis across three subplots so the training, testing, and validation losses remain directly comparable.
