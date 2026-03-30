# Hyper-Parameter Impact Report

This report summarises how the tuning workflow changed model performance and compares the best tuned runs across models.

## Best tuned configuration by model

| Model | Best validation MSE | Best testing MSE | Best training MSE | MAE | DA | Hyperparameters | Run ID |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| LSTM | 0.000242163 | 9.01477e-05 | 0.000200634 | 0.00651828 | 55.47% | `{"hidden": 128, "input_size": 8, "layers": 3, "random_seed": 42}` | `lstm_experiment-20260330T114154Z` |
| GRU | 0.000245734 | 9.2643e-05 | 0.000135784 | 0.00675122 | 50.19% | `{"hidden": 64, "input_size": 8, "layers": 3, "random_seed": 42}` | `gru_experiment-20260330T114345Z` |
| RNN | 0.000253168 | 0.000109641 | 0.000127584 | 0.00696879 | 52.70% | `{"hidden": 32, "input_size": 8, "layers": 3, "random_seed": 42}` | `rnn_experiment-20260330T114605Z` |
| Transformer | 0.000260058 | 0.000123406 | 0.000197555 | 0.00701141 | 52.83% | `{"d_model": 64, "dropout": 0.1, "input_size": 8, "nhead": 8, "num_layers": 2, "random_seed": 42}` | `transformer_experiment-20260330T115400Z` |

## Stage-by-stage hyper-parameter impact

The tuning workflow was sequential, so each stage winner was selected while earlier winners stayed frozen.

### GRU

- Stage 1 (`lr`): winner 0.0001 with validation MSE 0.000259204; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 64 with validation MSE 0.000259206; relative to the previous stage this worsened by 2.43082e-09.
- Stage 3 (`layers`): winner 3 with validation MSE 0.000245734; relative to the previous stage this improved by 1.34721e-05.
- Stage 4 (`batch_size`): winner 64 with validation MSE 0.000245735; relative to the previous stage this worsened by 4.3059e-10.

### LSTM

- Stage 1 (`lr`): winner 0.0005 with validation MSE 0.000248307; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 0.000246748; relative to the previous stage this improved by 1.5596e-06.
- Stage 3 (`layers`): winner 3 with validation MSE 0.000244537; relative to the previous stage this improved by 2.21042e-06.
- Stage 4 (`batch_size`): winner 32 with validation MSE 0.000242163; relative to the previous stage this improved by 2.37435e-06.

### RNN

- Stage 1 (`lr`): winner 0.0001 with validation MSE 0.000291066; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 32 with validation MSE 0.000288692; relative to the previous stage this improved by 2.37407e-06.
- Stage 3 (`layers`): winner 3 with validation MSE 0.000255642; relative to the previous stage this improved by 3.30497e-05.
- Stage 4 (`batch_size`): winner 32 with validation MSE 0.000253168; relative to the previous stage this improved by 2.47447e-06.

### Transformer

- Stage 1 (`lr`): winner 0.001 with validation MSE 0.000278321; relative to the previous stage this n/a.
- Stage 2 (`d_model`): winner 64 with validation MSE 0.0002943; relative to the previous stage this worsened by 1.59784e-05.
- Stage 3 (`num_layers`): winner 2 with validation MSE 0.000299094; relative to the previous stage this worsened by 4.79422e-06.
- Stage 4 (`nhead`): winner 8 with validation MSE 0.000290556; relative to the previous stage this improved by 8.53817e-06.
- Stage 5 (`batch_size`): winner 32 with validation MSE 0.000260058; relative to the previous stage this improved by 3.04978e-05.

## Interpretation

- **Validation winner:** LSTM achieved the lowest validation MSE at 0.000242163.
- **Testing winner:** LSTM achieved the lowest testing MSE at 9.01477e-05.
- **Directional winner:** LSTM achieved the highest directional accuracy at 55.47%.
- Across the current tuning archive, recurrent models stayed tightly grouped, while the Transformer remained materially higher-loss than the recurrent models after tuning.

## Figure

![Best tuned model losses](figures\hyperparameter_model_loss_summary_next_return.svg)

The figure uses one shared y-axis across three subplots so the training, testing, and validation losses remain directly comparable.
