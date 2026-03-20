# Hyper-Parameter Impact Report

This report summarises how the tuning workflow changed model performance and compares the best tuned runs across models.

## Best tuned configuration by model

| Model | Best validation MSE | Best testing MSE | Best training MSE | MAE | DA | Hyperparameters | Run ID |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| LSTM | 0.000134793 | 9.75223e-05 | 0.000119182 | 0.00647722 | 56.93% | `{"hidden": 64, "layers": 2}` | `lstm_experiment-20260319T090216Z` |
| GRU | 0.000135122 | 0.000100104 | 0.00011706 | 0.00649861 | 55.94% | `{"hidden": 64, "layers": 2}` | `gru_experiment-20260319T090755Z` |
| RNN | 0.000136693 | 0.000105177 | 0.000126819 | 0.00678929 | 47.69% | `{"hidden": 64, "layers": 2}` | `rnn_experiment-20260319T091307Z` |
| Transformer | 0.000140588 | 0.000108577 | 0.000284769 | 0.00681754 | 50.00% | `{"d_model": 32, "dropout": 0.1, "nhead": 8, "num_layers": 1}` | `transformer_experiment-20260319T094918Z` |

## Stage-by-stage hyper-parameter impact

The tuning workflow was sequential, so each stage winner was selected while earlier winners stayed frozen.

### GRU

- Stage 1 (`lr`): winner 0.001 with validation MSE 0.000135122; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 32 with validation MSE 0.000136346; relative to the previous stage this worsened by 1.22382e-06.
- Stage 3 (`layers`): winner 3 with validation MSE 0.000135815; relative to the previous stage this improved by 5.30509e-07.
- Stage 4 (`batch_size`): winner 64 with validation MSE 0.000135147; relative to the previous stage this improved by 6.6817e-07.

### LSTM

- Stage 1 (`lr`): winner 0.0005 with validation MSE 0.000134793; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 128 with validation MSE 0.000135944; relative to the previous stage this worsened by 1.15077e-06.
- Stage 3 (`layers`): winner 1 with validation MSE 0.000137246; relative to the previous stage this worsened by 1.30158e-06.
- Stage 4 (`batch_size`): winner 32 with validation MSE 0.000135338; relative to the previous stage this improved by 1.90745e-06.

### RNN

- Stage 1 (`lr`): winner 0.0001 with validation MSE 0.000143939; relative to the previous stage this n/a.
- Stage 2 (`hidden`): winner 64 with validation MSE 0.000144643; relative to the previous stage this worsened by 7.04046e-07.
- Stage 3 (`layers`): winner 2 with validation MSE 0.000136693; relative to the previous stage this improved by 7.94986e-06.
- Stage 4 (`batch_size`): winner 32 with validation MSE 0.000140908; relative to the previous stage this worsened by 4.21454e-06.

### Transformer

- Stage 1 (`lr`): winner 0.001 with validation MSE 0.000151465; relative to the previous stage this n/a.
- Stage 2 (`d_model`): winner 32 with validation MSE 0.000149703; relative to the previous stage this improved by 1.76238e-06.
- Stage 3 (`num_layers`): winner 1 with validation MSE 0.000146991; relative to the previous stage this improved by 2.71225e-06.
- Stage 4 (`nhead`): winner 8 with validation MSE 0.000140588; relative to the previous stage this improved by 6.40275e-06.
- Stage 5 (`batch_size`): winner 32 with validation MSE 0.000140705; relative to the previous stage this worsened by 1.1731e-07.

## Interpretation

- **Validation winner:** LSTM achieved the lowest validation MSE at 0.000134793.
- **Testing winner:** LSTM achieved the lowest testing MSE at 9.75223e-05.
- **Directional winner:** LSTM achieved the highest directional accuracy at 56.93%.
- Across the current tuning archive, recurrent models stayed tightly grouped, while the Transformer remained materially higher-loss than the recurrent models after tuning.

## Figure

![Best tuned model losses](figures/hyperparameter_model_loss_summary.svg)

The figure uses one shared y-axis across three subplots so the training, testing, and validation losses remain directly comparable.
