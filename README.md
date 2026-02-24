# Time Series Neural Models Project

This project implements various neural network models for time series forecasting (SPY daily returns).
The codebase has been refactored into modular components.

## Project Structure

```
src/
  ├── baselines/       # Baseline models (Persistence, Linear Regression)
  ├── lstm/            # LSTM model implementation
  ├── rnn/             # RNN model implementation
  ├── gru/             # GRU model implementation
  ├── transformer/     # Transformer model implementation
  ├── comparison/      # Script to compare all models
  ├── common/          # Shared utilities (data loading, training loop, config)
  └── archive/         # Old scripts from previous experiments
reports/               # Generated metrics, plots, and saved models
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure you are in the project root directory.

## Running Experiments

Run the scripts as modules to ensure imports work correctly.

### 1. Baselines
```bash
python -m src.baselines.main
```

### 2. Individual Models
Train and evaluate specific models:

```bash
python -m src.lstm.train
python -m src.gru.train
python -m src.rnn.train
python -m src.transformer.train
```

### 3. Model Comparison
Train all models and generate a comparison report:

```bash
python -m src.comparison.main
```

## Configuration

You can adjust hyperparameters (SEQ_LEN, LR, EPOCHS, etc.) in `src/common/config.py`.
