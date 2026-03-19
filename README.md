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
reports/               # Generated metrics, plots, saved models, and experiment logs
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

## Experiment Records

Training and comparison scripts now also maintain experiment records in `reports/`, including:
- `experiment_log.jsonl`, which keeps the full structured record for each run
- `experiment_log.csv`, which is intentionally simplified for the FYP fine-tuning section
- `model_comparison_record.json` / `model_comparison_record.csv` for shared comparison summaries

For the fine-tuning write-up, the simplified CSV view keeps the core tuning knobs front and center:
- hidden size for recurrent models, mirrored from `d_model` for Transformer rows
- Transformer-specific `d_model`, `nhead`, and `dropout` columns
- number of layers (works for both recurrent and Transformer runs)
- learning rate
- batch size

It also keeps a reduced set of result columns for quick comparison, such as validation/test metrics and the best epoch, while the JSONL log preserves the richer per-run metadata.

For tuning runs, you can now choose between:
- `python -m src.tuning.main --session-mode reset` for a full reset that removes old `reports/` artifacts before a fresh session
- `python -m src.tuning.main --session-mode append` to keep prior history and add new runs on top

The reset mode prints every removed file so you can confirm that the new tuning session started from a clean slate.

## New to the Codebase?

Read the beginner-friendly walkthrough in `docs/CODEBASE_TOUR.md` for a full end-to-end explanation of how the project is structured and how data flows through it.
