# Codebase Tour (Beginner-Friendly)

This guide explains the current repository structure and workflow for forecasting **SPY log returns** with baseline and neural models.

## 1) Project goal (current behavior)

The code predicts a **future SPY daily log return** from a window of past returns.

- Input: the last `SEQ_LEN` returns (default: 30 days).
- Target: the return at `HORIZON` steps ahead (default: 10 trading days ahead).
- Output: metrics and plots saved to `reports/`.

So this is currently a **multi-step-ahead forecasting setup** (not just next-day forecasting).

## 2) End-to-end pipeline

Most scripts follow the same sequence:

1. Download SPY data from Yahoo Finance.
2. Compute log returns.
3. Build either:
   - lag-feature tables (for baselines), or
   - sequence tensors shaped `(N, T, 1)` (for neural models).
4. Split data chronologically into train/val/test.
5. Fit scaler on train inputs and transform val/test.
6. Train model(s).
7. Evaluate with MAE, MSE, and Directional Accuracy (DA).
8. Save metrics, diagnostics, and figures into `reports/`.

## 3) Directory walkthrough

### `src/common/` (shared infrastructure)

- `config.py`
  - Central hyperparameters and paths:
    - data scope (`TICKER`, `START`)
    - task setup (`SEQ_LEN`, `HORIZON`, `LAGS`)
    - training controls (`EPOCHS`, `PATIENCE`, `MIN_DELTA`, `MIN_EPOCHS`, smoothing window)
  - Creates `reports/` and `reports/figures/` automatically.
  - Sets `DEVICE` (`cuda` if available, otherwise `cpu`).

- `data.py`
  - `load_data(...)`: downloads SPY and computes log returns.
  - `make_lag_features(...)`: lag matrix for baseline models.
  - `build_sequences(...)`: sequence targets with configurable horizon.
  - `chronological_split(...)`: time-ordered split into train/val/test.
  - `SeqDataset`: lightweight PyTorch dataset wrapper.

- `metrics.py`
  - `evaluate_preds(...)` returns:
    - `MAE`
    - `MSE`
    - `DA` (directional sign accuracy)

- `train.py`
  - Shared PyTorch training loop for neural models.
  - Uses Adam + MSE loss.
  - Early stopping uses:
    - optional validation-loss smoothing,
    - minimum epoch warmup,
    - patience/min-delta logic.
  - Saves best weights: `reports/<Model>.pt`.
  - Writes epoch diagnostics: `reports/<model>_diagnostics.json`.
  - Appends standardized experiment records to `reports/experiment_log.jsonl` and `reports/experiment_log.csv`.
  - Saves plots to `reports/figures/`:
    - loss curves,
    - test prediction slice,
    - true-vs-pred scatter.

### `src/baselines/`

- `main.py`
  - Runs:
    1. `Persistence` baseline (`ŷ = current return`),
    2. `LinearRegression` on lag features.
  - Saves:
    - `reports/metrics_baselines.csv`
    - `reports/metrics_baselines.json`

### `src/rnn/`, `src/lstm/`, `src/gru/`, `src/transformer/`

Each model folder has:
- `model.py`: architecture definition.
- `train.py`: model-specific entrypoint that prepares data, calls shared trainer, and saves `metrics_<model>.json`.

These entrypoints are intentionally parallel so experiments stay comparable.

### `src/comparison/`

- `main.py`
  - Builds one shared dataset split and scaling pipeline.
  - Computes a flattened-sequence **Baseline-LR** reference.
  - Trains RNN, LSTM, GRU, and Transformer with the shared trainer.
  - Saves:
    - `reports/metrics_comparison.csv`
    - `reports/metrics_comparison.json`
    - `reports/model_comparison_record.json`
    - `reports/model_comparison_record.csv`
    - `reports/figures/comparison_losses.png` (train/val/test MSE bar charts)

## 4) Important outputs in `reports/`

After running experiments, expect files such as:

- Model checkpoints: `RNN.pt`, `LSTM.pt`, `GRU.pt`, `Transformer.pt`
- Metrics JSON/CSV files (`metrics_*.json`, `metrics_*.csv`)
- Diagnostics JSON per neural model (`*_diagnostics.json`)
- Standardized experiment logs: `experiment_log.jsonl`, `experiment_log.csv`
- Model-comparison summary records: `model_comparison_record.json`, `model_comparison_record.csv`
- Figures in `reports/figures/`:
  - `loss_<Model>.png`
  - `<model>_pred_slice.png`
  - `<model>_scatter.png`
  - `comparison_losses.png`

## 5) How to run (typical workflow)

From repository root:

```bash
pip install -r requirements.txt
python -m src.baselines.main
python -m src.gru.train
python -m src.comparison.main
```

You can run other individual models similarly:

```bash
python -m src.rnn.train
python -m src.lstm.train
python -m src.transformer.train
```

## 6) What to tweak first

Start with `src/common/config.py`:

- `HORIZON`: how many days ahead to predict.
- `SEQ_LEN`: lookback window size.
- `LAGS`: baseline feature window size.
- `EPOCHS`, `PATIENCE`, `MIN_DELTA`, `MIN_EPOCHS`: early-stopping/training behavior.
- `LR`, `BATCH_SIZE`: optimization behavior.
- `TRAIN_RATIO`, `VAL_RATIO`: split proportions.

## 7) Mental model of the repository

Think of the project in three layers:

1. **Shared pipeline utilities** (`src/common/`)
2. **Experiment entrypoints** (`src/*/train.py`, `src/baselines/main.py`, `src/comparison/main.py`)
3. **Generated artifacts** (`reports/`)

Once you understand those layers, navigating and extending the codebase is straightforward.
