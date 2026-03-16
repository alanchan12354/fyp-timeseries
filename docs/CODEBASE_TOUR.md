# Codebase Tour (Beginner-Friendly)

This guide explains the project like a "map" so you can understand what each part does, even if you are new to coding.

## 1) What this project is trying to do

The project tries to **predict tomorrow's SPY daily return** (SPY is an ETF tracking the S&P 500) using past returns.

In simple words:
- Input: last few days of return data.
- Output: model's guess for next day return.
- Then it compares different model types to see which works best.

## 2) The big picture flow

No matter which model you run, the workflow is mostly:

1. Download price data from Yahoo Finance.
2. Convert prices into daily log returns.
3. Build features (lag vectors for baselines or sequences for neural nets).
4. Split data in time order (train/validation/test).
5. Scale input features.
6. Train model(s).
7. Evaluate with MAE, RMSE, and Directional Accuracy.
8. Save metrics and plots in `reports/`.

## 3) Folder-by-folder walkthrough

## `src/common/` (shared building blocks)

This is the most important folder to understand first.

- `config.py`
  - Holds global settings like ticker (`SPY`), start date, sequence length, train/val ratios, epochs, learning rate, and output directories.
  - Also creates `reports/` and `reports/figures/` if they don't exist.

- `data.py`
  - `load_data(...)`: downloads data and computes log returns.
  - `make_lag_features(...)`: creates classic tabular lag features for baseline models.
  - `build_sequences(...)`: creates 3D sequence tensors for neural models `(N, sequence_length, 1)`.
  - `chronological_split(...)`: split by time order (no shuffling across time).
  - `SeqDataset`: wraps arrays into a PyTorch dataset.

- `train.py`
  - Generic training loop used by all neural models.
  - Uses Adam optimizer + MSE loss.
  - Tracks train/validation loss each epoch.
  - Uses early stopping via patience.
  - Saves best model weights to `reports/<Model>.pt`.
  - Creates learning-curve and prediction/scatter plots.
  - Evaluates test predictions and returns metric dictionary.

- `metrics.py`
  - Defines:
    - MAE (absolute error)
    - RMSE (square-error based)
    - DA = directional accuracy (correct sign up/down)

## `src/baselines/`

- `main.py`
  - Runs two simple non-neural models:
    1. **Persistence**: predicts next return as current return.
    2. **Linear Regression** on lag features.
  - Saves results to:
    - `reports/metrics_baselines.csv`
    - `reports/metrics_baselines.json`

This is a "sanity check" and benchmark. If fancy models cannot beat this, they're not adding much value.

## `src/rnn/`, `src/lstm/`, `src/gru/`, `src/transformer/`

Each of these has:
- `model.py`: network architecture.
- `train.py`: data prep + training call + save metrics json.

Pattern is intentionally similar across model types so you can compare apples-to-apples.

### Model differences (high level)

- **RNN**: simplest recurrent model.
- **LSTM**: recurrent model with memory gates to handle longer dependencies.
- **GRU**: similar idea to LSTM but slightly simpler.
- **Transformer**: attention-based encoder model with positional encoding parameter.

## `src/comparison/`

- `main.py`
  - Trains all neural models one-by-one using shared pipeline.
  - Collects metrics into one table.
  - Saves:
    - `reports/metrics_comparison.csv`
    - `reports/figures/comparison_all.png` bar chart.

## `reports/`

Output artifacts from runs:
- `*.pt`: saved best model weights.
- `metrics_*.json` / `*.csv`: result summaries.
- figures in `reports/figures/` (loss curves, slices, scatter, comparison bars).

## `src/archive/`

Older one-off scripts from early experimentation. Useful for historical reference, but not the main maintained pipeline.

## 4) How to run things (practical sequence)

If you're new, this sequence gives confidence fast:

1. Install requirements.
2. Run baselines.
3. Run one neural model (like GRU).
4. Run comparison script.
5. Open files in `reports/` to inspect outcomes.

Commands (from project root):

```bash
pip install -r requirements.txt
python -m src.baselines.main
python -m src.gru.train
python -m src.comparison.main
```

## 5) Reading outputs like a non-coder

Key metrics:
- **MAE / RMSE**: lower is better (error magnitude).
- **DA**: higher is better (direction up/down correctness).

Depending on your objective:
- If you care about error size (risk/forecast quality), focus on RMSE/MAE.
- If you care about trading direction, focus more on DA.

## 6) Current result snapshot (from existing reports)

From the included report files:
- LSTM is strong on RMSE among neural models.
- GRU is strongest on directional accuracy among neural models.
- Linear regression baseline is surprisingly competitive.
- Transformer currently underperforms in this setup.

So this codebase already demonstrates a useful real-world lesson:
**simple models can be very hard to beat on noisy financial data.**

## 7) If you want to modify behavior safely

Start in `src/common/config.py`:
- `SEQ_LEN`: lookback window length.
- `EPOCHS`: max training epochs.
- `LR`: learning rate.
- `TRAIN_RATIO`, `VAL_RATIO`: split proportions.
- `TICKER`, `START`: data source scope.

Then rerun one model and compare metrics files.

## 8) Mental model for the whole repository

Think of this project as three layers:

1. **Data + utilities** (`src/common/`)
2. **Model runners** (`src/*/train.py`)
3. **Experiment outputs** (`reports/`)

Once you understand that pattern, the repository becomes predictable and easy to navigate.
