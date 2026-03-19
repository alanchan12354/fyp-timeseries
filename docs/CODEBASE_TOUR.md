# Codebase Tour (Beginner-Friendly)

This guide explains the current repository structure and workflow for forecasting **SPY daily log returns** with baseline and neural models.

## 1) Project goal (current behavior)

The repository now supports **two related experiment families**.

### A. Baseline lag-feature workflow

The standalone baseline script uses lag features and predicts the **next-day** return:

- Input: `LAGS` recent daily log returns.
- Target: `r_(t+1)`.
- Entrypoint: `src/baselines/main.py`.

### B. Sequence-model workflow

The neural-model pipeline uses a fixed lookback sequence and predicts a configurable horizon ahead:

- Input: the last `SEQ_LEN` returns, shaped `(N, T, 1)`.
- Target: the return at `t + HORIZON`.
- Entrypoints: `src/rnn/train.py`, `src/lstm/train.py`, `src/gru/train.py`, `src/transformer/train.py`, `src/comparison/main.py`, `src/tuning/main.py`.

With the current defaults in `src/common/config.py`:

- `SEQ_LEN = 30`
- `HORIZON = 10`
- `LAGS = 30`

So the repository is **not using one single forecasting target everywhere**. That distinction matters when interpreting results.

## 2) End-to-end pipeline

Most experiment scripts follow a shared sequence:

1. Download SPY data with `yfinance`.
2. Compute daily log returns from adjusted close data.
3. Build either:
   - lag-feature matrices for baseline models, or
   - sequence tensors for neural models.
4. Split data chronologically into train/validation/test.
5. Fit a scaler on training inputs and apply it to validation/test inputs.
6. Train the model or baseline.
7. Evaluate with MAE, MSE, and Directional Accuracy (DA).
8. Save metrics, plots, diagnostics, and structured experiment records under `reports/`.

## 3) Directory walkthrough

### `src/common/` — shared infrastructure

#### `config.py`
Central project defaults:

- market/data scope (`TICKER`, `START`),
- task definition (`SEQ_LEN`, `HORIZON`, `LAGS`),
- split ratios,
- training controls (`EPOCHS`, `PATIENCE`, `MIN_DELTA`, `MIN_EPOCHS`, `TRAIN_LOG_EVERY`),
- scheduler settings,
- reports/figure output paths,
- `DEVICE` selection (`cuda` if available, otherwise `cpu`).

#### `runtime_config.py`
Defines `RuntimeTrainingConfig`, which allows neural entrypoints to combine:

- defaults from `config.py`,
- explicit config objects,
- dictionaries,
- CLI overrides.

It also defines the shared command-line flags used by the neural training scripts.

#### `data.py`
Core data utilities:

- `load_data(...)`: downloads SPY data and computes log returns.
- `make_lag_features(...)`: builds the lag-table baseline dataset.
- `build_sequences(...)`: builds `(N, seq_len, 1)` sequence inputs with a configurable horizon target.
- `chronological_split(...)`: performs time-ordered splitting.
- `SeqDataset`: thin PyTorch dataset wrapper.

#### `experiment.py`
Builds a ready-to-train sequence experiment package through `prepare_sequence_experiment_run(...)`.
It bundles together:

- raw returns,
- scaled train/val/test arrays,
- train/val data loaders,
- split metadata,
- a reporting context for logging.

This is what the single-model neural entrypoints reuse.

#### `train.py`
Shared PyTorch training loop for the neural models.

It handles:

- Adam optimizer with MSE loss,
- optional learning-rate schedulers (`none`, `plateau`, `cosine`),
- validation-loss smoothing for early stopping,
- minimum-epoch warmup before patience starts counting down,
- checkpointing best weights to `reports/<Model>.pt`,
- saving diagnostics JSON,
- generating plots,
- appending structured experiment records.

Saved neural artifacts include:

- checkpoint: `reports/<Model>.pt`
- diagnostics: `reports/<model>_diagnostics.json`
- plots in `reports/figures/`:
  - `loss_<Model>.png`
  - `<model>_pred_slice.png`
  - `<model>_scatter.png`

#### `reporting.py`
Owns experiment logging, comparison records, and tuning-artifact reset utilities.

Important responsibilities:

- build run contexts with timestamps, split metadata, training metadata, and environment metadata,
- append JSONL and CSV experiment logs,
- write comparison-summary records,
- generate tuning summary CSVs,
- clear tuning/report artifacts for a fresh session.

The current reporting layer captures reproducibility metadata such as:

- Python version,
- platform,
- selected device,
- git commit,
- package versions for major dependencies.

#### `neural_entrypoint.py`
A thin helper layer that standardizes:

- parser creation,
- runtime-config resolution,
- shared sequence-data preparation for neural models.

### `src/baselines/`

#### `main.py`
Runs the standalone baseline benchmark workflow:

1. download returns,
2. create lag features,
3. split chronologically,
4. evaluate:
   - `Persistence`,
   - `LinearRegression`,
5. save baseline metrics and experiment-log rows.

Important nuance: this baseline script is configured for **next-day prediction**, not `HORIZON`-ahead prediction.

### `src/rnn/`, `src/lstm/`, `src/gru/`, `src/transformer/`

Each model directory has:

- `model.py`: the architecture definition,
- `train.py`: the runtime-configurable entrypoint that prepares a shared sequence experiment and calls the common trainer.

These entrypoints accept CLI hyperparameter overrides, which is useful for manual sweeps or scripted tuning.

### `src/comparison/`

#### `main.py`
Runs the shared comparison workflow for sequence models.

What it does:

- builds one common `SEQ_LEN` / `HORIZON` dataset split,
- scales the shared split once,
- fits a flattened-sequence linear regression reference (`Baseline-LR`),
- trains RNN, LSTM, GRU, and Transformer on the same split,
- writes comparison metrics and a comparison-plot figure,
- writes a model-comparison record selecting the best model by validation MSE.

This file is the best source for the repository's **apples-to-apples sequence comparison**.

### `src/tuning/`

#### `main.py`
Runs staged hyperparameter sweeps.

Default stage order:

- recurrent models: `lr -> hidden -> layers -> batch_size`
- transformer: `lr -> d_model -> num_layers -> nhead -> batch_size`

Capabilities include:

- tune one model or all models,
- provide overrides from a JSON file or inline JSON,
- reset or append tuning outputs,
- dry-run the resolved plan,
- write per-stage run logs and winner summaries.

## 4) Important outputs in `reports/`

Depending on which workflows you run, you may see files such as:

- checkpoints: `RNN.pt`, `LSTM.pt`, `GRU.pt`, `Transformer.pt`
- per-model metrics: `metrics_gru.json`, `metrics_lstm.json`, etc.
- summary metrics:
  - `metrics_baselines.csv`
  - `metrics_baselines.json`
  - `metrics_comparison.csv`
  - `metrics_comparison.json`
- diagnostics JSON: `*_diagnostics.json`
- experiment logs:
  - `experiment_log.jsonl` (full structured records)
  - `experiment_log.csv` (flattened summary view)
- comparison summaries:
  - `model_comparison_record.json`
  - `model_comparison_record.csv`
- tuning summaries:
  - `tuning_runs.csv`
  - `tuning_winners.csv`
  - `tuning_all_runs.csv`
  - `tuning_best_configs.csv`
- figures:
  - `loss_<Model>.png`
  - `<model>_pred_slice.png`
  - `<model>_scatter.png`
  - `comparison_losses.png`

## 5) How to run

From the repository root:

```bash
pip install -r requirements.txt
python -m src.baselines.main
python -m src.gru.train --learning-rate 5e-4 --recurrent-hidden-size 128
python -m src.comparison.main
python -m src.tuning.main --model all --session-mode append
```

## 6) What to tweak first

Start with `src/common/config.py` if you want to change global defaults:

- `HORIZON`: how far ahead sequence models predict.
- `SEQ_LEN`: neural-model lookback window.
- `LAGS`: baseline lag window.
- `TRAIN_RATIO`, `VAL_RATIO`: chronological split proportions.
- `EPOCHS`, `PATIENCE`, `MIN_DELTA`, `MIN_EPOCHS`: early-stopping behavior.
- `LR`, `BATCH_SIZE`: default optimization settings.
- scheduler config values if you want to test learning-rate adaptation.

If you only need a one-off training change, prefer CLI overrides on the neural entrypoints instead of editing config constants.

## 7) Mental model of the repository

A useful way to think about the project is in four layers:

1. **Task and data definition** — `src/common/config.py`, `src/common/data.py`
2. **Prepared experiment workflows** — `src/common/experiment.py`, `src/common/neural_entrypoint.py`
3. **Training/reporting orchestration** — `src/common/train.py`, `src/common/reporting.py`
4. **User-facing entrypoints** — `src/baselines/main.py`, `src/*/train.py`, `src/comparison/main.py`, `src/tuning/main.py`

Once you keep those layers separate, the refactored codebase is much easier to navigate.
