# Codebase Tour (Beginner-Friendly)

This guide explains the current repository structure and workflow for a **four-task forecasting setup** (`sine_next_day`, `next_return`, `next_volatility`, `next_mean_return`) with neural models plus a shared linear-regression comparison baseline.

## 1) Project goal (current behavior)

The repository now supports **one primary experiment family** with one shared comparison reference.

### A. Sequence-model workflow

The neural-model pipeline uses a fixed lookback sequence and predicts a configurable return target:

- Input: the last `SEQ_LEN` timesteps of 8 engineered SPY features, shaped `(N, T, 8)`.
- Target: one of:
  - `horizon_return`: `log_ret` at `t + HORIZON`
  - `next_return`: `log_ret` at `t + 1`
  - `next3_mean_return` (legacy): mean `log_ret` over the next `TARGET_SMOOTH_WINDOW` steps
  - `next_mean_return`: mean `log_ret` over the next `TARGET_SMOOTH_WINDOW` steps (MA-style target)
  - `next_volatility`: std of `log_ret` over the next `TARGET_SMOOTH_WINDOW` steps (volatility target)
  - `sine_next_day`: alias of next-step target for sine next-day prediction runs
- Entrypoints: `src/rnn/train.py`, `src/lstm/train.py`, `src/gru/train.py`, `src/transformer/train.py`, `src/comparison/main.py`, `src/comparison/best_tuned_main.py`, `src/tuning/main.py`.

### B. Shared comparison baseline

The comparison workflows also fit a flattened-sequence linear regression on the exact same prepared dataset used by both comparison entrypoints:

- Input: the last `SEQ_LEN` timesteps of all 8 features, flattened into one tabular row.
- Target: future `log_ret` at `t + HORIZON`.
- Entrypoints: `src/comparison/main.py`, `src/comparison/best_tuned_main.py`.

That flattened-sequence linear-regression baseline is part of both the shared comparison workflow and the best-tuned comparison workflow.

With the current defaults in `src/common/config.py`:

- `SEQ_LEN = 30`
- `HORIZON = 1`
- `TARGET_MODE = "horizon_return"`
- `TARGET_SMOOTH_WINDOW = 3`
- `RANDOM_SEED = 42` (default reproducibility seed for neural runs)
So the default user-facing workflow predicts one-day-ahead returns from a 30-step lookback, while still allowing target difficulty switches via runtime options.

## 2) End-to-end pipeline

Most experiment scripts follow a shared sequence:

1. Download SPY data with `yfinance`.
2. Compute an 8-column SPY feature frame from `Open`, `High`, `Low`, `Close`, and `Volume` returned by `yfinance`.
3. Build sequence tensors for neural models and flattened versions of those same multi-feature sequences for the linear-regression reference.
4. Split data chronologically into train/validation/test.
5. Fit a scaler on training inputs and apply it to validation/test inputs.
6. Train the model or baseline.
7. Evaluate with MAE, MSE, and Directional Accuracy (DA).
8. Save metrics, plots, diagnostics, and structured experiment records under the active report directory (default: `reports/sessions/run_<UTC timestamp>/`).

## 3) Directory walkthrough

### `src/common/` — shared infrastructure

#### `config.py`
Central project defaults:

- market/data scope (`TICKER`, `START`),
- task definition (`SEQ_LEN`, `HORIZON`, `TARGET_MODE`, `TARGET_SMOOTH_WINDOW`, `LAGS`),
- split ratios,
- training controls (`EPOCHS`, `PATIENCE`, `MIN_DELTA`, `MIN_EPOCHS`, `TRAIN_LOG_EVERY`),
- reproducibility defaults (`RANDOM_SEED`),
- scheduler settings,
- reports/figure output paths (with per-run session directories by default),
- `DEVICE` selection (`cuda` if available, otherwise `cpu`).

#### `runtime_config.py`
Defines `RuntimeTrainingConfig`, which allows neural entrypoints to combine:

- defaults from `config.py`,
- explicit config objects,
- dictionaries,
- CLI overrides.

It also defines the shared command-line flags used by the neural training scripts.
This includes `--random-seed` so single-model runs, tuning, and comparison reruns can be reproduced consistently.

#### `data.py`
Core data utilities:

- `load_data(...)`: downloads SPY data and computes close-to-close log returns (legacy utility retained for compatibility).
- `build_spy_feature_frame(...)`: builds the shared 8-feature input frame:
  - `log_ret = log(Close / Close.shift(1))`
  - `oc_ret = log(Close / Open)`
  - `hl_range = (High - Low) / Close`
  - `vol_chg = log(Volume / Volume.shift(1))`
  - `ma_5_gap = (Close / Close.rolling(5).mean()) - 1`
  - `ma_20_gap = (Close / Close.rolling(20).mean()) - 1`
  - `volatility_5 = log_ret.rolling(5).std()`
  - `volatility_20 = log_ret.rolling(20).std()`
- `make_lag_features(...)`: helper for lag-feature tabular inputs retained as a reusable data utility.
- `build_sequences(...)`: builds `(N, seq_len, 8)` sequence inputs with configurable target modes (`horizon_return`, `next_return`, `next3_mean_return`, `next_mean_return`, `next_volatility`, `sine_next_day`) while keeping labels tied to future `log_ret`.
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
- checkpointing best weights to `<reports_dir>/<Model>.pt`,
- saving diagnostics JSON,
- generating plots,
- appending structured experiment records.

Saved neural artifacts include:

- checkpoint: `<reports_dir>/<Model>.pt`
- diagnostics: `<reports_dir>/<model>_diagnostics.json`
- plots in `<reports_dir>/figures/`:
  - `loss_<Model>.png` (pattern example; `src/common/train.py` may append run-context identifiers via `_build_figure_stem`, e.g. `loss_<model>_<run_id>.png`)
  - `<model>_pred_slice.png` (pattern example; actual files may include run-context suffixes, e.g. `<model>_<run_id>_pred_slice.png`)
  - `<model>_scatter.png` (pattern example; actual files may include run-context suffixes, e.g. `<model>_<run_id>_scatter.png`)

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
- shared sequence-data preparation for neural models, including runtime target controls (`horizon`, `target_mode`, `target_smooth_window`).

### `src/rnn/`, `src/lstm/`, `src/gru/`, `src/transformer/`

Each model directory has:

- `model.py`: the architecture definition,
- `train.py`: the runtime-configurable entrypoint that prepares a shared sequence experiment and calls the common trainer.

These entrypoints accept CLI hyperparameter overrides, which is useful for manual sweeps or scripted tuning.

### `src/comparison/`

#### `main.py`
Runs the shared comparison workflow for sequence models.

What it does:

- resolves runtime task controls at the entrypoint (`task_id`, `data_source`, `target_mode`, `target_smooth_window`, `horizon`),
- builds one common task-aware sequence split via `prepare_sequence_experiment_run(...)`,
- applies deterministic train-loader shuffling and training seeds from `runtime_config.random_seed` (default 42),
- fits a flattened-sequence linear regression reference (`Baseline-LR`),
- trains RNN, LSTM, GRU, and Transformer on the same split,
- writes task-scoped comparison metrics and a task-scoped comparison-plot figure:
  - `metrics_comparison_<task_id>.csv`
  - `metrics_comparison_<task_id>.json`
  - `comparison_losses_<task_id>.png`
- writes a model-comparison record selecting the best model by validation MSE, including task metadata in the summary payload.

This file is the best source for the repository's **apples-to-apples sequence comparison**.

#### `best_configs.py`
Loads a tuning artifact and normalizes the best per-model hyperparameters into the runtime aliases accepted by the shared neural entrypoints.

By default it treats `<reports_dir>/tuning_winners.csv` as the canonical source of "best parameters" because that file captures the final frozen configuration produced by sequential staged tuning.

`<reports_dir>/tuning_best_configs.csv` is also supported when you want the single best archived run per model instead.

#### `best_tuned_main.py`
Runs a tuned-model comparison using the tuned-best settings recovered from the tuning artifacts plus the shared linear-regression baseline.

What it does:

- selects a tuning source (`tuning_winners.csv` by default),
- rebuilds prepared sequence runs using the shared experiment helper,
- applies deterministic seeding for those reruns (`random_seed`, default 42),
- computes the flattened-sequence linear-regression baseline on the same shared split,
- reuses each model's existing training entrypoint,
- includes train / validation / test MSE summary columns in the final report,
- writes `<reports_dir>/best_tuned_comparison.csv` and `<reports_dir>/best_tuned_comparison.md`,
- summarizes the best model by validation and test MSE.

#### `best_tuned_charts.py`
Generates standalone SVG summary charts from the best-tuned comparison CSV.

What it does:

- reads `<reports_dir>/best_tuned_comparison.csv`,
- validates that the `model`, `best_train_MSE`, `best_test_MSE`, and `best_val_MSE` columns are present,
- renders one SVG bar chart per metric with all best-tuned models shown together,
- writes the charts to `<reports_dir>/figures/` by default,
- exposes a CLI entrypoint through `python -m src.comparison.best_tuned_charts`.

### `src/tuning/`

#### `main.py`
Runs staged hyperparameter sweeps.

Default stage order:

- recurrent models: `lr -> hidden -> layers -> batch_size`
- transformer: `lr -> d_model -> num_layers -> nhead -> batch_size`

Capabilities include:

- tune one model or all models,
- provide overrides from a JSON file or inline JSON,
- pass task-scoping runtime settings (`task_id`, `target_mode`, `target_smooth_window`, `horizon`, `data_source`, etc.) so tuning rows are tied to one forecast task,
- pass/override reproducibility controls (`random_seed`, default 42),
- reset or append tuning outputs,
- dry-run the resolved plan,
- write per-stage run logs and winner summaries.

### `scripts/run_multitask_final_reports.sh`
One-click orchestration script for final-report packaging across the four core tasks:

1. `sine_next_day`
2. `next_return`
3. `next_volatility` with a 5-step forward window
4. `next_mean_return` with a 5-step forward window

For each task it:

- tunes all four neural models (`src.tuning.main`),
- runs tuned-best comparison (`src.comparison.best_tuned_main`),
- builds hyperparameter impact report (`src.tuning.report`),
- exports tuned-comparison charts (`src.comparison.best_tuned_charts`),
- writes outputs into a dedicated task folder under one shared root bundle for easier final-report use.
- sets `MPLBACKEND=Agg` (if unset) to keep Matplotlib non-interactive and avoid Tk/Tcl crashes on Windows/headless shells.



## 4) Important outputs in `<reports_dir>`

By default, `<reports_dir>` means `reports/sessions/run_<UTC timestamp>/`. You can force legacy direct output to `reports/` by setting `FYP_REPORTS_DISABLE_SESSION_DIR=1`, or set `FYP_REPORTS_DIR` to a custom path.

Depending on which workflows you run, you may see files such as:

- checkpoints: `RNN.pt`, `LSTM.pt`, `GRU.pt`, `Transformer.pt`
- per-model metrics: `metrics_gru.json`, `metrics_lstm.json`, etc.
- summary metrics:
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
  - `loss_<Model>.png` (pattern example; `src/common/train.py` may append run-context identifiers via `_build_figure_stem`, e.g. `loss_<model>_<run_id>.png`)
  - `<model>_pred_slice.png` (pattern example; actual files may include run-context suffixes, e.g. `<model>_<run_id>_pred_slice.png`)
  - `<model>_scatter.png` (pattern example; actual files may include run-context suffixes, e.g. `<model>_<run_id>_scatter.png`)
  - `comparison_losses.png`
  - `best_tuned_training_loss.svg`
  - `best_tuned_testing_loss.svg`
  - `best_tuned_validation_loss.svg`

## 5) How to run

From the repository root:

```bash
pip install -r requirements.txt
python -m src.gru.train --learning-rate 5e-4 --recurrent-hidden-size 128
python -m src.comparison.main
python -m src.tuning.main --model all --session-mode append
python -m src.comparison.best_tuned_main
python -m src.comparison.best_tuned_charts
bash scripts/run_multitask_final_reports.sh
```

On Windows, run the same command from Git Bash.

## 6) What to tweak first

Start with `src/common/config.py` if you want to change global defaults:

- `HORIZON`: how far ahead sequence models predict in `horizon_return` mode.
- `TARGET_MODE`: sequence target definition (`horizon_return`, `next_return`, `next3_mean_return`, `next_mean_return`, `next_volatility`, `sine_next_day`).
- `TARGET_SMOOTH_WINDOW`: forward window for rolling target modes (`next3_mean_return`, `next_mean_return`, `next_volatility`).
- `SEQ_LEN`: neural-model lookback window.
- `LAGS`: retained lag-window helper setting for tabular-feature utilities.
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
4. **User-facing entrypoints** — `src/*/train.py`, `src/comparison/main.py`, `src/tuning/main.py`

Once you keep those layers separate, the refactored codebase is much easier to navigate.
