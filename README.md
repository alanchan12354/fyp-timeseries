# Time Series Neural Models Project

This repository benchmarks neural approaches for forecasting **SPY daily log returns**, with a linear-regression reference included in the shared comparison workflow.
The codebase now includes shared experiment-preparation utilities, runtime-configurable training entrypoints, structured experiment logging, and a sequential tuning workflow.

## What the project does

The repository currently supports a shared forecasting setup across the neural training and comparison workflows:

- **Sequence-model workflow** (`src/*/train.py`, `src/comparison/main.py`, `src/tuning/main.py`): predicts a configurable return target from the last `SEQ_LEN` windows of an 8-feature SPY input schema.
- **Comparison reference baseline** (`src/comparison/main.py`): fits a flattened-sequence linear regression on that same shared `SEQ_LEN` / `HORIZON` dataset.

With the default configuration in `src/common/config.py`:

- `TICKER = "SPY"`
- `START = "2010-01-01"`
- `SEQ_LEN = 30`
- `HORIZON = 1`
- `TARGET_MODE = "horizon_return"`
- `TARGET_SMOOTH_WINDOW = 3`

So, by default, experiments forecast the **next trading day's** return from a **30-step multivariate lookback window**.

The default per-step input feature schema is:

- `log_ret = log(Close / Close.shift(1))`
- `oc_ret = log(Close / Open)`
- `hl_range = (High - Low) / Close`
- `vol_chg = log(Volume / Volume.shift(1))`
- `ma_5_gap = (Close / Close.rolling(5).mean()) - 1`
- `ma_20_gap = (Close / Close.rolling(20).mean()) - 1`
- `volatility_5 = log_ret.rolling(5).std()`
- `volatility_20 = log_ret.rolling(20).std()`

Targets remain based on future **`log_ret`** values (`horizon_return`, `next_return`, or `next3_mean_return`), so feature engineering and labels stay clearly separated.

Available neural target modes are:
- `horizon_return`: `r_{t+horizon}`
- `next_return`: `r_{t+1}`
- `next3_mean_return`: mean of the next `target_smooth_window` returns

## Repository structure

```text
src/
  common/          Shared config, data prep, training, reporting, runtime config
  comparison/      Shared-split comparison pipeline across models
  gru/             GRU model and entrypoint
  lstm/            LSTM model and entrypoint
  rnn/             Vanilla RNN model and entrypoint
  transformer/     Transformer encoder model and entrypoint
  tuning/          Sequential hyperparameter tuning workflow

docs/
  CODEBASE_TOUR.md         Beginner-friendly architecture and workflow guide
  FYP_REPORT_REVIEW.md     FYP-report readiness review based on current code
  analysis-0224.md         Snapshot analysis write-up of one reported result set

reports/
  sessions/
    run_<UTC timestamp>/
      Generated metrics, checkpoints, diagnostics, figures, and tuning summaries

tests/
  Regression tests for reporting/tuning utilities
```

## Setup

1. Create or activate a Python environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run commands from the repository root.

## Main workflows

> **Report output location**
>
> Most workflows write outputs to a per-run session folder:
> `reports/sessions/run_<UTC timestamp>/...`
>
> You can override this behavior with:
> - `FYP_REPORTS_DIR=/your/path` (force exact output directory)
> - `FYP_REPORTS_DISABLE_SESSION_DIR=1` (write directly to `reports/` like legacy behavior)

### 1. Train a single neural model

```bash
python -m src.lstm.train
python -m src.gru.train
python -m src.rnn.train
python -m src.transformer.train
```

Each training entrypoint uses the shared sequence pipeline and writes:

- `<reports_dir>/<Model>.pt`
- `<reports_dir>/<model>_diagnostics.json`
- `<reports_dir>/metrics_<model>.json`
- plots in `<reports_dir>/figures/`
- structured experiment-log rows

### 2. Override runtime hyperparameters from the CLI

The neural entrypoints accept runtime configuration flags without editing source files.
For example:

```bash
python -m src.gru.train \
  --learning-rate 5e-4 \
  --batch-size 32 \
  --recurrent-hidden-size 128 \
  --recurrent-layer-count 3 \
  --target-mode next_return \
  --run-note "manual_gru_sweep"
```

Transformer entrypoints also support:

```bash
python -m src.transformer.train \
  --learning-rate 5e-4 \
  --batch-size 32 \
  --d-model 128 \
  --transformer-num-layers 3 \
  --nhead 8 \
  --target-mode next3_mean_return \
  --target-smooth-window 3 \
  --run-note "manual_transformer_sweep"
```

Target-focused runtime flags available on all neural entrypoints:
- `--horizon`
- `--target-mode {horizon_return,next_return,next3_mean_return}`
- `--target-smooth-window`

### 3. Run the shared model-comparison pipeline

```bash
python -m src.comparison.main
```

This workflow:

- builds one shared sequence split from the 8-feature SPY frame,
- trains RNN, LSTM, GRU, and Transformer on that split,
- computes a flattened-sequence linear-regression reference across all sequence features,
- saves comparison metrics and a loss-comparison figure,
- writes a model-comparison record summarizing the winner by validation MSE.

### 4. Run the tuning workflow

```bash
python -m src.tuning.main --model all --session-mode append
```

Useful variants:

```bash
python -m src.tuning.main --model gru --session-mode reset
python -m src.tuning.main --model transformer --dry-run
python -m src.tuning.main --plan-file path/to/plan.json
python -m src.tuning.main --plan-json '{"gru": {"hidden": [32, 64]}}'
```

The tuning runner performs staged sweeps over model-specific parameter groups and writes summary CSVs such as:

- `<reports_dir>/tuning_runs.csv`
- `<reports_dir>/tuning_winners.csv`
- `<reports_dir>/tuning_all_runs.csv`
- `<reports_dir>/tuning_best_configs.csv`

`<reports_dir>/tuning_winners.csv` is the canonical source of "best parameters" for downstream comparison because it stores the final frozen winner after each sequential tuning stage. `<reports_dir>/tuning_best_configs.csv` remains available when you want the single best archived run per model instead.

If `--session-mode reset` is used, the runner clears prior tuning artifacts in the active `<reports_dir>` before starting a fresh session.

### 5. Compare models with their tuned-best configurations

```bash
python -m src.comparison.best_tuned_main
```

Optional variant if you want to compare the single best archived run per model instead of the final staged winners:

```bash
python -m src.comparison.best_tuned_main --config-source tuning_best_configs
```

This workflow:

- loads tuned per-model hyperparameters from `<reports_dir>/tuning_winners.csv` by default,
- computes the same flattened-sequence linear-regression baseline on the shared split,
- reuses the shared sequence experiment-preparation flow,
- calls the existing model training entrypoints with the tuned settings,
- reports train / validation / test MSE summary columns for every row,
- writes `<reports_dir>/best_tuned_comparison.csv` and `<reports_dir>/best_tuned_comparison.md`.

### 6. Generate SVG charts for the best-tuned comparison

```bash
python -m src.comparison.best_tuned_charts
```

Optional variant if you want to point at a different comparison CSV or output folder:

```bash
python -m src.comparison.best_tuned_charts \
  --csv-path <reports_dir>/best_tuned_comparison.csv \
  --output-dir <reports_dir>/figures
```

This workflow:

- reads `<reports_dir>/best_tuned_comparison.csv`,
- validates that the best-tuned comparison includes model, training, testing, and validation MSE columns,
- generates one SVG bar chart per metric with all best-tuned models shown on the same graph,
- writes:
  - `<reports_dir>/figures/best_tuned_training_loss.svg`
  - `<reports_dir>/figures/best_tuned_testing_loss.svg`
  - `<reports_dir>/figures/best_tuned_validation_loss.svg`

### 7. Generate the hyper-parameter impact report

```bash
python -m src.tuning.report
```

This report reads the existing tuning artifacts and generates:

- `<reports_dir>/hyperparameter_impact_report.md`
- `<reports_dir>/figures/hyperparameter_model_loss_summary.svg`

The figure places the tuned models' training, testing, and validation losses in three side-by-side subplots that share the same y-axis. This report is separate from `src.comparison.best_tuned_charts`, which writes individual SVG charts from `<reports_dir>/best_tuned_comparison.csv`.

## Experiment logging and reproducibility

The reporting utilities now capture more than just final metrics.
Depending on the workflow, the repository records:

- run IDs and UTC timestamps,
- task metadata (ticker, start date, input window, horizon, target mode, target smoothing window, split ratios),
- split sizes,
- training metadata,
- environment metadata such as Python, platform, device, package versions, and git commit,
- hyperparameters,
- final metrics and tuning annotations,
- artifact paths.

Key generated files include:

- `<reports_dir>/experiment_log.jsonl`: full structured run records
- `<reports_dir>/experiment_log.csv`: flattened summary view
- `<reports_dir>/model_comparison_record.json`
- `<reports_dir>/model_comparison_record.csv`
- `<reports_dir>/best_tuned_comparison.csv`
- `<reports_dir>/best_tuned_comparison.md`
- `<reports_dir>/tuning_all_runs.csv`
- `<reports_dir>/tuning_best_configs.csv`

## Comparison framing

Use `src/comparison/main.py` as the canonical apples-to-apples benchmark workflow.
It compares the neural models against a **flattened-sequence linear regression** built on the shared sequence framing, so the reported targets stay aligned when you discuss results.

## New to the codebase?

Start with `docs/CODEBASE_TOUR.md` for the architecture walkthrough, then read `docs/FYP_REPORT_REVIEW.md` for guidance on how the current outputs map to an FYP report.
