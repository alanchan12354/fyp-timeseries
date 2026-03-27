# Time Series Neural Models Project

This repository benchmarks neural approaches across **four equally important forecasting tasks**: `sine_next_day`, `next_return`, `next_volatility`, and `next_mean_return`, with a linear-regression reference included in the shared comparison workflow.
The codebase includes shared experiment-preparation utilities, runtime-configurable training entrypoints, structured experiment logging, and sequential tuning workflows per task.

## What the project does

The repository currently supports a shared forecasting setup across the neural training and comparison workflows:

- **Sequence-model workflow** (`src/*/train.py`, `src/comparison/main.py`, `src/tuning/main.py`): predicts a configurable return target from the last `SEQ_LEN` windows of an 8-feature SPY input schema.
- **Comparison reference baseline** (`src/comparison/main.py`): fits a flattened-sequence linear regression on that same shared `SEQ_LEN` / `HORIZON` dataset.

With the default configuration in `src/common/config.py`:

- `TICKER = "SPY"`
- `START = "2005-01-01"`
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

Targets remain based on future **`log_ret`** values, so feature engineering and labels stay clearly separated.

Available neural target modes are:
- `horizon_return`: `r_{t+horizon}`
- `next_return`: `r_{t+1}`
- `next3_mean_return`: legacy alias for mean of the next `target_smooth_window` returns
- `next_mean_return`: mean of the next `target_smooth_window` returns (for MA(5), MA(10), etc.)
- `next_volatility`: rolling std of the next `target_smooth_window` returns (for volatility tasks)
- `sine_next_day`: alias of next-step return used for sine next-day prediction workflows

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
- plots in `<reports_dir>/figures/`, including pattern examples:
  - `loss_<Model>.png` (pattern example; `src/common/train.py` may append run-context identifiers via `_build_figure_stem`, e.g. `loss_<model>_<run_id>.png`)
  - `<model>_pred_slice.png` (pattern example; actual files may include run-context suffixes, e.g. `<model>_<run_id>_pred_slice.png`)
  - `<model>_scatter.png` (pattern example; actual files may include run-context suffixes, e.g. `<model>_<run_id>_scatter.png`)
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
  --target-mode next_mean_return \
  --target-smooth-window 3 \
  --run-note "manual_transformer_sweep"
```

Target-focused runtime flags available on all neural entrypoints:
- `--horizon`
- `--target-mode {horizon_return,next_return,next3_mean_return,next_mean_return,next_volatility,sine_next_day}`
- `--target-smooth-window`

### Predefined sanity profile (`--run-note sanity_sine`)

For a quick, reproducible sanity pass before longer sweeps, use the predefined
`sanity_sine` runtime profile. When `--run-note sanity_sine` is set, runtime config
automatically applies these stable settings:

- `target_mode=next_return`
- `horizon=1`
- moderate model width (`recurrent_hidden_size=64`, `transformer_d_model=64`)
- enough training budget (`epochs=80`)
- scheduler disabled (`scheduler_type=none`)

Example (GRU):

```bash
python -m src.gru.train \
  --run-note sanity_sine \
  --horizon 1 \
  --target-mode next_return \
  --recurrent-hidden-size 64 \
  --epochs 80 \
  --scheduler-type none
```

The same profile trigger works for LSTM/RNN/Transformer entrypoints.

Expected sanity outcomes for this profile:

- **Low test MSE** relative to the normalized wave baseline (the training code currently
  uses an automatic pass/fail threshold of `MSE <= 0.02` for `sanity_sine` runs).
- **True-vs-predicted scatter** should look close to a diagonal trend.
- **Prediction-slice chart** should overlap the true wave shape with limited phase drift.
- The generated `metrics_*.json` includes `sanity_check` and `sanity_check_passed` so you
  can report pass/fail quickly without manual plot inspection.

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
python -m src.tuning.main --model all --task-id next_volatility --target-mode next_volatility --target-smooth-window 5 --horizon 1
```

The tuning runner performs staged sweeps over model-specific parameter groups and writes summary CSVs such as:

- `<reports_dir>/tuning_runs.csv`
- `<reports_dir>/tuning_winners.csv`
- `<reports_dir>/tuning_all_runs.csv`
- `<reports_dir>/tuning_best_configs.csv`

`<reports_dir>/tuning_winners.csv` is the canonical source of "best parameters" for downstream comparison because it stores the final frozen winner after each sequential tuning stage. `<reports_dir>/tuning_best_configs.csv` remains available when you want the single best archived run per model instead.

If `--session-mode reset` is used, the runner clears prior tuning artifacts in the active `<reports_dir>` before starting a fresh session.

Task-scoping flags are available directly on the tuning runner:

- `--task-id`
- `--horizon`
- `--data-source {spy,sine}`
- `--target-mode {horizon_return,next_return,next3_mean_return,next_mean_return,next_volatility,sine_next_day}`
- `--target-smooth-window`
- `--epochs`
- `--scheduler-type {none,plateau,cosine}`

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


## Multi-task experiments

Use **task** to mean one forecast configuration identified by a stable `task_id` and defined by `target_mode` + `horizon` (plus optional smoothing via `target_smooth_window`).

### Canonical four-task presets

1. **Sine curve next-day prediction**

```bash
python -m src.gru.train \
  --data-source sine \
  --task-id sine_next_day \
  --target-mode sine_next_day \
  --horizon 1
```

2. **SPY next-day return prediction**

```bash
python -m src.gru.train \
  --task-id next_return \
  --target-mode next_return \
  --horizon 1
```

3. **SPY volatility prediction (next 5-day rolling std)**

```bash
python -m src.gru.train \
  --task-id next_volatility \
  --target-mode next_volatility \
  --target-smooth-window 5 \
  --horizon 1
```

4. **SPY mean-return prediction (next 5-day rolling mean)**

```bash
python -m src.gru.train \
  --task-id next_mean_return \
  --target-mode next_mean_return \
  --target-smooth-window 5 \
  --horizon 1
```

### 1) Run tuning/comparison per core task

Current tuning output (`tuning_winners.csv`) is keyed by `task_id`. Run tuning separately for each core task so each tuned winner row is tied to its intended `task_id`:

```bash
python -m src.tuning.main --model all --session-mode reset --task-id sine_next_day --data-source sine --target-mode sine_next_day --horizon 1 --target-smooth-window 1
python -m src.tuning.main --model all --session-mode reset --task-id next_return --data-source spy --target-mode next_return --horizon 1 --target-smooth-window 1
python -m src.tuning.main --model all --session-mode reset --task-id next_volatility --data-source spy --target-mode next_volatility --horizon 1 --target-smooth-window 5
python -m src.tuning.main --model all --session-mode reset --task-id next_mean_return --data-source spy --target-mode next_mean_return --horizon 1 --target-smooth-window 5
```

Then run task-scoped tuned-best comparisons:

```bash
python -m src.comparison.best_tuned_main \
  --task-ids sine_next_day next_return next_volatility next_mean_return
```

This writes per-task files such as:

- `<reports_dir>/best_tuned_comparison_sine_next_day.csv`
- `<reports_dir>/best_tuned_comparison_next_return.csv`
- `<reports_dir>/best_tuned_comparison_next_volatility.csv`
- `<reports_dir>/best_tuned_comparison_next_mean_return.csv`
- `<reports_dir>/multi_task_summary.md`

### 2) Generate per-task and aggregate reports

Per-task hyperparameter impact reports:

```bash
python -m src.tuning.report --task-ids sine_next_day next_return next_volatility next_mean_return
```

Outputs include:

- `<reports_dir>/hyperparameter_impact_report_sine_next_day.md`
- `<reports_dir>/hyperparameter_impact_report_next_return.md`
- `<reports_dir>/hyperparameter_impact_report_next_volatility.md`
- `<reports_dir>/hyperparameter_impact_report_next_mean_return.md`
- `<reports_dir>/multi_task_summary.md` (task-level report index generated by the tuning report workflow)

> **Note**
>
> `src.comparison.best_tuned_main` and `src.tuning.report` both currently write
> `<reports_dir>/multi_task_summary.md`. If you run both workflows in the same
> report directory, the later workflow will overwrite that summary file.

For one consolidated cross-task view in your write-up, combine key rows from the per-task `best_tuned_comparison_<task_id>.csv` files into a single results table in `docs/final_report.md`.

### One-click final-report pipeline (all 4 core tasks)

Use the helper script below to run all four tasks end-to-end (tune all models, compare tuned best models, generate impact reports, generate charts, and produce a cross-task summary) in one command:

```bash
bash scripts/run_multitask_final_reports.sh
```

Optional: provide an explicit output root folder:

```bash
bash scripts/run_multitask_final_reports.sh reports/final_report_tasks/my_final_bundle
```

The script creates a clear task-by-task structure:

- `<root>/sine_next_day/`
- `<root>/next_return/`
- `<root>/next_volatility/`
- `<root>/next_mean_return/`
- `<root>/overall_task_summary.csv` and `<root>/overall_task_summary.md` (cross-task winner summary)
- `<root>/README.md` (index of generated artifacts and rerun command)

## Experiment logging and reproducibility

The reporting utilities now capture more than just final metrics.
Depending on the workflow, the repository records:

- run IDs and UTC timestamps,
- task metadata (`task_id`, ticker, start date, input window, `horizon`, `target_mode`, `target_smooth_window`, split ratios),
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
