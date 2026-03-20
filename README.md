# Time Series Neural Models Project

This repository benchmarks baseline and neural approaches for forecasting **SPY daily log returns**.
The codebase now includes shared experiment-preparation utilities, runtime-configurable training entrypoints, structured experiment logging, and a sequential tuning workflow.

## What the project does

The repository currently supports a shared forecasting setup across baseline and neural workflows:

- **Baseline workflow** (`src/baselines/main.py`): predicts the log return at `t + HORIZON` from the last `SEQ_LEN` returns using persistence and linear regression baselines.
- **Sequence-model workflow** (`src/*/train.py`, `src/comparison/main.py`, `src/tuning/main.py`): predicts the log return at `t + HORIZON` from the last `SEQ_LEN` returns.

With the default configuration in `src/common/config.py`:

- `TICKER = "SPY"`
- `START = "2010-01-01"`
- `SEQ_LEN = 30`
- `HORIZON = 10`
- `LAGS = 30`

That means both the baseline and neural experiments are, by default, forecasting **10 trading days ahead** from a **30-return lookback window**.

## Repository structure

```text
src/
  baselines/       Benchmark baselines (Persistence, Linear Regression)
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

### 1. Run baseline benchmarks

```bash
python -m src.baselines.main
```

Outputs include:

- `reports/metrics_baselines.csv`
- `reports/metrics_baselines.json`
- experiment-log rows in `reports/experiment_log.jsonl` and `reports/experiment_log.csv`

### 2. Train a single neural model

```bash
python -m src.lstm.train
python -m src.gru.train
python -m src.rnn.train
python -m src.transformer.train
```

Each training entrypoint uses the shared sequence pipeline and writes:

- `reports/<Model>.pt`
- `reports/<model>_diagnostics.json`
- `reports/metrics_<model>.json`
- plots in `reports/figures/`
- structured experiment-log rows

### 3. Override runtime hyperparameters from the CLI

The neural entrypoints accept runtime configuration flags without editing source files.
For example:

```bash
python -m src.gru.train \
  --learning-rate 5e-4 \
  --batch-size 32 \
  --recurrent-hidden-size 128 \
  --recurrent-layer-count 3 \
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
  --run-note "manual_transformer_sweep"
```

### 4. Run the shared model-comparison pipeline

```bash
python -m src.comparison.main
```

This workflow:

- builds one shared sequence split,
- trains RNN, LSTM, GRU, and Transformer on that split,
- computes a flattened-sequence linear-regression reference,
- saves comparison metrics and a loss-comparison figure,
- writes a model-comparison record summarizing the winner by validation MSE.

### 5. Run the tuning workflow

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

- `reports/tuning_runs.csv`
- `reports/tuning_winners.csv`
- `reports/tuning_all_runs.csv`
- `reports/tuning_best_configs.csv`

`reports/tuning_winners.csv` is the canonical source of "best parameters" for downstream comparison because it stores the final frozen winner after each sequential tuning stage. `reports/tuning_best_configs.csv` remains available when you want the single best archived run per model instead.

If `--session-mode reset` is used, the runner clears prior tuning artifacts in `reports/` before starting a fresh session.

### 6. Compare models with their tuned-best configurations

```bash
python -m src.comparison.best_tuned_main
```

Optional variant if you want to compare the single best archived run per model instead of the final staged winners:

```bash
python -m src.comparison.best_tuned_main --config-source tuning_best_configs
```

This workflow:

- loads tuned per-model hyperparameters from `reports/tuning_winners.csv` by default,
- reuses the shared sequence experiment-preparation flow,
- calls the existing model training entrypoints with the tuned settings,
- writes `reports/best_tuned_comparison.csv` and `reports/best_tuned_comparison.md`.

### 7. Generate the hyper-parameter impact report

```bash
python -m src.tuning.report
```

This report reads the existing tuning artifacts and generates:

- `reports/hyperparameter_impact_report.md`
- `reports/figures/hyperparameter_model_loss_summary.svg`

The figure places the tuned models' training, testing, and validation losses in three side-by-side subplots that share the same y-axis.

## Experiment logging and reproducibility

The reporting utilities now capture more than just final metrics.
Depending on the workflow, the repository records:

- run IDs and UTC timestamps,
- task metadata (ticker, start date, input window, horizon, split ratios),
- split sizes,
- training metadata,
- environment metadata such as Python, platform, device, package versions, and git commit,
- hyperparameters,
- final metrics and tuning annotations,
- artifact paths.

Key generated files include:

- `reports/experiment_log.jsonl`: full structured run records
- `reports/experiment_log.csv`: flattened summary view
- `reports/model_comparison_record.json`
- `reports/model_comparison_record.csv`
- `reports/best_tuned_comparison.csv`
- `reports/best_tuned_comparison.md`
- `reports/tuning_all_runs.csv`
- `reports/tuning_best_configs.csv`

## Important caveat about comparisons

There is an intentional documentation caveat you should keep in mind when writing results:

- `src/baselines/main.py` now uses the shared `SEQ_LEN` / `HORIZON` task definition for persistence and linear regression.
- `src/comparison/main.py` compares neural models against a **flattened-sequence linear regression** built on that same `SEQ_LEN` / `HORIZON` dataset.

So the baseline script and the comparison workflow now refer to the same target definition when you report results.

## New to the codebase?

Start with `docs/CODEBASE_TOUR.md` for the architecture walkthrough, then read `docs/FYP_REPORT_REVIEW.md` for guidance on how the current outputs map to an FYP report.
