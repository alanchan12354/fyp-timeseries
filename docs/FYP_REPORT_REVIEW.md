# FYP Report Review: Current Readiness After the Refactor

## Verdict

The repository is now in a **much stronger position** for an FYP report than before, because it includes:

- reproducible entrypoints,
- structured experiment logging,
- tuning summary outputs,
- diagnostics and plots,
- environment metadata capture,
- comparison-summary records.

However, it is **still not automatically a complete report**. The remaining work is mostly about **using the generated evidence carefully and writing the methodology consistently**.

## What is already strong enough now

### 1. Structured experiment logs exist

The project now writes machine-readable experiment records to:

- `reports/experiment_log.jsonl`
- `reports/experiment_log.csv`

These records include more than final metrics. They can include:

- timestamps,
- run IDs,
- task metadata,
- split sizes,
- training metadata,
- environment/package metadata,
- hyperparameters,
- artifact paths,
- tuning notes.

That is a clear improvement over a repo that only saves one metrics CSV.

### 2. Reproducibility support is materially better

The reporting utilities now capture key reproducibility details such as:

- Python version,
- platform,
- device,
- git commit,
- major package versions.

For an FYP, this is already a reasonable reproducibility baseline.

### 3. Fine-tuning records now exist in dedicated outputs

The tuning workflow produces summary CSV files such as:

- `reports/tuning_runs.csv`
- `reports/tuning_winners.csv`
- `reports/tuning_all_runs.csv`
- `reports/tuning_best_configs.csv`

That means the repository now supports a proper **hyperparameter tuning evidence trail**, not just ad hoc code comments.

### 4. The comparison workflow is better defined

`src/comparison/main.py` now provides a clear shared-split comparison for:

- flattened-sequence linear regression,
- RNN,
- LSTM,
- GRU,
- Transformer.

This is the most defensible comparison pipeline in the repo because it uses one aligned sequence dataset for all compared models in that script.

### 5. Diagnostics and figures are report-friendly

The common trainer exports:

- checkpoint files,
- per-epoch diagnostics JSON,
- loss curves,
- prediction slices,
- scatter plots.

These are useful raw materials for methodology and results chapters.

## What still needs care in the final report

### 1. The repository still contains two task definitions

This is the most important issue to explain clearly.

- `src/baselines/main.py` is a **next-day lag-feature** workflow.
- The neural-training and comparison workflow is, by default, a **30-step input / 10-step-ahead target** workflow.

So the report must not casually claim that every result table compares the exact same target unless that is actually true for the specific table.

### 2. One analysis document is still a historical snapshot, not a guaranteed live truth

`docs/analysis-0224.md` should be treated as a **snapshot analysis note** tied to a particular recorded result set, not as a permanent statement about the latest repository defaults.

That is acceptable, but the report should reference exact experiment artifacts and dates when citing results.

### 3. Single-run conclusions may still be weak academically

Even with good logging, a final academic comparison is stronger if you report either:

- repeated runs with mean ± standard deviation, or
- a clear limitation statement explaining that results come from single runs.

The current code helps record runs, but it does not automatically enforce repeated-evaluation reporting.

### 4. Data-source drift should still be acknowledged

The repository downloads market data from Yahoo Finance at run time.
That means the exact dataset can change as new trading days are added or historical adjustments are revised.

For a final dissertation/report submission, it is safer to state:

- when data was downloaded,
- the exact date range used in the cited experiment,
- which output files correspond to the final frozen result set.

## Recommended report framing

## 1. Experimental Setup

State explicitly:

- data source: Yahoo Finance via `yfinance`,
- ticker: SPY,
- start date used,
- return definition: daily log return,
- split strategy: chronological train/validation/test,
- scaling strategy: fit scaler on train inputs only,
- exact target definition for each table.

## 2. Comparison Section

Use `src/comparison/main.py` artifacts when you want the cleanest shared-target model comparison.
If you also include the standalone baseline script, label it separately as a **next-day lag baseline benchmark** unless you reconfigure the project so the targets align.

## 3. Hyperparameter Tuning Section

Base this section on the tuning outputs rather than on prose alone.
A solid subsection should include:

- search stages per model,
- candidate values tried,
- validation metric used for selection,
- final winning configuration,
- a short interpretation of what changed performance.

## 4. Reproducibility Section

Use the logged metadata to state:

- software environment,
- device used,
- commit hash,
- package versions,
- run IDs for the experiments discussed in the report.

## Suggested final improvements before submission

1. **Freeze a final result set** by running the final experiments once, then archive the exact output files used in the report.
2. **Reference run IDs** in the report when discussing tuning winners and final comparisons.
3. **Add repeated runs** if time permits, especially for neural models.
4. **Keep terminology consistent** everywhere: next-day baseline vs. horizon-ahead sequence forecast.
5. **Promote one canonical results table** for the main comparison section, ideally from `src/comparison/main.py`.

## Bottom line

**Short answer:** the repository is now **close to report-ready as an experimentation system**, because the logging and tuning infrastructure are meaningfully improved.

The remaining risk is no longer “missing documentation structure”; it is mainly **methodology consistency and careful interpretation of which experiment output supports which claim**.
