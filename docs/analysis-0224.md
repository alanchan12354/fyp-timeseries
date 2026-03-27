# Model Analysis Report

**Snapshot date:** February 24, 2026
**Document role:** historical analysis note tied to one recorded metrics snapshot, not a guarantee of current default outputs.

## 1. Scope of this note

This document summarizes a specific reported result set for the SPY forecasting project.
It should be read as a **point-in-time interpretation** of exported metrics, not as a source of truth for the repository's latest configuration.

The codebase has since evolved to support:

- runtime-configurable neural training,
- structured experiment logging,
- model-comparison records,
- staged tuning workflows,
- richer reproducibility metadata.

## 2. Metrics source used for this snapshot

This write-up is based on the following generated artifacts from that run period:

- `reports/metrics_comparison.csv`
- `reports/metrics_baselines.csv`

Because those files are generated outputs rather than version-controlled source files, you should verify the exact archived copies used in your report before citing any numeric values.

## 3. Important interpretation warning

Keep this document as a **historical snapshot** tied to one prior analysis window.

The repository contains **two forecasting setups**:

- the standalone baseline script uses **next-day** targets,
- the sequence-model workflow uses a configurable `HORIZON` target; this snapshot may have used horizon 10, but current defaults should be checked in `src/common/config.py`.

Current code defaults may differ from this snapshot, and any conclusions should cite the exact run IDs and generated artifacts used.

If the metrics underlying this historical note came from mixed workflows, then the comparison must be described carefully in any formal report.

## 4. Snapshot results

### Neural models (test-set metrics)

| Model | MAE | MSE | Directional Accuracy (DA) | Best Val MSE |
| :--- | :--- | :--- | :--- | :--- |
| **RNN** | 0.00680 | 1.02e-4 | 47.37% | 1.40e-4 |
| **LSTM** | 0.00662 | **9.78e-5** | 46.55% | **1.34e-4** |
| **GRU** | **0.00657** | 1.01e-4 | **54.28%** | 1.38e-4 |
| **Transformer** | 0.00900 | 1.62e-4 | 46.71% | 1.96e-4 |

### Baselines (test-set metrics)

| Model | MAE | MSE | Directional Accuracy (DA) |
| :--- | :--- | :--- | :--- |
| **Persistence** | 0.00952 | 2.08e-4 | 52.63% |
| **Linear Regression** | 0.00665 | 9.92e-5 | 52.14% |

## 5. Snapshot interpretation

### Validation performance

- **LSTM** achieved the lowest reported validation MSE in this snapshot.
- **Transformer** underperformed relative to the recurrent models on this particular exported result set.

### Test-set error magnitude

- **LSTM** and **Linear Regression** were the strongest by MSE in this snapshot.
- The strong linear-regression result suggests the signal may be weak enough that simple models remain highly competitive.

### Directional accuracy

- **GRU** had the highest reported directional accuracy in this snapshot.
- Several models appear to have achieved relatively low magnitude error without matching that with strong sign prediction.

## 6. How to use this document safely

Use this file as:

- a narrative example of how to discuss a result set,
- a historical note tied to a dated output snapshot.

Do **not** use it as the authoritative definition of the current repository defaults.
For that, consult:

- `src/common/config.py` for default task/training settings,
- `src/common/reporting.py` for current logging outputs,
- `src/comparison/main.py` for the current shared comparison workflow,
- `src/tuning/main.py` for the current staged tuning process.

## 7. Recommended next step

Before incorporating these numbers into a final report, rerun or verify the exact archived experiment artifacts and cite the corresponding run IDs, timestamps, and generated output files.
