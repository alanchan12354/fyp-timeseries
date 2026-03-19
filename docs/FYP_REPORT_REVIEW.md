# FYP Report Review: Model Comparison and Fine-Tuning Record

## Verdict

The current repository is a **good starting point** for an FYP technical report, but it is **not yet sufficient as a complete, standard model-comparison and fine-tuning record**.

You already have:
- reproducible experiment scripts,
- automatic metric export,
- per-epoch diagnostics for neural models,
- saved plots and checkpoints,
- a narrative analysis document.

However, for a stronger academic report, the project still needs a clearer **experiment log**, **hyperparameter tuning table**, **dataset snapshot/version record**, and **run-to-run reproducibility details**.

## What is already good enough

### 1. Model comparison pipeline exists
- `src/comparison/main.py` runs multiple models on one shared dataset split and writes comparison outputs to JSON/CSV.
- This is a solid foundation for a report section called **Model Comparison**.

### 2. Training diagnostics are already captured for neural models
- `src/common/train.py` stores per-epoch training/validation losses, early-stopping behavior, and selected hyperparameters in `reports/<model>_diagnostics.json`.
- This is useful evidence for a **Fine-Tuning / Training Dynamics** subsection.

### 3. Standard evaluation metrics are present
- The project computes MAE, MSE, and directional accuracy.
- These are acceptable baseline metrics for a forecasting comparison section.

### 4. Visual outputs are supported
- Loss curves, prediction slices, scatter plots, and comparison bar charts are already generated.
- These help the report look more complete and defensible.

## What is not yet good enough for a standard FYP report

### 1. Fine-tuning is not recorded systematically
Current config comments mention tuning ideas, but there is no structured experiment table showing:
- which hyperparameters were tried,
- why they were chosen,
- which model used which final settings,
- what metric was used to pick the final configuration.

**Why this matters:** in an FYP report, examiners usually expect a tuning record, not just final values.

### 2. Baseline and neural tasks are not fully aligned in documentation
- Baselines currently predict the next step using lag features.
- Neural models use `HORIZON` and default to a 10-step-ahead target.

That means the report must be very careful when claiming a direct comparison across all models. If not explained, this can weaken the validity of the comparison section.

### 3. The narrative report has scope inconsistencies
The existing analysis document says the task is **next-day log return prediction** and also says the input uses **past 10 days**, while the current config uses:
- `SEQ_LEN = 30`
- `HORIZON = 10`

So the written report is currently not fully aligned with the implementation.

### 4. No explicit reproducibility record
A strong FYP report usually states:
- data download date,
- exact date range used,
- package versions,
- random seeds,
- hardware/device used,
- number of runs per model.

At the moment, the repo does not capture these as part of experiment outputs.

### 5. No formal tuning summary table
There is no single file like:
- `reports/tuning_summary.csv`, or
- `docs/experiment_log.md`

that summarizes each experiment run and final chosen model.

### 6. Single-run results may be too weak academically
If results come from one run only, neural comparisons may be unstable. For an FYP, it is stronger to report either:
- repeated runs with mean ± standard deviation, or
- a clear note that results are based on one deterministic run with stated limitations.

## Minimum additions recommended before writing the final report

### A. Add an experiment log table
Create one table for every run with columns such as:
- run_id,
- date,
- model,
- seq_len,
- horizon,
- lr,
- batch_size,
- hidden/d_model/layers,
- patience,
- min_delta,
- best_epoch,
- best_val_MSE,
- test_MSE,
- test_MAE,
- DA,
- notes.

This can be CSV, JSON, or Markdown.

### B. Add a dedicated fine-tuning section
In the report, include:
1. hyperparameters considered,
2. search strategy (manual/grid/random),
3. selection criterion,
4. final chosen hyperparameters per model,
5. observations from tuning.

### C. Fix task-definition consistency
Before submission, make sure every document says the same thing about:
- prediction horizon,
- input window length,
- whether baselines and neural models solve the same target,
- train/validation/test split.

### D. Record reproducibility metadata
For each experiment, save:
- timestamp,
- device,
- Python/package versions,
- seed,
- ticker and date range,
- number of samples in each split.

### E. Make final comparison academically tighter
For the final report, include at least one of these:
- repeated runs,
- confidence intervals / standard deviations,
- justification for single-run comparison.

## Suggested report structure

### 1. Experimental Setup
- dataset source,
- date range,
- target definition,
- input representation,
- split method,
- preprocessing/scaling.

### 2. Models Compared
- Persistence,
- Linear Regression,
- RNN,
- LSTM,
- GRU,
- Transformer.

### 3. Hyperparameter Tuning
- search space,
- tuning method,
- validation criterion,
- final settings table.

### 4. Results
- comparison table,
- loss curves,
- prediction plots,
- directional accuracy discussion.

### 5. Discussion
- strongest model by MSE,
- strongest model by DA,
- why linear baselines are competitive,
- limitations of the Transformer in this setup,
- threats to validity.

## Bottom line

**Short answer:** the current logs and report are **partly good enough**, but **not yet strong enough for a standard FYP technical report without additional experiment-recording structure**.

If you add:
1. a formal tuning log,
2. reproducibility metadata,
3. consistent task definitions,
4. a clearer experiment summary table,

then the project will be in much better shape for final submission.
