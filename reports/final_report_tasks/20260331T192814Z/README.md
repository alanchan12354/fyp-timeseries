# Final-report multi-task experiment bundle

Generated at: 2026-03-31T20:31:21Z

This folder contains one subfolder per final-report task.
Each task subfolder stores its own tuning history, tuned-best comparison reports, and charts.
The root folder also includes an overall summary that compares the best model from each task.

## Tasks

- `sine_next_day/`
  - tuning outputs: `tuning_runs.csv`, `tuning_winners.csv`
  - comparison reports: `best_tuned_comparison_sine_next_day.csv`, `best_tuned_comparison_sine_next_day.md`
  - impact report: `hyperparameter_impact_report_sine_next_day.md`
  - figures: `figures/`
- `next_return/`
  - tuning outputs: `tuning_runs.csv`, `tuning_winners.csv`
  - comparison reports: `best_tuned_comparison_next_return.csv`, `best_tuned_comparison_next_return.md`
  - impact report: `hyperparameter_impact_report_next_return.md`
  - figures: `figures/`
- `next_volatility/`
  - tuning outputs: `tuning_runs.csv`, `tuning_winners.csv`
  - comparison reports: `best_tuned_comparison_next_volatility.csv`, `best_tuned_comparison_next_volatility.md`
  - impact report: `hyperparameter_impact_report_next_volatility.md`
  - figures: `figures/`
- `next_mean_return/`
  - tuning outputs: `tuning_runs.csv`, `tuning_winners.csv`
  - comparison reports: `best_tuned_comparison_next_mean_return.csv`, `best_tuned_comparison_next_mean_return.md`
  - impact report: `hyperparameter_impact_report_next_mean_return.md`
  - figures: `figures/`

## Overall cross-task outputs

- `overall_task_summary.csv`
- `overall_task_summary.md`

## Re-run command

```bash
bash scripts/run_multitask_final_reports.sh "reports/final_report_tasks/20260331T192814Z"
```
