#!/usr/bin/env pwsh
param(
  [string]$RootDir = (Join-Path 'reports/final_report_tasks' ([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

New-Item -ItemType Directory -Force -Path $RootDir | Out-Null

# Use a non-interactive Matplotlib backend to avoid Tk/Tcl teardown crashes
# in headless or non-main-thread contexts (common on Windows shells).
if (-not $env:MPLBACKEND) {
  $env:MPLBACKEND = 'Agg'
}

function Invoke-Task {
  param(
    [Parameter(Mandatory = $true)][string]$TaskId,
    [Parameter(Mandatory = $true)][string]$TargetMode,
    [Parameter(Mandatory = $true)][int]$SmoothWindow,
    [Parameter(Mandatory = $true)][string]$DataSource
  )

  $TaskDir = Join-Path $RootDir $TaskId
  New-Item -ItemType Directory -Force -Path $TaskDir | Out-Null

  Write-Host ''
  Write-Host '============================================================'
  Write-Host "Task: $TaskId"
  Write-Host "Target mode: $TargetMode | smooth window: $SmoothWindow | data source: $DataSource"
  Write-Host "Reports dir: $TaskDir"
  Write-Host '============================================================'

  $oldReportsDir = $env:FYP_REPORTS_DIR
  $oldDisableSessionDir = $env:FYP_REPORTS_DISABLE_SESSION_DIR
  try {
    $env:FYP_REPORTS_DIR = $TaskDir
    $env:FYP_REPORTS_DISABLE_SESSION_DIR = '1'

    & python -m src.tuning.main `
      --model all `
      --session-mode reset `
      --task-id $TaskId `
      --horizon 1 `
      --data-source $DataSource `
      --target-mode $TargetMode `
      --target-smooth-window $SmoothWindow

    & python -m src.comparison.best_tuned_main `
      --task-id $TaskId `
      --horizon 1 `
      --data-source $DataSource `
      --target-mode $TargetMode `
      --target-smooth-window $SmoothWindow

    & python -m src.tuning.report --task-ids $TaskId

    & python -m src.comparison.best_tuned_charts `
      --csv-path (Join-Path $TaskDir "best_tuned_comparison_${TaskId}.csv") `
      --output-dir (Join-Path $TaskDir 'figures')
  }
  finally {
    $env:FYP_REPORTS_DIR = $oldReportsDir
    $env:FYP_REPORTS_DISABLE_SESSION_DIR = $oldDisableSessionDir
  }
}

Invoke-Task -TaskId 'sine_next_day' -TargetMode 'sine_next_day' -SmoothWindow 1 -DataSource 'sine'
Invoke-Task -TaskId 'next_return' -TargetMode 'next_return' -SmoothWindow 1 -DataSource 'spy'
Invoke-Task -TaskId 'next_volatility' -TargetMode 'next_volatility' -SmoothWindow 5 -DataSource 'spy'
Invoke-Task -TaskId 'next_mean_return' -TargetMode 'next_mean_return' -SmoothWindow 5 -DataSource 'spy'

$summaryScript = @'
import csv
import os
import sys

root_dir = sys.argv[1]
tasks = [
    {"task_id": "sine_next_day", "target_mode": "sine_next_day", "horizon": "1", "target_smooth_window": "1"},
    {"task_id": "next_return", "target_mode": "next_return", "horizon": "1", "target_smooth_window": "1"},
    {"task_id": "next_volatility", "target_mode": "next_volatility", "horizon": "1", "target_smooth_window": "5"},
    {"task_id": "next_mean_return", "target_mode": "next_mean_return", "horizon": "1", "target_smooth_window": "5"},
]
summary_csv = os.path.join(root_dir, "overall_task_summary.csv")
summary_md = os.path.join(root_dir, "overall_task_summary.md")

rows = []
for task in tasks:
    task_id = task["task_id"]
    comparison_path = os.path.join(root_dir, task_id, f"best_tuned_comparison_{task_id}.csv")
    if not os.path.exists(comparison_path):
        continue
    with open(comparison_path, "r", encoding="utf-8", newline="") as f:
        model_rows = list(csv.DictReader(f))
    if not model_rows:
        continue
    if "best_test_MSE" in model_rows[0]:
        best_row = min(model_rows, key=lambda row: float(row["best_test_MSE"]))
        best_test_mse = best_row.get("best_test_MSE", "")
        best_val_mse = best_row.get("best_val_MSE", "")
        best_train_mse = best_row.get("best_train_MSE", "")
    else:
        best_row = min(model_rows, key=lambda row: float(row["test_mse"]))
        best_test_mse = best_row.get("test_mse", "")
        best_val_mse = best_row.get("val_mse", "")
        best_train_mse = best_row.get("train_mse", "")
    rows.append(
        {
            "task_id": task_id,
            "target_mode": task["target_mode"],
            "horizon": task["horizon"],
            "target_smooth_window": task["target_smooth_window"],
            "best_model_by_test_mse": best_row.get("model", ""),
            "best_test_mse": best_test_mse,
            "best_val_mse": best_val_mse,
            "best_train_mse": best_train_mse,
        }
    )

with open(summary_csv, "w", encoding="utf-8", newline="") as f:
    fieldnames = [
        "task_id",
        "target_mode",
        "horizon",
        "target_smooth_window",
        "best_model_by_test_mse",
        "best_test_mse",
        "best_val_mse",
        "best_train_mse",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

with open(summary_md, "w", encoding="utf-8") as f:
    f.write("# Overall task performance summary\\n\\n")
    f.write("This table compares the best-tuned winner in each of the four core tasks.\\n\\n")
    if not rows:
        f.write("_No per-task comparison files were found._\\n")
    else:
        f.write("| task_id | target_mode | horizon | target_smooth_window | best_model_by_test_mse | best_test_mse | best_val_mse | best_train_mse |\\n")
        f.write("|---|---|---:|---:|---|---:|---:|---:|\\n")
        for row in rows:
            f.write(
                f"| {row['task_id']} | {row['target_mode']} | {row['horizon']} | {row['target_smooth_window']} | "
                f"{row['best_model_by_test_mse']} | {row['best_test_mse']} | {row['best_val_mse']} | {row['best_train_mse']} |\\n"
            )
'@

$summaryTemp = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "fyp_multitask_summary_$([guid]::NewGuid().ToString('N')).py")
Set-Content -Path $summaryTemp -Value $summaryScript -NoNewline -Encoding UTF8
try {
  & python $summaryTemp $RootDir
}
finally {
  Remove-Item -Path $summaryTemp -ErrorAction SilentlyContinue
}

$generatedAt = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
$readmeBody = @"
# Final-report multi-task experiment bundle

Generated at: $generatedAt

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

```pwsh
pwsh -File scripts/run_multitask_final_reports.ps1 -RootDir '$RootDir'
```

```bash
bash scripts/run_multitask_final_reports.sh "$RootDir"
```
"@
Set-Content -Path (Join-Path $RootDir 'README.md') -Value $readmeBody -Encoding UTF8

Write-Host ''
Write-Host "Done. Multi-task reports written to: $RootDir"
