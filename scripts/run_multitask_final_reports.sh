#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-reports/final_report_tasks/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${ROOT_DIR}"

run_task() {
  local task_id="$1"
  local target_mode="$2"
  local smooth_window="$3"
  local data_source="$4"

  local task_dir="${ROOT_DIR}/${task_id}"
  mkdir -p "${task_dir}"

  echo ""
  echo "============================================================"
  echo "Task: ${task_id}"
  echo "Target mode: ${target_mode} | smooth window: ${smooth_window} | data source: ${data_source}"
  echo "Reports dir: ${task_dir}"
  echo "============================================================"

  FYP_REPORTS_DIR="${task_dir}" FYP_REPORTS_DISABLE_SESSION_DIR=1 \
    python -m src.tuning.main \
      --model all \
      --session-mode reset \
      --task-id "${task_id}" \
      --horizon 1 \
      --data-source "${data_source}" \
      --target-mode "${target_mode}" \
      --target-smooth-window "${smooth_window}"

  FYP_REPORTS_DIR="${task_dir}" FYP_REPORTS_DISABLE_SESSION_DIR=1 \
    python -m src.comparison.best_tuned_main --task-id "${task_id}"

  FYP_REPORTS_DIR="${task_dir}" FYP_REPORTS_DISABLE_SESSION_DIR=1 \
    python -m src.tuning.report --task-ids "${task_id}"

  FYP_REPORTS_DIR="${task_dir}" FYP_REPORTS_DISABLE_SESSION_DIR=1 \
    python -m src.comparison.best_tuned_charts \
      --csv-path "${task_dir}/best_tuned_comparison_${task_id}.csv" \
      --output-dir "${task_dir}/figures"
}

run_task "sine_next_day" "sine_next_day" 1 "sine"
run_task "next_return" "next_return" 1 "spy"
run_task "next_volatility" "next_volatility" 5 "spy"
run_task "next_mean_return" "next_mean_return" 5 "spy"

python - <<'PY' "${ROOT_DIR}"
import csv
import os
import sys

root_dir = sys.argv[1]
tasks = ["sine_next_day", "next_return", "next_volatility", "next_mean_return"]
summary_csv = os.path.join(root_dir, "overall_task_summary.csv")
summary_md = os.path.join(root_dir, "overall_task_summary.md")

rows = []
for task_id in tasks:
    comparison_path = os.path.join(root_dir, task_id, f"best_tuned_comparison_{task_id}.csv")
    if not os.path.exists(comparison_path):
        continue
    with open(comparison_path, "r", encoding="utf-8", newline="") as f:
        model_rows = list(csv.DictReader(f))
    if not model_rows:
        continue
    best_row = min(model_rows, key=lambda row: float(row["test_mse"]))
    rows.append(
        {
            "task_id": task_id,
            "target_mode": best_row.get("target_mode", ""),
            "horizon": best_row.get("horizon", ""),
            "target_smooth_window": best_row.get("target_smooth_window", ""),
            "best_model_by_test_mse": best_row.get("model", ""),
            "best_test_mse": best_row.get("test_mse", ""),
            "best_val_mse": best_row.get("val_mse", ""),
            "best_train_mse": best_row.get("train_mse", ""),
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
    f.write("# Overall task performance summary\n\n")
    f.write("This table compares the best-tuned winner in each of the four core tasks.\n\n")
    if not rows:
        f.write("_No per-task comparison files were found._\n")
    else:
        f.write("| task_id | target_mode | horizon | target_smooth_window | best_model_by_test_mse | best_test_mse | best_val_mse | best_train_mse |\n")
        f.write("|---|---|---:|---:|---|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['task_id']} | {row['target_mode']} | {row['horizon']} | {row['target_smooth_window']} | "
                f"{row['best_model_by_test_mse']} | {row['best_test_mse']} | {row['best_val_mse']} | {row['best_train_mse']} |\n"
            )
PY

cat > "${ROOT_DIR}/README.md" <<EOF
# Final-report multi-task experiment bundle

Generated at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")

This folder contains one subfolder per final-report task.
Each task subfolder stores its own tuning history, tuned-best comparison reports, and charts.
The root folder also includes an overall summary that compares the best model from each task.

## Tasks

- \`sine_next_day/\`
  - tuning outputs: \`tuning_runs.csv\`, \`tuning_winners.csv\`
  - comparison reports: \`best_tuned_comparison_sine_next_day.csv\`, \`best_tuned_comparison_sine_next_day.md\`
  - impact report: \`hyperparameter_impact_report_sine_next_day.md\`
  - figures: \`figures/\`
- \`next_return/\`
  - tuning outputs: \`tuning_runs.csv\`, \`tuning_winners.csv\`
  - comparison reports: \`best_tuned_comparison_next_return.csv\`, \`best_tuned_comparison_next_return.md\`
  - impact report: \`hyperparameter_impact_report_next_return.md\`
  - figures: \`figures/\`
- \`next_volatility/\`
  - tuning outputs: \`tuning_runs.csv\`, \`tuning_winners.csv\`
  - comparison reports: \`best_tuned_comparison_next_volatility.csv\`, \`best_tuned_comparison_next_volatility.md\`
  - impact report: \`hyperparameter_impact_report_next_volatility.md\`
  - figures: \`figures/\`
- \`next_mean_return/\`
  - tuning outputs: \`tuning_runs.csv\`, \`tuning_winners.csv\`
  - comparison reports: \`best_tuned_comparison_next_mean_return.csv\`, \`best_tuned_comparison_next_mean_return.md\`
  - impact report: \`hyperparameter_impact_report_next_mean_return.md\`
  - figures: \`figures/\`

## Overall cross-task outputs

- \`overall_task_summary.csv\`
- \`overall_task_summary.md\`

## Re-run command

\`\`\`bash
bash scripts/run_multitask_final_reports.sh "${ROOT_DIR}"
\`\`\`
EOF

echo ""
echo "Done. Multi-task reports written to: ${ROOT_DIR}"
