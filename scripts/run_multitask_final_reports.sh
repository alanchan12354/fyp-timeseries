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
run_task "spy_next5_volatility" "next_volatility" 5 "spy"
run_task "spy_next5_mean_return" "next_mean_return" 5 "spy"

cat > "${ROOT_DIR}/README.md" <<EOF
# Final-report multi-task experiment bundle

Generated at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")

This folder contains one subfolder per final-report task.  
Each task subfolder stores its own tuning history, tuned-best comparison reports, and charts.

## Tasks

- \`sine_next_day/\`
  - tuning outputs: \`tuning_runs.csv\`, \`tuning_winners.csv\`
  - comparison reports: \`best_tuned_comparison_sine_next_day.csv\`, \`best_tuned_comparison_sine_next_day.md\`
  - impact report: \`hyperparameter_impact_report_sine_next_day.md\`
  - figures: \`figures/\`
- \`spy_next5_volatility/\`
  - tuning outputs: \`tuning_runs.csv\`, \`tuning_winners.csv\`
  - comparison reports: \`best_tuned_comparison_spy_next5_volatility.csv\`, \`best_tuned_comparison_spy_next5_volatility.md\`
  - impact report: \`hyperparameter_impact_report_spy_next5_volatility.md\`
  - figures: \`figures/\`
- \`spy_next5_mean_return/\`
  - tuning outputs: \`tuning_runs.csv\`, \`tuning_winners.csv\`
  - comparison reports: \`best_tuned_comparison_spy_next5_mean_return.csv\`, \`best_tuned_comparison_spy_next5_mean_return.md\`
  - impact report: \`hyperparameter_impact_report_spy_next5_mean_return.md\`
  - figures: \`figures/\`

## Re-run command

\`\`\`bash
bash scripts/run_multitask_final_reports.sh "${ROOT_DIR}"
\`\`\`
EOF

echo ""
echo "Done. Multi-task reports written to: ${ROOT_DIR}"
