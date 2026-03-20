import csv
import json
import math
import os
from collections import defaultdict
from html import escape
from typing import Any, Dict, Iterable, List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


EXPERIMENT_LOG_JSONL = os.path.join(REPORTS_DIR, "experiment_log.jsonl")
TUNING_WINNERS_CSV = os.path.join(REPORTS_DIR, "tuning_winners.csv")
REPORT_PATH = os.path.join(REPORTS_DIR, "hyperparameter_impact_report.md")
FIGURE_PATH = os.path.join(FIGURES_DIR, "hyperparameter_model_loss_summary.svg")


def _load_experiment_rows(path: str = EXPERIMENT_LOG_JSONL) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _is_tuning_run(row: Dict[str, Any]) -> bool:
    notes_metadata = row.get("notes_metadata") or {}
    notes = row.get("notes") or ""
    return row.get("record_type") == "neural_model" and (
        bool(notes_metadata.get("stage")) or "stage=" in notes
    )


def _parse_float(value: Any) -> float:
    if value in ("", None):
        return float("nan")
    return float(value)


def _format_metric(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.6g}"


def _format_change(previous: float, current: float) -> str:
    if math.isnan(previous) or math.isnan(current):
        return "n/a"
    delta = current - previous
    direction = "improved" if delta < 0 else "worsened" if delta > 0 else "matched"
    return f"{direction} by {abs(delta):.6g}"


def _collect_best_runs(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    best_by_model: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not _is_tuning_run(row):
            continue
        model_name = row["model_name"]
        current_best = best_by_model.get(model_name)
        if current_best is None or row["metrics"]["best_val_MSE"] < current_best["metrics"]["best_val_MSE"]:
            best_by_model[model_name] = row
    return best_by_model


def _collect_stage_impacts(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    previous_by_model: Dict[str, float] = {}
    display_names = {
        "lstm": "LSTM",
        "gru": "GRU",
        "rnn": "RNN",
        "transformer": "Transformer",
    }
    for row in rows:
        model_name = display_names.get(row["model"].lower(), row["model"])
        current = _parse_float(row["best_val_MSE"])
        previous = previous_by_model.get(model_name, float("nan"))
        grouped[model_name].append(
            {
                "stage_index": int(row["stage_index"]),
                "stage_name": row["stage_name"],
                "parameter_group": row["parameter_group"],
                "winning_value_json": row["winning_value_json"],
                "best_val_MSE": current,
                "change_from_previous": _format_change(previous, current),
            }
        )
        previous_by_model[model_name] = current
    return grouped


def _build_comparison_figure(best_by_model: Dict[str, Dict[str, Any]], out_path: str = FIGURE_PATH) -> None:
    models = sorted(best_by_model)
    loss_specs: List[Tuple[str, str]] = [
        ("best_train_MSE", "Training Loss"),
        ("best_test_MSE", "Testing Loss"),
        ("best_val_MSE", "Validation Loss"),
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _write_loss_summary_svg(models, loss_specs, best_by_model, out_path)


def _write_loss_summary_svg(
    models: List[str],
    loss_specs: List[Tuple[str, str]],
    best_by_model: Dict[str, Dict[str, Any]],
    out_path: str,
) -> None:
    width = 1380
    height = 430
    margin_top = 55
    margin_bottom = 75
    margin_left = 65
    panel_width = 390
    panel_gap = 40
    chart_height = 250
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759"]

    all_values = [
        float(best_by_model[model]["metrics"][metric_key])
        for model in models
        for metric_key, _ in loss_specs
    ]
    y_max = max(all_values) * 1.05 if all_values else 1.0
    y_ticks = [0.0, y_max * 0.25, y_max * 0.5, y_max * 0.75, y_max]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        '<text x="690" y="28" font-size="20" text-anchor="middle" font-family="Arial">Best Tuned Model Losses on a Shared Y-Axis</text>',
        '<text x="25" y="210" font-size="14" text-anchor="middle" transform="rotate(-90 25 210)" font-family="Arial">MSE</text>',
    ]

    for panel_index, (metric_key, title) in enumerate(loss_specs):
        panel_x = margin_left + panel_index * (panel_width + panel_gap)
        chart_y = margin_top
        chart_x2 = panel_x + panel_width
        chart_y2 = chart_y + chart_height
        parts.append(f'<text x="{panel_x + panel_width / 2}" y="45" font-size="16" text-anchor="middle" font-family="Arial">{escape(title)}</text>')

        for tick in y_ticks:
            y = chart_y2 - (tick / y_max * chart_height if y_max else 0)
            parts.append(f'<line x1="{panel_x}" y1="{y:.2f}" x2="{chart_x2}" y2="{y:.2f}" stroke="#dddddd" stroke-width="1" />')
            if panel_index == 0:
                parts.append(f'<text x="{panel_x - 8}" y="{y + 4:.2f}" font-size="10" text-anchor="end" font-family="Arial">{tick:.2e}</text>')

        parts.append(f'<line x1="{panel_x}" y1="{chart_y}" x2="{panel_x}" y2="{chart_y2}" stroke="#333333" stroke-width="1" />')
        parts.append(f'<line x1="{panel_x}" y1="{chart_y2}" x2="{chart_x2}" y2="{chart_y2}" stroke="#333333" stroke-width="1" />')

        bar_gap = 18
        bar_width = (panel_width - bar_gap * (len(models) + 1)) / max(len(models), 1)
        for idx, model in enumerate(models):
            value = float(best_by_model[model]["metrics"][metric_key])
            bar_height = (value / y_max * chart_height) if y_max else 0
            x = panel_x + bar_gap + idx * (bar_width + bar_gap)
            y = chart_y2 - bar_height
            color = colors[idx % len(colors)]
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}" />')
            parts.append(f'<text x="{x + bar_width / 2:.2f}" y="{y - 6:.2f}" font-size="10" text-anchor="middle" font-family="Arial">{value:.2e}</text>')
            parts.append(f'<text x="{x + bar_width / 2:.2f}" y="{chart_y2 + 20:.2f}" font-size="11" text-anchor="middle" font-family="Arial">{escape(model)}</text>')

        parts.append(f'<text x="{panel_x + panel_width / 2}" y="{height - 18}" font-size="12" text-anchor="middle" font-family="Arial">Model</text>')

    parts.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))


def _write_report(
    best_by_model: Dict[str, Dict[str, Any]],
    stage_impacts: Dict[str, List[Dict[str, Any]]],
    out_path: str = REPORT_PATH,
    figure_path: str = FIGURE_PATH,
) -> None:
    ordered_models = sorted(best_by_model.values(), key=lambda row: row["metrics"]["best_val_MSE"])
    lines: List[str] = [
        "# Hyper-Parameter Impact Report",
        "",
        "This report summarises how the tuning workflow changed model performance and compares the best tuned runs across models.",
        "",
        "## Best tuned configuration by model",
        "",
        "| Model | Best validation MSE | Best testing MSE | Best training MSE | MAE | DA | Hyperparameters | Run ID |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |",
    ]

    for row in ordered_models:
        metrics = row["metrics"]
        hyperparameters = json.dumps(row["hyperparameters"], sort_keys=True)
        lines.append(
            "| {model} | {best_val} | {best_test} | {best_train} | {mae} | {da} | `{hyper}` | `{run_id}` |".format(
                model=row["model_name"],
                best_val=_format_metric(metrics["best_val_MSE"]),
                best_test=_format_metric(metrics["best_test_MSE"]),
                best_train=_format_metric(metrics["best_train_MSE"]),
                mae=_format_metric(metrics["MAE"]),
                da=f"{metrics['DA']:.2%}",
                hyper=hyperparameters,
                run_id=row["run_id"],
            )
        )

    lines.extend(
        [
            "",
            "## Stage-by-stage hyper-parameter impact",
            "",
            "The tuning workflow was sequential, so each stage winner was selected while earlier winners stayed frozen.",
        ]
    )

    for model_name in sorted(stage_impacts):
        lines.extend(["", f"### {model_name}", ""])
        for stage in sorted(stage_impacts[model_name], key=lambda item: item["stage_index"]):
            lines.append(
                "- Stage {stage_index} (`{parameter_group}`): winner {winner} with validation MSE {metric}; relative to the previous stage this {change}.".format(
                    stage_index=stage["stage_index"],
                    parameter_group=stage["parameter_group"],
                    winner=stage["winning_value_json"],
                    metric=_format_metric(stage["best_val_MSE"]),
                    change=stage["change_from_previous"],
                )
            )

    best_validation_model = ordered_models[0]
    best_testing_model = min(ordered_models, key=lambda row: row["metrics"]["best_test_MSE"])
    best_direction_model = max(ordered_models, key=lambda row: row["metrics"]["DA"])

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- **Validation winner:** {best_validation_model['model_name']} achieved the lowest validation MSE at {_format_metric(best_validation_model['metrics']['best_val_MSE'])}.",
            f"- **Testing winner:** {best_testing_model['model_name']} achieved the lowest testing MSE at {_format_metric(best_testing_model['metrics']['best_test_MSE'])}.",
            f"- **Directional winner:** {best_direction_model['model_name']} achieved the highest directional accuracy at {best_direction_model['metrics']['DA']:.2%}.",
            "- Across the current tuning archive, recurrent models stayed tightly grouped, while the Transformer remained materially higher-loss than the recurrent models after tuning.",
            "",
            "## Figure",
            "",
            f"![Best tuned model losses]({os.path.relpath(figure_path, os.path.dirname(out_path))})",
            "",
            "The figure uses one shared y-axis across three subplots so the training, testing, and validation losses remain directly comparable.",
        ]
    )

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> Dict[str, str]:
    experiment_rows = _load_experiment_rows()
    best_by_model = _collect_best_runs(experiment_rows)
    stage_impacts = _collect_stage_impacts(_load_csv_rows(TUNING_WINNERS_CSV))
    _build_comparison_figure(best_by_model)
    _write_report(best_by_model, stage_impacts)
    return {"report_path": REPORT_PATH, "figure_path": FIGURE_PATH}


if __name__ == "__main__":
    outputs = main()
    print(json.dumps(outputs, indent=2))
