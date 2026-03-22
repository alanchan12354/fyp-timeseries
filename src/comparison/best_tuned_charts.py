from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from html import escape
from pathlib import Path

try:
    from src.common.config import FIGURES_DIR, REPORTS_DIR
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    _BASE_DIR = Path(__file__).resolve().parents[2]
    REPORTS_DIR = str(_BASE_DIR / "reports")
    FIGURES_DIR = str(Path(REPORTS_DIR) / "figures")

DEFAULT_CSV_PATH = Path(REPORTS_DIR) / "best_tuned_comparison.csv"
DEFAULT_OUTPUT_DIR = Path(FIGURES_DIR)
MODEL_COLUMN = "model"
CHART_SPECS = (
    ("best_train_MSE", "Training Loss (MSE)", "best_tuned_training_loss.svg"),
    ("best_test_MSE", "Testing Loss (MSE)", "best_tuned_testing_loss.svg"),
    ("best_val_MSE", "Validation Loss (MSE)", "best_tuned_validation_loss.svg"),
)
_COLOR_CYCLE = ("#4C78A8", "#F58518", "#54A24B")


@dataclass(frozen=True)
class ChartArtifact:
    metric: str
    title: str
    output_path: Path


class BestTunedChartError(ValueError):
    pass


@dataclass(frozen=True)
class ComparisonRow:
    model: str
    metrics: dict[str, float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate SVG loss charts from best_tuned_comparison.csv.")
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV_PATH), help="Path to the best_tuned_comparison.csv file.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory where the SVG charts should be written.")
    return parser



def load_best_tuned_comparison(csv_path: str | Path = DEFAULT_CSV_PATH) -> list[ComparisonRow]:
    path = Path(csv_path)
    if not path.exists():
        raise BestTunedChartError(f"Comparison CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise BestTunedChartError(f"Comparison CSV has no header row: {path}")
        required_columns = {MODEL_COLUMN, *(metric for metric, _title, _filename in CHART_SPECS)}
        missing_columns = sorted(required_columns - set(reader.fieldnames))
        if missing_columns:
            raise BestTunedChartError(f"Comparison CSV is missing required column(s): {', '.join(missing_columns)}")

        rows: list[ComparisonRow] = []
        for row in reader:
            metrics = {metric: _parse_float(row.get(metric), metric=metric) for metric, _title, _filename in CHART_SPECS}
            rows.append(ComparisonRow(model=(row.get(MODEL_COLUMN) or "").strip(), metrics=metrics))

    if not rows:
        raise BestTunedChartError(f"Comparison CSV is empty: {path}")
    if any(not row.model for row in rows):
        raise BestTunedChartError(f"Comparison CSV contains a row without a model name: {path}")
    return rows



def _parse_float(value: str | None, *, metric: str) -> float:
    if value in (None, ""):
        raise BestTunedChartError(f"Missing value for metric '{metric}'.")
    try:
        return float(value)
    except ValueError as exc:
        raise BestTunedChartError(f"Invalid numeric value for metric '{metric}': {value}") from exc



def generate_best_tuned_svg_charts(csv_path: str | Path = DEFAULT_CSV_PATH, *, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> list[ChartArtifact]:
    rows = load_best_tuned_comparison(csv_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    artifacts: list[ChartArtifact] = []
    for index, (metric, title, filename) in enumerate(CHART_SPECS):
        output_path = destination / filename
        output_path.write_text(
            build_svg_chart(rows, metric=metric, title=title, bar_color=_COLOR_CYCLE[index % len(_COLOR_CYCLE)]),
            encoding="utf-8",
        )
        artifacts.append(ChartArtifact(metric=metric, title=title, output_path=output_path))
    return artifacts



def build_svg_chart(rows: list[ComparisonRow], *, metric: str, title: str, bar_color: str) -> str:
    ordered = sorted(rows, key=lambda row: (row.metrics[metric], row.model.lower()))
    width, height = 960, 540
    margin = {"top": 70, "right": 40, "bottom": 120, "left": 100}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]
    max_value = max(row.metrics[metric] for row in ordered)
    safe_max = max_value * 1.1 if max_value > 0 else 1.0
    bar_gap = 18
    bar_width = max(40, int((plot_width - (bar_gap * (len(ordered) - 1))) / max(len(ordered), 1)))
    total_bars_width = len(ordered) * bar_width + max(len(ordered) - 1, 0) * bar_gap
    start_x = margin["left"] + max((plot_width - total_bars_width) / 2, 0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2933}.title{font-size:24px;font-weight:bold}.axis{font-size:14px}.label{font-size:13px}.value{font-size:12px}.grid{stroke:#d9e2ec;stroke-width:1}.axis-line{stroke:#52606d;stroke-width:2}</style>',
        f'<rect width="{width}" height="{height}" fill="white" />',
        f'<text x="{width/2}" y="36" text-anchor="middle" class="title">{escape(title)}</text>',
    ]

    for tick_index in range(6):
        tick_value = safe_max * tick_index / 5
        y = margin["top"] + plot_height - ((tick_value / safe_max) * plot_height)
        parts.append(f'<line x1="{margin["left"]}" y1="{y:.2f}" x2="{width - margin["right"]}" y2="{y:.2f}" class="grid" />')
        parts.append(f'<text x="{margin["left"] - 12}" y="{y + 5:.2f}" text-anchor="end" class="axis">{tick_value:.6f}</text>')

    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_height}" x2="{width - margin["right"]}" y2="{margin["top"] + plot_height}" class="axis-line" />')
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_height}" class="axis-line" />')

    for index, row in enumerate(ordered):
        value = row.metrics[metric]
        bar_height = 0 if safe_max == 0 else (value / safe_max) * plot_height
        x = start_x + index * (bar_width + bar_gap)
        y = margin["top"] + plot_height - bar_height
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width}" height="{bar_height:.2f}" fill="{bar_color}" stroke="#1f2933" />')
        parts.append(f'<text x="{x + bar_width / 2:.2f}" y="{y - 8:.2f}" text-anchor="middle" class="value">{value:.6f}</text>')
        parts.append(f'<text x="{x + bar_width / 2:.2f}" y="{margin["top"] + plot_height + 24:.2f}" text-anchor="end" transform="rotate(-25 {x + bar_width / 2:.2f} {margin["top"] + plot_height + 24:.2f})" class="label">{escape(row.model)}</text>')

    parts.append(f'<text x="{margin["left"] + plot_width / 2}" y="{height - 30}" text-anchor="middle" class="axis">Model</text>')
    parts.append(f'<text x="24" y="{margin["top"] + plot_height / 2}" text-anchor="middle" transform="rotate(-90 24 {margin["top"] + plot_height / 2})" class="axis">MSE</text>')
    parts.append('</svg>')
    return ''.join(parts)



def main(argv: list[str] | None = None) -> list[ChartArtifact]:
    args = build_parser().parse_args(argv)
    artifacts = generate_best_tuned_svg_charts(args.csv_path, output_dir=args.output_dir)
    for artifact in artifacts:
        print(f"Saved {artifact.title} chart to {artifact.output_path}")
    return artifacts


if __name__ == "__main__":
    main()
