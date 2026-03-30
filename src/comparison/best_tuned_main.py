from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

from src.common.config import HORIZON, RANDOM_SEED, REPORTS_DIR, SEQ_LEN, TARGET_MODE, TARGET_SMOOTH_WINDOW

from .best_configs import (
    CANONICAL_SOURCE,
    BestConfigSelection,
    ensure_task_ids_have_tuning_winners,
    load_best_configs,
)

CSV_REPORT_PATH = Path(REPORTS_DIR) / "best_tuned_comparison.csv"
MD_REPORT_PATH = Path(REPORTS_DIR) / "best_tuned_comparison.md"
MULTI_TASK_SUMMARY_PATH = Path(REPORTS_DIR) / "multi_task_summary.md"
MODEL_ORDER = ("lstm", "gru", "rnn", "transformer")
BASELINE_NAME = "Baseline-LR"
DISPLAY_NAMES = {
    "lstm": "LSTM",
    "gru": "GRU",
    "rnn": "RNN",
    "transformer": "Transformer",
}



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare models using tuned best hyperparameter configs.")
    parser.add_argument(
        "--config-source",
        choices=("tuning_winners", "tuning_best_configs"),
        default=CANONICAL_SOURCE,
        help="Which tuning artifact should be treated as the source of best per-model hyperparameters.",
    )
    parser.add_argument("--config-path", default=None, help="Optional override path to the selected tuning CSV artifact.")
    parser.add_argument("--task-id", help="Single task identifier to scope tuned config loading and reporting.")
    parser.add_argument(
        "--task-ids",
        nargs="+",
        help="One or more task identifiers to generate per-task tuned comparisons. Overrides --task-id.",
    )
    parser.add_argument("--horizon", type=int, default=None, help="Optional target horizon override used for comparison reruns.")
    parser.add_argument(
        "--data-source",
        choices=("spy", "sine"),
        default=None,
        help="Optional data source override used for comparison reruns.",
    )
    parser.add_argument(
        "--target-mode",
        choices=("horizon_return", "next_return", "next3_mean_return", "next_mean_return", "next_volatility", "sine_next_day"),
        default=None,
        help="Optional target-mode override used for comparison reruns.",
    )
    parser.add_argument(
        "--target-smooth-window",
        type=int,
        default=None,
        help="Optional target smoothing window override used for comparison reruns.",
    )
    return parser



def run_best_tuned_comparison(
    *,
    selection: BestConfigSelection,
    horizon: int | None = None,
    data_source: str | None = None,
    target_mode: str | None = None,
    target_smooth_window: int | None = None,
    task_id: str | None = None,
) -> list[dict[str, Any]]:
    entrypoints = _load_entrypoints()
    prepared_runs = _prepare_runs(
        selection.configs,
        horizon=horizon,
        data_source=data_source,
        target_mode=target_mode,
        target_smooth_window=target_smooth_window,
        task_id=task_id,
    )
    shared_run = prepared_runs[MODEL_ORDER[0]]
    rows = [_build_baseline_row(shared_run=shared_run, selection=selection)]
    for model_key in MODEL_ORDER:
        runtime_config = dict(selection.configs[model_key])
        if horizon is not None:
            runtime_config["horizon"] = int(horizon)
        if data_source is not None:
            runtime_config["data_source"] = data_source
        if target_mode is not None:
            runtime_config["target_mode"] = target_mode
        if target_smooth_window is not None:
            runtime_config["target_smooth_window"] = int(target_smooth_window)
        if task_id:
            runtime_config["task_id"] = task_id
        prepared_run = prepared_runs[model_key]
        run_id = prepared_run.run_context["run_id"]
        metrics = _run_entrypoint(
            entrypoints[model_key],
            config_dict={
                **runtime_config,
                "run_note": (
                    f"Best-tuned comparison using {selection.source_key}. "
                    f"Canonical source: {selection.source_path.name}."
                ),
            },
            prepared_run=prepared_run,
        )
        rows.append(
            build_report_row(
                model_name=DISPLAY_NAMES[model_key],
                config_source=selection.source_path.name,
                tuned_config=runtime_config,
                metrics=metrics,
                run_id=run_id,
            )
        )

    return sorted(rows, key=lambda row: (row["best_val_MSE"], row["best_test_MSE"], row["model"]))


def _build_baseline_row(*, shared_run: Any, selection: BestConfigSelection) -> dict[str, Any]:
    from src.baseline_lr.train import main as baseline_lr_main

    metrics = baseline_lr_main(
        config_dict={
            "run_note": (
                f"Best-tuned comparison using {selection.source_key}. "
                f"Canonical source: {selection.source_path.name}."
            ),
            "task_id": shared_run.run_context["task"]["task_id"],
            "data_source": shared_run.run_context["task"]["data_source"],
            "target_mode": shared_run.run_context["task"]["target_mode"],
            "target_smooth_window": shared_run.run_context["task"]["target_smooth_window"],
            "horizon": shared_run.run_context["task"]["prediction_horizon"],
            "seq_len": shared_run.run_context["task"]["input_window"],
            "batch_size": 64,
            "learning_rate": 0.001,
        },
        prepared_run=shared_run,
    )

    return build_report_row(
        model_name=BASELINE_NAME,
        config_source=selection.source_path.name,
        tuned_config={"model": "LinearRegression", "flattened_sequence": True, "seq_len": shared_run.run_context["task"]["input_window"]},
        metrics=metrics,
        run_id=f"{shared_run.run_context['run_id']}-baseline-lr",
    )



def _prepare_runs(
    configs: dict[str, dict[str, Any]],
    *,
    horizon: int | None = None,
    data_source: str | None = None,
    target_mode: str | None = None,
    target_smooth_window: int | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    from src.common.experiment import prepare_sequence_experiment_run

    return {
        model_key: prepare_sequence_experiment_run(
            batch_size=int(configs[model_key]["batch_size"]),
            experiment_name=f"best_tuned_{model_key}_comparison",
            run_note="Shared split prepared for best-tuned model comparison.",
            training_metadata={
                "lr": configs[model_key]["lr"],
                "batch_size": configs[model_key]["batch_size"],
                "seq_len": int(configs[model_key].get("seq_len", SEQ_LEN)),
            },
            seq_len=int(configs[model_key].get("seq_len", SEQ_LEN)),
            horizon=int(horizon) if horizon is not None else int(HORIZON),
            data_source=(data_source or "spy"),
            target_mode=(target_mode or TARGET_MODE),
            target_smooth_window=int(target_smooth_window) if target_smooth_window is not None else int(TARGET_SMOOTH_WINDOW),
            task_id=task_id,
            random_seed=int(configs[model_key].get("random_seed", RANDOM_SEED)),
        )
        for model_key in MODEL_ORDER
    }



def _load_entrypoints() -> dict[str, Callable[..., dict[str, Any]]]:
    from src.gru.train import main as gru_main
    from src.lstm.train import main as lstm_main
    from src.rnn.train import main as rnn_main
    from src.transformer.train import main as transformer_main

    return {
        "lstm": lstm_main,
        "gru": gru_main,
        "rnn": rnn_main,
        "transformer": transformer_main,
    }



def _run_entrypoint(entrypoint: Callable[..., dict[str, Any]], *, config_dict: dict[str, Any], prepared_run: Any) -> dict[str, Any]:
    return entrypoint(config_dict=config_dict, prepared_run=prepared_run)



def build_report_row(*, model_name: str, config_source: str, tuned_config: dict[str, Any], metrics: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "tuned_hyperparameters": json.dumps(tuned_config, sort_keys=True),
        "best_train_MSE": float(metrics.get("best_train_MSE", float("nan"))),
        "best_val_MSE": float(metrics["best_val_MSE"]),
        "best_test_MSE": float(metrics.get("best_test_MSE", metrics.get("MSE"))),
        "MSE": float(metrics["MSE"]),
        "MAE": float(metrics["MAE"]),
        "DA": float(metrics["DA"]),
        "run_id": run_id,
        "config_source": config_source,
    }



def write_reports(results: list[dict[str, Any]], *, selection: BestConfigSelection, csv_path: Path = CSV_REPORT_PATH, md_path: Path = MD_REPORT_PATH) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    md_path.write_text(build_markdown_report(results, selection=selection), encoding="utf-8")


def _resolve_task_ids(args: argparse.Namespace) -> list[str]:
    if args.task_ids:
        task_ids = [task_id.strip() for task_id in args.task_ids if task_id and task_id.strip()]
    elif args.task_id:
        task_ids = [args.task_id.strip()]
    else:
        task_ids = []
    return list(dict.fromkeys(task_ids))


def _slugify_task_id(task_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in task_id.strip()) or "unknown_task"


def _task_output_paths(task_id: str | None) -> tuple[Path, Path]:
    if not task_id:
        return CSV_REPORT_PATH, MD_REPORT_PATH
    suffix = _slugify_task_id(task_id)
    return (
        Path(REPORTS_DIR) / f"best_tuned_comparison_{suffix}.csv",
        Path(REPORTS_DIR) / f"best_tuned_comparison_{suffix}.md",
    )


def _write_multi_task_summary(entries: list[dict[str, str]], out_path: Path = MULTI_TASK_SUMMARY_PATH) -> None:
    lines = [
        "# Multi-task best tuned comparison summary",
        "",
        "| Task ID | Markdown report | CSV report |",
        "| --- | --- | --- |",
    ]
    for entry in entries:
        md_rel = entry["md_path"]
        csv_rel = entry["csv_path"]
        lines.append(
            f"| `{entry['task_id']}` | [{Path(md_rel).name}]({md_rel}) | [{Path(csv_rel).name}]({csv_rel}) |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def build_markdown_report(results: list[dict[str, Any]], *, selection: BestConfigSelection) -> str:
    best_val = min(results, key=lambda row: row["best_val_MSE"])
    best_test = min(results, key=lambda row: row["best_test_MSE"])
    ranking = sorted(results, key=lambda row: (row["best_val_MSE"], row["best_test_MSE"], row["model"]))

    lines = [
        "# Best Tuned Model Comparison",
        "",
        f"- Config source: `{selection.source_path.name}`",
        f"- Source note: {selection.note}",
        "- Baseline: shared flattened-sequence linear regression on the same split",
        "",
        "## Summary",
        "",
        f"- Best model by validation MSE: **{best_val['model']}** ({best_val['best_val_MSE']:.12f})",
        f"- Best model by test MSE: **{best_test['model']}** ({best_test['best_test_MSE']:.12f})",
        "- Ranking by validation MSE:",
    ]
    for index, row in enumerate(ranking, start=1):
        lines.append(f"  {index}. {row['model']} — val MSE {row['best_val_MSE']:.12f}, test MSE {row['best_test_MSE']:.12f}")
    lines.extend(
        [
            f"- Note: This comparison {selection.note.lower()}",
            "",
            "## Results",
            "",
            "| Model | Hyperparameters | Train MSE | Validation MSE | Test MSE | MAE | Directional Accuracy | Run ID | Config source |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in results:
        lines.append(
            "| {model} | `{params}` | {best_train_MSE:.12f} | {best_val_MSE:.12f} | {best_test_MSE:.12f} | {MAE:.12f} | {DA:.6f} | `{run_id}` | `{config_source}` |".format(
                model=row["model"],
                params=row["tuned_hyperparameters"],
                best_train_MSE=row["best_train_MSE"],
                best_val_MSE=row["best_val_MSE"],
                best_test_MSE=row["best_test_MSE"],
                MAE=row["MAE"],
                DA=row["DA"],
                run_id=row["run_id"],
                config_source=row["config_source"],
            )
        )
    return "\n".join(lines) + "\n"



def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    args = build_parser().parse_args(argv)
    task_ids = _resolve_task_ids(args)
    ensure_task_ids_have_tuning_winners(task_ids=task_ids)

    if not task_ids:
        selection = load_best_configs(args.config_source, source_path=args.config_path)
        results = run_best_tuned_comparison(
            selection=selection,
            horizon=args.horizon,
            data_source=args.data_source,
            target_mode=args.target_mode,
            target_smooth_window=args.target_smooth_window,
            task_id=args.task_id,
        )
        write_reports(results, selection=selection)
        print(json.dumps(results, indent=2))
        print(f"Wrote {CSV_REPORT_PATH}")
        print(f"Wrote {MD_REPORT_PATH}")
        return results

    aggregate_results: list[dict[str, Any]] = []
    summary_entries: list[dict[str, str]] = []
    for task_id in task_ids:
        selection = load_best_configs(
            args.config_source,
            source_path=args.config_path,
            task_ids=[task_id],
        )
        results = run_best_tuned_comparison(
            selection=selection,
            horizon=args.horizon,
            data_source=args.data_source,
            target_mode=args.target_mode,
            target_smooth_window=args.target_smooth_window,
            task_id=task_id,
        )
        csv_path, md_path = _task_output_paths(task_id)
        write_reports(results, selection=selection, csv_path=csv_path, md_path=md_path)
        summary_entries.append(
            {
                "task_id": task_id,
                "csv_path": str(csv_path.relative_to(REPORTS_DIR)),
                "md_path": str(md_path.relative_to(REPORTS_DIR)),
            }
        )
        aggregate_results.extend(results)
        print(f"Wrote {csv_path}")
        print(f"Wrote {md_path}")

    if len(task_ids) > 1:
        _write_multi_task_summary(summary_entries)
        print(f"Wrote {MULTI_TASK_SUMMARY_PATH}")
    print(json.dumps(aggregate_results, indent=2))
    return aggregate_results


if __name__ == "__main__":
    main()
