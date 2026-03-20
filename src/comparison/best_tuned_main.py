from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

from src.common.config import REPORTS_DIR

from .best_configs import CANONICAL_SOURCE, BestConfigSelection, load_best_configs

CSV_REPORT_PATH = Path(REPORTS_DIR) / "best_tuned_comparison.csv"
MD_REPORT_PATH = Path(REPORTS_DIR) / "best_tuned_comparison.md"
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
    return parser



def run_best_tuned_comparison(*, selection: BestConfigSelection) -> list[dict[str, Any]]:
    entrypoints = _load_entrypoints()
    prepared_runs = _prepare_runs(selection.configs)
    shared_run = prepared_runs[MODEL_ORDER[0]]
    rows = [_build_baseline_row(shared_run=shared_run, selection=selection)]
    for model_key in MODEL_ORDER:
        runtime_config = dict(selection.configs[model_key])
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
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from src.common.metrics import directional_accuracy

    X_tr, y_tr = shared_run.train_data
    X_va, y_va = shared_run.val_data
    X_te, y_te = shared_run.test_data

    X_tr_flat = X_tr.reshape(len(X_tr), -1)
    X_va_flat = X_va.reshape(len(X_va), -1)
    X_te_flat = X_te.reshape(len(X_te), -1)

    baseline = LinearRegression()
    baseline.fit(X_tr_flat, y_tr)

    yhat_tr = baseline.predict(X_tr_flat)
    yhat_va = baseline.predict(X_va_flat)
    yhat_te = baseline.predict(X_te_flat)

    return build_report_row(
        model_name=BASELINE_NAME,
        config_source=selection.source_path.name,
        tuned_config={"model": "LinearRegression", "flattened_sequence": True},
        metrics={
            "best_train_MSE": float(mean_squared_error(y_tr, yhat_tr)),
            "best_val_MSE": float(mean_squared_error(y_va, yhat_va)),
            "best_test_MSE": float(mean_squared_error(y_te, yhat_te)),
            "MSE": float(mean_squared_error(y_te, yhat_te)),
            "MAE": float(abs(y_te - yhat_te).mean()),
            "DA": directional_accuracy(y_te, yhat_te),
        },
        run_id=f"{shared_run.run_context['run_id']}-baseline-lr",
    )



def _prepare_runs(configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from src.common.experiment import prepare_sequence_experiment_run

    return {
        model_key: prepare_sequence_experiment_run(
            batch_size=int(configs[model_key]["batch_size"]),
            experiment_name=f"best_tuned_{model_key}_comparison",
            run_note="Shared split prepared for best-tuned model comparison.",
            training_metadata={"lr": configs[model_key]["lr"], "batch_size": configs[model_key]["batch_size"]},
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
    selection = load_best_configs(args.config_source, source_path=args.config_path)
    results = run_best_tuned_comparison(selection=selection)
    write_reports(results, selection=selection)
    print(json.dumps(results, indent=2))
    print(f"Wrote {CSV_REPORT_PATH}")
    print(f"Wrote {MD_REPORT_PATH}")
    return results


if __name__ == "__main__":
    main()
