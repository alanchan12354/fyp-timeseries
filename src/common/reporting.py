import csv
import glob
import json
import os
import platform
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from importlib.metadata import PackageNotFoundError, version

import torch

from .config import (
    BASE_DIR,
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    FIGURES_DIR,
    HORIZON,
    LAGS,
    LR,
    MIN_DELTA,
    MIN_EPOCHS,
    PATIENCE,
    REPORTS_DIR,
    SEQ_LEN,
    START,
    TICKER,
    TRAIN_RATIO,
    TRAIN_LOG_EVERY,
    VAL_LOSS_SMOOTH_WINDOW,
    VAL_RATIO,
)


EXPERIMENT_LOG_JSONL = os.path.join(REPORTS_DIR, "experiment_log.jsonl")
EXPERIMENT_LOG_CSV = os.path.join(REPORTS_DIR, "experiment_log.csv")
MODEL_COMPARISON_RECORD_JSON = os.path.join(REPORTS_DIR, "model_comparison_record.json")
MODEL_COMPARISON_RECORD_CSV = os.path.join(REPORTS_DIR, "model_comparison_record.csv")
TUNING_ALL_RUNS_CSV = os.path.join(REPORTS_DIR, "tuning_all_runs.csv")
TUNING_BEST_CONFIGS_CSV = os.path.join(REPORTS_DIR, "tuning_best_configs.csv")


SUMMARY_FIELDNAMES = [
    "timestamp_utc",
    "run_id",
    "model_name",
    "record_type",
    "selection_metric",
    "selection_split",
    "hidden_size",
    "d_model",
    "nhead",
    "num_layers",
    "dropout",
    "lr",
    "batch_size",
    "hyperparameters_json",
    "best_val_MSE",
    "MSE",
    "MAE",
    "DA",
    "best_epoch",
    "tuning_notes",
    "notes",
]

TUNING_REPORT_FIELDNAMES = [
    "model",
    "run_id",
    "timestamp",
    "hyperparameters",
    "hidden_size",
    "d_model",
    "nhead",
    "num_layers",
    "best_val_MSE",
    "test_MSE",
    "MAE",
    "DA",
    "best_epoch",
    "notes",
    "selection_reason",
]

SELECTION_REASON_LOWEST_VAL_MSE = "selected_by=lowest_validation_MSE"


ARTIFACT_RESET_PATTERNS = [
    "experiment_log.csv",
    "experiment_log.jsonl",
    "*_diagnostics.json",
    "*.pt",
    "tuning_runs.csv",
    "tuning_winners.csv",
    "tuning_all_runs.csv",
    "tuning_best_configs.csv",
    "tuning_summary*.csv",
]
FIGURE_RESET_PATTERNS = ["*.png"]


def collect_tuning_artifacts_for_reset(*, reports_dir: str = REPORTS_DIR) -> List[str]:
    patterns = [os.path.join(reports_dir, pattern) for pattern in ARTIFACT_RESET_PATTERNS]
    figures_dir = os.path.join(reports_dir, os.path.basename(FIGURES_DIR))
    patterns.extend(os.path.join(figures_dir, pattern) for pattern in FIGURE_RESET_PATTERNS)

    matches = set()
    for pattern in patterns:
        matches.update(path for path in glob.glob(pattern) if os.path.isfile(path))
    return sorted(matches)


def reset_tuning_artifacts(*, reports_dir: str = REPORTS_DIR) -> Dict[str, Any]:
    removed_files: List[str] = []
    for path in collect_tuning_artifacts_for_reset(reports_dir=reports_dir):
        os.remove(path)
        removed_files.append(path)

    figures_dir = os.path.join(reports_dir, os.path.basename(FIGURES_DIR))
    empty_figures_dir = os.path.isdir(figures_dir) and not any(os.scandir(figures_dir))

    return {
        "reports_dir": reports_dir,
        "removed_count": len(removed_files),
        "removed_files": removed_files,
        "empty_figures_dir": empty_figures_dir,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=BASE_DIR,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return None


def _package_version(name: str) -> Optional[str]:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def default_environment_metadata() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": DEVICE,
        "git_commit": _safe_git_commit(),
        "packages": {
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit_learn": _package_version("scikit-learn"),
            "torch": torch.__version__,
            "matplotlib": _package_version("matplotlib"),
            "yfinance": _package_version("yfinance"),
        },
    }


def default_task_metadata(task_kind: str = "forecasting") -> Dict[str, Any]:
    return {
        "task_kind": task_kind,
        "ticker": TICKER,
        "start_date": START,
        "input_window": SEQ_LEN,
        "prediction_horizon": HORIZON,
        "baseline_lags": LAGS,
        "target": f"log return at t+{HORIZON}",
        "input_description": f"previous {SEQ_LEN} daily log returns",
        "split": {
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": 1.0 - TRAIN_RATIO - VAL_RATIO,
        },
    }


def split_metadata(train_size: int, val_size: int, test_size: int) -> Dict[str, int]:
    return {
        "train_samples": int(train_size),
        "val_samples": int(val_size),
        "test_samples": int(test_size),
    }


def default_training_metadata(**overrides: Any) -> Dict[str, Any]:
    metadata = {
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "min_epochs": MIN_EPOCHS,
        "val_loss_smooth_window": VAL_LOSS_SMOOTH_WINDOW,
        "train_log_every": TRAIN_LOG_EVERY,
    }
    metadata.update(overrides)
    return metadata


def create_run_context(
    experiment_name: str,
    split_meta: Dict[str, Any],
    *,
    comparison_group: Optional[str] = None,
    task_meta: Optional[Dict[str, Any]] = None,
    training_meta: Optional[Dict[str, Any]] = None,
    environment_meta: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "run_id": f"{experiment_name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "experiment_name": experiment_name,
        "comparison_group": comparison_group,
        "timestamp_utc": _utc_now(),
        "task": task_meta or default_task_metadata(),
        "split": deepcopy(split_meta),
        "training": training_meta or default_training_metadata(),
        "environment": environment_meta or default_environment_metadata(),
        "notes": notes,
    }


def build_experiment_record(
    *,
    model_name: str,
    record_type: str,
    metrics: Dict[str, Any],
    hyperparameters: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    selection_metric: str = "best_val_MSE",
    selection_split: str = "validation",
    tuning: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    context = deepcopy(context or {})
    record = {
        "timestamp_utc": _utc_now(),
        "run_id": context.get("run_id"),
        "experiment_name": context.get("experiment_name"),
        "comparison_group": context.get("comparison_group"),
        "model_name": model_name,
        "record_type": record_type,
        "selection_metric": selection_metric,
        "selection_split": selection_split,
        "task": deepcopy(context.get("task", default_task_metadata())),
        "split": deepcopy(context.get("split", {})),
        "training": deepcopy(context.get("training", default_training_metadata())),
        "environment": deepcopy(context.get("environment", default_environment_metadata())),
        "hyperparameters": deepcopy(hyperparameters or {}),
        "metrics": deepcopy(metrics),
        "tuning": deepcopy(tuning or {}),
        "artifacts": deepcopy(artifacts or {}),
        "notes": context.get("notes"),
    }
    if extra:
        record.update(deepcopy(extra))
    return record


def _extract_num_layers(hyper: Dict[str, Any]) -> Any:
    num_layers = hyper.get("layers")
    if num_layers is None:
        num_layers = hyper.get("num_layers")
    return num_layers


def _flatten_record_for_csv(record: Dict[str, Any]) -> Dict[str, Any]:
    training = record.get("training", {})
    metrics = record.get("metrics", {})
    hyper = record.get("hyperparameters", {})
    tuning = record.get("tuning", {})

    hidden_size = hyper.get("hidden")
    d_model = hyper.get("d_model")

    return {
        "timestamp_utc": record.get("timestamp_utc"),
        "run_id": record.get("run_id"),
        "model_name": record.get("model_name"),
        "record_type": record.get("record_type"),
        "selection_metric": record.get("selection_metric"),
        "selection_split": record.get("selection_split"),
        "hidden_size": hidden_size,
        "d_model": d_model,
        "nhead": hyper.get("nhead"),
        "num_layers": _extract_num_layers(hyper),
        "dropout": hyper.get("dropout"),
        "lr": training.get("lr"),
        "batch_size": training.get("batch_size"),
        "hyperparameters_json": json.dumps(hyper, sort_keys=True),
        "best_val_MSE": metrics.get("best_val_MSE"),
        "MSE": metrics.get("MSE"),
        "MAE": metrics.get("MAE"),
        "DA": metrics.get("DA"),
        "best_epoch": tuning.get("best_epoch"),
        "tuning_notes": tuning.get("tuning_notes"),
        "notes": record.get("notes"),
    }


def _stringify_hyperparameters(record: Dict[str, Any]) -> str:
    hyper = deepcopy(record.get("hyperparameters", {}))
    if not hyper:
        return ""
    return json.dumps(hyper, sort_keys=True, separators=(",", ":"))


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_experiment_log_records(reports_dir: str = REPORTS_DIR) -> List[Dict[str, Any]]:
    jsonl_path = os.path.join(reports_dir, os.path.basename(EXPERIMENT_LOG_JSONL))
    csv_path = os.path.join(reports_dir, os.path.basename(EXPERIMENT_LOG_CSV))

    if os.path.exists(jsonl_path) and os.path.getsize(jsonl_path) > 0:
        records: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        records = []
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                hyper = {}
                if row.get("hyperparameters_json"):
                    try:
                        hyper = json.loads(row["hyperparameters_json"])
                    except json.JSONDecodeError:
                        hyper = {}
                hidden_size = row.get("hidden_size")
                d_model = row.get("d_model")
                if hidden_size and "hidden" not in hyper and not d_model:
                    hyper["hidden"] = _coerce_int(hidden_size)
                if d_model and "d_model" not in hyper:
                    hyper["d_model"] = _coerce_int(d_model)
                if row.get("nhead") and "nhead" not in hyper:
                    hyper["nhead"] = _coerce_int(row.get("nhead"))
                if row.get("num_layers") and "num_layers" not in hyper and "layers" not in hyper:
                    hyper["num_layers"] = _coerce_int(row.get("num_layers"))
                records.append(
                    {
                        "timestamp_utc": row.get("timestamp_utc"),
                        "run_id": row.get("run_id"),
                        "model_name": row.get("model_name"),
                        "record_type": row.get("record_type"),
                        "selection_metric": row.get("selection_metric"),
                        "selection_split": row.get("selection_split"),
                        "hyperparameters": hyper,
                        "metrics": {
                            "best_val_MSE": _coerce_float(row.get("best_val_MSE")),
                            "MSE": _coerce_float(row.get("MSE")),
                            "MAE": _coerce_float(row.get("MAE")),
                            "DA": _coerce_float(row.get("DA")),
                        },
                        "training": {
                            "lr": _coerce_float(row.get("lr")),
                            "batch_size": _coerce_int(row.get("batch_size")),
                        },
                        "tuning": {
                            "best_epoch": _coerce_int(row.get("best_epoch")),
                            "tuning_notes": row.get("tuning_notes"),
                        },
                        "notes": row.get("notes"),
                    }
                )
        return records

    return []


def _is_tuning_record(record: Dict[str, Any]) -> bool:
    if str(record.get("record_type") or "").lower() != "neural_model":
        return False
    note_fields = [
        record.get("notes"),
        record.get("tuning", {}).get("tuning_notes"),
        record.get("training", {}).get("run_note"),
    ]
    note_blob = " ".join(str(value) for value in note_fields if value)
    return "sweep" in note_blob.lower()


def _tuning_row_from_record(record: Dict[str, Any], *, selection_reason: str = "") -> Dict[str, Any]:
    hyper = deepcopy(record.get("hyperparameters", {}))
    metrics = record.get("metrics", {})
    tuning = record.get("tuning", {})
    model_name = record.get("model_name")
    is_transformer = str(model_name or "").strip().lower() == "transformer"

    hidden_size = hyper.get("hidden")
    d_model = hyper.get("d_model")
    if is_transformer:
        hidden_size = None
    else:
        d_model = None

    return {
        "model": model_name,
        "run_id": record.get("run_id"),
        "timestamp": record.get("timestamp_utc"),
        "hyperparameters": _stringify_hyperparameters(record),
        "hidden_size": hidden_size,
        "d_model": d_model,
        "nhead": hyper.get("nhead") if is_transformer else None,
        "num_layers": _extract_num_layers(hyper),
        "best_val_MSE": metrics.get("best_val_MSE"),
        "test_MSE": metrics.get("MSE"),
        "MAE": metrics.get("MAE"),
        "DA": metrics.get("DA"),
        "best_epoch": tuning.get("best_epoch"),
        "notes": record.get("notes") or tuning.get("tuning_notes"),
        "selection_reason": selection_reason,
    }


def generate_tuning_reports(*, reports_dir: str = REPORTS_DIR) -> Dict[str, Any]:
    records = _load_experiment_log_records(reports_dir)
    tuning_records = [record for record in records if _is_tuning_record(record)]

    all_rows = [_tuning_row_from_record(record) for record in tuning_records]
    all_rows.sort(
        key=lambda row: (
            str(row.get("model") or ""),
            row.get("best_val_MSE") is None,
            row.get("best_val_MSE") if row.get("best_val_MSE") is not None else float("inf"),
            str(row.get("timestamp") or ""),
            str(row.get("run_id") or ""),
        )
    )

    best_rows: List[Dict[str, Any]] = []
    rows_by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        rows_by_model.setdefault(str(row.get("model")), []).append(row)

    for model_name in sorted(rows_by_model):
        model_rows = rows_by_model[model_name]
        winner = min(
            model_rows,
            key=lambda row: (
                row.get("best_val_MSE") is None,
                row.get("best_val_MSE") if row.get("best_val_MSE") is not None else float("inf"),
                str(row.get("timestamp") or ""),
                str(row.get("run_id") or ""),
            ),
        )
        winner_with_reason = deepcopy(winner)
        winner_with_reason["selection_reason"] = SELECTION_REASON_LOWEST_VAL_MSE
        best_rows.append(winner_with_reason)

    all_rows_with_reason = []
    best_keys = {(row["model"], row["run_id"], row["timestamp"]) for row in best_rows}
    for row in all_rows:
        row_copy = deepcopy(row)
        if (row_copy["model"], row_copy["run_id"], row_copy["timestamp"]) in best_keys:
            row_copy["selection_reason"] = SELECTION_REASON_LOWEST_VAL_MSE
        all_rows_with_reason.append(row_copy)

    os.makedirs(reports_dir, exist_ok=True)
    all_runs_path = os.path.join(reports_dir, os.path.basename(TUNING_ALL_RUNS_CSV))
    best_configs_path = os.path.join(reports_dir, os.path.basename(TUNING_BEST_CONFIGS_CSV))

    with open(all_runs_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TUNING_REPORT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows_with_reason)

    with open(best_configs_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TUNING_REPORT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(best_rows)

    return {
        "source_records": len(records),
        "tuning_records": len(tuning_records),
        "all_runs_path": all_runs_path,
        "best_configs_path": best_configs_path,
    }


def append_experiment_record(record: Dict[str, Any], *, reports_dir: str = REPORTS_DIR) -> None:
    os.makedirs(reports_dir, exist_ok=True)

    jsonl_path = os.path.join(reports_dir, os.path.basename(EXPERIMENT_LOG_JSONL))
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    csv_path = os.path.join(reports_dir, os.path.basename(EXPERIMENT_LOG_CSV))
    row = _flatten_record_for_csv(record)
    needs_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)

    generate_tuning_reports(reports_dir=reports_dir)


def write_model_comparison_record(
    records: Iterable[Dict[str, Any]],
    *,
    reports_dir: str = REPORTS_DIR,
    file_stem: str = "model_comparison_record",
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "generated_at_utc": _utc_now(),
        "summary": deepcopy(summary or {}),
        "records": list(records),
    }
    json_path = os.path.join(reports_dir, f"{file_stem}.json")
    csv_path = os.path.join(reports_dir, f"{file_stem}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    rows = [_flatten_record_for_csv(record) for record in payload["records"]]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return payload
