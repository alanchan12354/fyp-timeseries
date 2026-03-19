import csv
import json
import os
import platform
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from importlib.metadata import PackageNotFoundError, version

import torch

from .config import (
    BASE_DIR,
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
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


SUMMARY_FIELDNAMES = [
    "timestamp_utc",
    "run_id",
    "model_name",
    "hidden_size",
    "num_layers",
    "lr",
    "batch_size",
    "best_val_MSE",
    "MSE",
    "MAE",
    "DA",
    "best_epoch",
    "notes",
]


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


def _flatten_record_for_csv(record: Dict[str, Any]) -> Dict[str, Any]:
    training = record.get("training", {})
    metrics = record.get("metrics", {})
    hyper = record.get("hyperparameters", {})
    tuning = record.get("tuning", {})
    return {
        "timestamp_utc": record.get("timestamp_utc"),
        "run_id": record.get("run_id"),
        "model_name": record.get("model_name"),
        "hidden_size": hyper.get("hidden"),
        "num_layers": hyper.get("layers"),
        "lr": training.get("lr"),
        "batch_size": training.get("batch_size"),
        "best_val_MSE": metrics.get("best_val_MSE"),
        "MSE": metrics.get("MSE"),
        "MAE": metrics.get("MAE"),
        "DA": metrics.get("DA"),
        "best_epoch": tuning.get("best_epoch"),
        "notes": record.get("notes"),
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
