from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.config import REPORTS_DIR

CANONICAL_SOURCE = "tuning_winners"
SOURCE_FILES = {
    "tuning_winners": Path(REPORTS_DIR) / "tuning_winners.csv",
    "tuning_best_configs": Path(REPORTS_DIR) / "tuning_best_configs.csv",
}
MODEL_ORDER = ("lstm", "gru", "rnn", "transformer")
REQUIRED_KEYS = {
    "lstm": {"lr", "batch_size", "hidden", "layers"},
    "gru": {"lr", "batch_size", "hidden", "layers"},
    "rnn": {"lr", "batch_size", "hidden", "layers"},
    "transformer": {"lr", "batch_size", "d_model", "num_layers", "nhead"},
}


@dataclass(frozen=True)
class BestConfigSelection:
    source_key: str
    source_path: Path
    configs: dict[str, dict[str, Any]]
    source_rows: dict[str, dict[str, str]]
    note: str


class BestConfigError(ValueError):
    pass



def load_best_configs(source_key: str = CANONICAL_SOURCE, *, source_path: str | Path | None = None) -> BestConfigSelection:
    normalized_source = source_key.strip().lower()
    if normalized_source not in SOURCE_FILES:
        raise BestConfigError(
            f"Unsupported config source '{source_key}'. Expected one of: {', '.join(sorted(SOURCE_FILES))}."
        )

    resolved_path = Path(source_path) if source_path is not None else SOURCE_FILES[normalized_source]
    if not resolved_path.exists():
        raise BestConfigError(f"Tuning config source not found: {resolved_path}")

    if normalized_source == "tuning_winners":
        configs, source_rows = _load_from_tuning_winners(resolved_path)
        note = "Uses the final frozen staged winners from sequential tuning for each model."
    else:
        configs, source_rows = _load_from_tuning_best_configs(resolved_path)
        note = "Uses the single best archived run per model across the tuning log."

    missing_models = [model for model in MODEL_ORDER if model not in configs]
    if missing_models:
        raise BestConfigError(
            f"Missing tuned configs for model(s): {', '.join(missing_models)} in {resolved_path}"
        )

    return BestConfigSelection(
        source_key=normalized_source,
        source_path=resolved_path,
        configs=configs,
        source_rows=source_rows,
        note=note,
    )



def _load_from_tuning_winners(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    latest_rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = _normalize_model_name(row.get("model"))
            stage_index = _coerce_int(row.get("stage_index"), field_name="stage_index", model=model)
            prior_stage = _coerce_int(latest_rows[model].get("stage_index"), field_name="stage_index", model=model) if model in latest_rows else -1
            if stage_index >= prior_stage:
                latest_rows[model] = row

    configs = {
        model: _validate_required_keys(model, _coerce_types(_load_json_field(row, "frozen_config_json", model=model)), path)
        for model, row in latest_rows.items()
    }
    return configs, latest_rows



def _load_from_tuning_best_configs(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    rows_by_model: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = _normalize_model_name(row.get("model"))
            best_val = _coerce_float(row.get("best_val_MSE"), field_name="best_val_MSE", model=model)
            current = rows_by_model.get(model)
            if current is None or best_val < _coerce_float(current.get("best_val_MSE"), field_name="best_val_MSE", model=model):
                rows_by_model[model] = row

    configs = {
        model: _validate_required_keys(model, _config_from_best_configs_row(row, path), path)
        for model, row in rows_by_model.items()
    }
    return configs, rows_by_model



def _config_from_best_configs_row(row: dict[str, str], path: Path) -> dict[str, Any]:
    model = _normalize_model_name(row.get("model"))
    hyperparameters = _coerce_types(_load_json_field(row, "hyperparameters", model=model))
    notes = _parse_structured_notes(row.get("notes", ""))

    if "lr" not in hyperparameters and "lr" in notes:
        hyperparameters["lr"] = _coerce_scalar(notes["lr"])
    if "batch_size" not in hyperparameters:
        batch_note = notes.get("batch") or notes.get("batch_size")
        if batch_note is not None:
            hyperparameters["batch_size"] = _coerce_scalar(batch_note)

    if model == "transformer":
        _fill_if_present(hyperparameters, "d_model", row.get("d_model"))
        _fill_if_present(hyperparameters, "nhead", row.get("nhead"))
        _fill_if_present(hyperparameters, "num_layers", row.get("num_layers"))
    else:
        _fill_if_present(hyperparameters, "hidden", row.get("hidden_size"))
        _fill_if_present(hyperparameters, "layers", row.get("num_layers"))

    return hyperparameters



def _fill_if_present(config: dict[str, Any], key: str, raw_value: str | None) -> None:
    if key not in config and raw_value not in (None, ""):
        config[key] = _coerce_scalar(raw_value)



def _validate_required_keys(model: str, config: dict[str, Any], path: Path) -> dict[str, Any]:
    missing = sorted(REQUIRED_KEYS[model] - set(config))
    if missing:
        raise BestConfigError(
            f"Incomplete tuned config for model '{model}' in {path}: missing {', '.join(missing)}"
        )
    return {key: config[key] for key in sorted(config)}



def _normalize_model_name(raw_model: str | None) -> str:
    if not raw_model:
        raise BestConfigError("Encountered tuning row without a model name.")
    normalized = raw_model.strip().lower()
    if normalized not in REQUIRED_KEYS:
        raise BestConfigError(f"Unsupported model name '{raw_model}'.")
    return normalized



def _load_json_field(row: dict[str, str], field_name: str, *, model: str) -> dict[str, Any]:
    payload = row.get(field_name)
    if not payload:
        raise BestConfigError(f"Missing {field_name} for model '{model}'.")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise BestConfigError(f"Invalid JSON in {field_name} for model '{model}': {exc}") from exc
    if not isinstance(data, dict):
        raise BestConfigError(f"Expected {field_name} to decode to an object for model '{model}'.")
    return data



def _coerce_types(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: _coerce_scalar(value) for key, value in payload.items()}



def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return value
        try:
            numeric = float(text)
        except ValueError:
            return value
        return int(numeric) if numeric.is_integer() else numeric
    return value



def _coerce_int(value: Any, *, field_name: str, model: str) -> int:
    coerced = _coerce_scalar(value)
    if not isinstance(coerced, int):
        raise BestConfigError(f"Invalid {field_name} for model '{model}': {value}")
    return coerced



def _coerce_float(value: Any, *, field_name: str, model: str) -> float:
    coerced = _coerce_scalar(value)
    if not isinstance(coerced, (int, float)):
        raise BestConfigError(f"Invalid {field_name} for model '{model}': {value}")
    return float(coerced)



def _parse_structured_notes(notes: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in notes.split(";"):
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed
