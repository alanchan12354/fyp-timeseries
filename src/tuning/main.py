import argparse
import csv
import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

from src.common.config import BATCH_SIZE, LR, REPORTS_DIR, SEQ_LEN
from src.common.runtime_config import RuntimeTrainingConfig
from src.common.reporting import format_structured_notes, reset_tuning_artifacts
from src.baseline_lr import train as baseline_lr_train
from src.gru import train as gru_train
from src.lstm import train as lstm_train
from src.rnn import train as rnn_train
from src.transformer import train as transformer_train


TUNING_RUNS_CSV = os.path.join(REPORTS_DIR, "tuning_runs.csv")
TUNING_WINNERS_CSV = os.path.join(REPORTS_DIR, "tuning_winners.csv")

RECURRENT_MODELS = {"lstm", "gru", "rnn"}
DEFAULT_MODEL_ORDER = ["lstm", "gru", "rnn", "transformer", "baseline_lr"]
RUNS_FIELDNAMES = [
    "task_id",
    "model",
    "stage_index",
    "stage_name",
    "parameter_group",
    "candidate_index",
    "note",
    "best_val_MSE",
    "selected",
    "config_json",
]
WINNERS_FIELDNAMES = [
    "task_id",
    "model",
    "stage_index",
    "stage_name",
    "parameter_group",
    "winning_value_json",
    "best_val_MSE",
    "frozen_config_json",
]


@dataclass(frozen=True)
class ModelSpec:
    cli_name: str
    display_name: str
    train_entrypoint: Callable[..., Dict[str, Any]]
    baseline: Dict[str, Any]
    stage_order: List[str]
    default_plan: Dict[str, List[Any]]


def _recurrent_baseline() -> Dict[str, Any]:
    return {
        "lr": LR,
        "hidden": 64,
        "layers": 2,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
    }


MODEL_SPECS: Dict[str, ModelSpec] = {
    "lstm": ModelSpec(
        cli_name="lstm",
        display_name="LSTM",
        train_entrypoint=lstm_train.main,
        baseline=_recurrent_baseline(),
        stage_order=["lr", "hidden", "layers", "batch_size", "seq_len"],
        default_plan={
            "lr": [1e-3, 5e-4, 1e-4],
            "hidden": [32, 64, 128],
            "layers": [1, 2, 3],
            "batch_size": [32, 64, 128],
            "seq_len": [20, 30, 60],
        },
    ),
    "gru": ModelSpec(
        cli_name="gru",
        display_name="GRU",
        train_entrypoint=gru_train.main,
        baseline=_recurrent_baseline(),
        stage_order=["lr", "hidden", "layers", "batch_size", "seq_len"],
        default_plan={
            "lr": [1e-3, 5e-4, 1e-4],
            "hidden": [32, 64, 128],
            "layers": [1, 2, 3],
            "batch_size": [32, 64, 128],
            "seq_len": [20, 30, 60],
        },
    ),
    "rnn": ModelSpec(
        cli_name="rnn",
        display_name="RNN",
        train_entrypoint=rnn_train.main,
        baseline=_recurrent_baseline(),
        stage_order=["lr", "hidden", "layers", "batch_size", "seq_len"],
        default_plan={
            "lr": [1e-3, 5e-4, 1e-4],
            "hidden": [32, 64, 128],
            "layers": [1, 2, 3],
            "batch_size": [32, 64, 128],
            "seq_len": [20, 30, 60],
        },
    ),
    "transformer": ModelSpec(
        cli_name="transformer",
        display_name="Transformer",
        train_entrypoint=transformer_train.main,
        baseline={
            "lr": LR,
            "d_model": 64,
            "num_layers": 2,
            "nhead": 4,
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
        },
        stage_order=["lr", "d_model", "num_layers", "nhead", "batch_size", "seq_len"],
        default_plan={
            "lr": [1e-3, 5e-4, 1e-4],
            "d_model": [32, 64, 128],
            "num_layers": [1, 2, 3],
            "nhead": [2, 4, 8],
            "batch_size": [32, 64, 128],
            "seq_len": [20, 30, 60],
        },
    ),
    "baseline_lr": ModelSpec(
        cli_name="baseline_lr",
        display_name="Baseline-LR",
        train_entrypoint=baseline_lr_train.main,
        baseline={
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
        },
        stage_order=["seq_len"],
        default_plan={
            "seq_len": [20, 30, 60],
        },
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sequential model tuning workflow runner.")
    parser.add_argument(
        "--model",
        choices=DEFAULT_MODEL_ORDER + ["all"],
        default="all",
        help="Tune a single model or all supported models.",
    )
    parser.add_argument(
        "--plan-file",
        help="Path to a JSON file with per-model tuning plans.",
    )
    parser.add_argument(
        "--plan-json",
        help="Inline JSON object with per-model tuning plans.",
    )
    parser.add_argument(
        "--session-mode",
        choices=["reset", "append"],
        default="append",
        help="Choose whether to clear prior tuning artifacts before running or append to existing history.",
    )
    parser.add_argument(
        "--clear-outputs",
        action="store_true",
        help="Deprecated alias for --session-mode reset.",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Deprecated alias for --session-mode append.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved tuning plan without starting training.",
    )
    parser.add_argument("--horizon", type=int, help="Task horizon passed to each tuned training run.")
    parser.add_argument(
        "--data-source",
        choices=["spy", "sine"],
        help="Dataset source passed to each tuned training run.",
    )
    parser.add_argument(
        "--target-mode",
        choices=[
            "horizon_return",
            "next_return",
            "next3_mean_return",
            "next_mean_return",
            "next_volatility",
            "sine_next_day",
        ],
        help="Target definition passed to each tuned training run.",
    )
    parser.add_argument(
        "--target-smooth-window",
        type=int,
        help="Forward window passed to rolling target modes during tuning.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        help="Input lookback window passed to each tuned training run.",
    )
    parser.add_argument(
        "--task-id",
        help="Stable task identifier to persist on all tuning rows for this run.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Epoch budget override passed to each tuned training run.",
    )
    parser.add_argument(
        "--scheduler-type",
        choices=["none", "plateau", "cosine"],
        help="Scheduler override passed to each tuned training run.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Random seed passed to each tuned training run for reproducibility.",
    )
    return parser


def _load_plan_overrides(args: argparse.Namespace) -> Dict[str, Dict[str, List[Any]]]:
    payload: Dict[str, Any] = {}
    if args.plan_file:
        with open(args.plan_file, "r", encoding="utf-8") as handle:
            payload.update(json.load(handle))
    if args.plan_json:
        payload.update(json.loads(args.plan_json))
    return _normalize_plan_payload(payload)


def _normalize_plan_payload(payload: Mapping[str, Any]) -> Dict[str, Dict[str, List[Any]]]:
    normalized: Dict[str, Dict[str, List[Any]]] = {}
    for raw_model, stage_map in payload.items():
        model_name = str(raw_model).strip().lower()
        if model_name not in MODEL_SPECS:
            raise ValueError(f"Unsupported model in tuning plan: {raw_model}")
        if not isinstance(stage_map, Mapping):
            raise ValueError(f"Tuning plan for {raw_model} must be a JSON object.")
        normalized[model_name] = {}
        for raw_stage, candidates in stage_map.items():
            stage_name = str(raw_stage).strip()
            if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
                raise ValueError(
                    f"Plan stage '{raw_stage}' for model '{raw_model}' must be a JSON array of candidates."
                )
            normalized[model_name][stage_name] = list(candidates)
    return normalized


def _resolve_models(model_arg: str) -> List[str]:
    if model_arg == "all":
        return DEFAULT_MODEL_ORDER.copy()
    return [model_arg]


def _merge_plan(spec: ModelSpec, overrides: Mapping[str, List[Any]] | None) -> Dict[str, List[Any]]:
    plan = deepcopy(spec.default_plan)
    for stage_name, candidates in (overrides or {}).items():
        if stage_name not in spec.stage_order:
            raise ValueError(f"Unsupported stage '{stage_name}' for model '{spec.cli_name}'.")
        if not candidates:
            raise ValueError(f"Stage '{stage_name}' for model '{spec.cli_name}' must have at least one candidate.")
        plan[stage_name] = list(candidates)
    return plan


def _stage_display_name(spec: ModelSpec, stage_name: str) -> str:
    return f"{spec.display_name} {stage_name} sweep"


def _config_to_runtime_dict(
    model_name: str,
    frozen_config: Mapping[str, Any],
    note: str,
    *,
    runtime_overrides: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    runtime = RuntimeTrainingConfig.from_sources(
        config_dict=runtime_overrides or {},
        run_note=note,
    ).to_dict()
    runtime["learning_rate"] = frozen_config["lr"]
    runtime["batch_size"] = frozen_config["batch_size"]
    runtime["seq_len"] = frozen_config["seq_len"]
    if model_name in RECURRENT_MODELS:
        runtime["recurrent_hidden_size"] = frozen_config["hidden"]
        runtime["recurrent_layer_count"] = frozen_config["layers"]
    else:
        runtime["transformer_d_model"] = frozen_config["d_model"]
        runtime["transformer_num_layers"] = frozen_config["num_layers"]
        runtime["transformer_nhead"] = frozen_config["nhead"]
    return runtime


def _note_alias_map(model_name: str) -> Dict[str, str]:
    if model_name == "baseline_lr":
        return {
            "lr": "lr",
            "batch_size": "batch",
            "seq_len": "seq_len",
        }
    if model_name in RECURRENT_MODELS:
        return {
            "hidden": "hidden",
            "layers": "layers",
            "lr": "lr",
            "batch_size": "batch",
            "seq_len": "seq_len",
        }
    return {
        "d_model": "d_model",
        "num_layers": "layers",
        "nhead": "nhead",
        "lr": "lr",
        "batch_size": "batch",
        "seq_len": "seq_len",
    }


def _structured_note_metadata(
    spec: ModelSpec,
    stage_name: str,
    *,
    candidates: Sequence[Any],
    candidate_config: Mapping[str, Any],
    selection_metric: str = "best_val_MSE",
) -> Dict[str, Any]:
    alias_map = _note_alias_map(spec.cli_name)
    fixed_baseline = {
        alias_map[key]: candidate_config[key]
        for key in spec.baseline
        if key != stage_name and key in candidate_config and key in alias_map
    }
    metadata: Dict[str, Any] = {
        "model": spec.display_name,
        "stage": f"{stage_name}_sweep",
        "candidate_param": alias_map.get(stage_name, stage_name),
        "candidates": list(candidates),
        "fixed": fixed_baseline,
        "selection": selection_metric,
    }
    for key, alias in alias_map.items():
        if key in candidate_config:
            metadata[alias] = candidate_config[key]
    return metadata


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _resolve_session_mode(args: argparse.Namespace) -> str:
    if args.clear_outputs and args.keep_outputs:
        raise ValueError("Choose either --clear-outputs or --keep-outputs, not both.")
    if args.clear_outputs:
        return "reset"
    if args.keep_outputs:
        return "append"
    return args.session_mode


def _print_reset_summary(summary: Mapping[str, Any]) -> None:
    removed_files = list(summary.get("removed_files", []))
    reports_dir = str(summary.get("reports_dir", REPORTS_DIR))
    print("Artifact reset summary:")
    if removed_files:
        for path in removed_files:
            print(f"- removed {os.path.relpath(path, reports_dir)}")
    else:
        print("- no prior tuning artifacts found under reports/")
    print(f"Removed {summary.get('removed_count', 0)} file(s) from {reports_dir}")


def _append_csv(path: str, fieldnames: List[str], row: Dict[str, Any]) -> None:
    _ensure_parent(path)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _print_resolved_plan(models: Iterable[str], plan_by_model: Mapping[str, Dict[str, List[Any]]]) -> None:
    print("Resolved tuning workflow plan:")
    for model_name in models:
        spec = MODEL_SPECS[model_name]
        print(f"- {spec.display_name}")
        print(f"  baseline: {json.dumps(spec.baseline, sort_keys=True)}")
        for stage_name in spec.stage_order:
            print(f"  {stage_name}: {plan_by_model[model_name][stage_name]}")


def tune_model(
    model_name: str,
    plan: Mapping[str, List[Any]],
    *,
    dry_run: bool = False,
    runtime_overrides: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    spec = MODEL_SPECS[model_name]
    frozen_config = deepcopy(spec.baseline)
    winners: List[Dict[str, Any]] = []
    model_start = time.perf_counter()

    for stage_index, stage_name in enumerate(spec.stage_order, start=1):
        candidates = list(plan[stage_name])
        if dry_run:
            winners.append(
                {
                    "model": model_name,
                    "stage_index": stage_index,
                    "stage_name": _stage_display_name(spec, stage_name),
                    "parameter_group": stage_name,
                    "winning_value": frozen_config.get(stage_name),
                    "best_val_MSE": None,
                    "frozen_config": deepcopy(frozen_config),
                }
            )
            continue

        stage_results: List[Dict[str, Any]] = []
        for candidate_index, candidate_value in enumerate(candidates, start=1):
            candidate_config = deepcopy(frozen_config)
            candidate_config[stage_name] = candidate_value
            note_metadata = _structured_note_metadata(
                spec,
                stage_name,
                candidates=candidates,
                candidate_config=candidate_config,
            )
            note = format_structured_notes(note_metadata)
            runtime_config = _config_to_runtime_dict(
                model_name,
                candidate_config,
                note,
                runtime_overrides=runtime_overrides,
            )
            metrics = spec.train_entrypoint(config_dict=runtime_config)
            best_val = float(metrics["best_val_MSE"])
            stage_results.append({
                "task_id": runtime_config["task_id"],
                "model": model_name,
                "stage_index": stage_index,
                "stage_name": _stage_display_name(spec, stage_name),
                "parameter_group": stage_name,
                "candidate_index": candidate_index,
                "note": note,
                "best_val_MSE": best_val,
                "selected": False,
                "config_json": json.dumps(candidate_config, sort_keys=True),
                "candidate_value": candidate_value,
                "candidate_config": candidate_config,
            })

        winner = min(stage_results, key=lambda row: row["best_val_MSE"])
        for row in stage_results:
            row["selected"] = row is winner
            _append_csv(TUNING_RUNS_CSV, RUNS_FIELDNAMES, {key: row[key] for key in RUNS_FIELDNAMES})
        frozen_config[stage_name] = winner["candidate_value"]

        winner_row = {
            "task_id": winner["task_id"],
            "model": model_name,
            "stage_index": stage_index,
            "stage_name": _stage_display_name(spec, stage_name),
            "parameter_group": stage_name,
            "winning_value_json": json.dumps(winner["candidate_value"]),
            "best_val_MSE": winner["best_val_MSE"],
            "frozen_config_json": json.dumps(frozen_config, sort_keys=True),
        }
        _append_csv(TUNING_WINNERS_CSV, WINNERS_FIELDNAMES, winner_row)
        winners.append({
            "model": model_name,
            "stage_index": stage_index,
            "stage_name": _stage_display_name(spec, stage_name),
            "parameter_group": stage_name,
            "winning_value": winner["candidate_value"],
            "best_val_MSE": winner["best_val_MSE"],
            "frozen_config": deepcopy(frozen_config),
        })

    return {
        "model": model_name,
        "final_config": frozen_config,
        "winners": winners,
        "elapsed_seconds": round(time.perf_counter() - model_start, 3),
    }


def main(cli_args: argparse.Namespace | None = None) -> List[Dict[str, Any]]:
    workflow_start = time.perf_counter()
    args = cli_args or build_parser().parse_args()
    session_mode = _resolve_session_mode(args)

    selected_models = _resolve_models(args.model)
    plan_overrides = _load_plan_overrides(args)
    plan_by_model = {
        model_name: _merge_plan(MODEL_SPECS[model_name], plan_overrides.get(model_name))
        for model_name in selected_models
    }
    runtime_overrides = {
        key: value
        for key, value in {
            "horizon": args.horizon,
            "data_source": args.data_source,
            "target_mode": args.target_mode,
            "target_smooth_window": args.target_smooth_window,
            "task_id": args.task_id,
            "seq_len": args.seq_len,
            "epochs": args.epochs,
            "scheduler_type": args.scheduler_type,
            "random_seed": args.random_seed,
        }.items()
        if value is not None
    }

    _print_resolved_plan(selected_models, plan_by_model)
    print(f"Session mode: {session_mode}")
    print(f"Reports directory: {REPORTS_DIR}")
    if args.dry_run:
        print("Dry run enabled; no training jobs were started.")
        return [
            {
                "model": model_name,
                "final_config": deepcopy(MODEL_SPECS[model_name].baseline),
                "winners": [],
            }
            for model_name in selected_models
        ]

    if session_mode == "reset":
        _print_reset_summary(reset_tuning_artifacts())
    else:
        print("Append mode selected; keeping prior reports/ artifacts and extending history.")

    summaries = [
        tune_model(
            model_name,
            plan_by_model[model_name],
            runtime_overrides=runtime_overrides,
        )
        for model_name in selected_models
    ]
    print("Completed tuning workflow.")
    elapsed_seconds = round(time.perf_counter() - workflow_start, 3)
    print(f"Total tuning workflow time: {elapsed_seconds:.3f}s")
    print(json.dumps(summaries, indent=2))
    return summaries


if __name__ == "__main__":
    main()
