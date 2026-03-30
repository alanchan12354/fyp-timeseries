import argparse
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

from .config import BATCH_SIZE, EPOCHS, HORIZON, LR, RANDOM_SEED, SCHEDULER_TYPE, SEQ_LEN, TARGET_MODE, TARGET_SMOOTH_WINDOW


SANITY_SINE_PROFILE_NAME = "sanity_sine"
SANITY_SINE_PROFILE = {
    "horizon": 1,
    "target_mode": "next_return",
    "recurrent_hidden_size": 64,
    "transformer_d_model": 64,
    "epochs": 80,
    "scheduler_type": "none",
}


@dataclass
class RuntimeTrainingConfig:
    learning_rate: float = LR
    batch_size: int = BATCH_SIZE
    recurrent_hidden_size: int = 64
    recurrent_layer_count: int = 2
    transformer_d_model: int = 64
    transformer_num_layers: int = 2
    transformer_nhead: int = 4
    input_size: int = 8
    seq_len: int = SEQ_LEN
    horizon: int = HORIZON
    data_source: str = "spy"
    target_mode: str = TARGET_MODE
    target_smooth_window: int = TARGET_SMOOTH_WINDOW
    task_id: Optional[str] = None
    epochs: int = EPOCHS
    scheduler_type: str = SCHEDULER_TYPE
    random_seed: int = RANDOM_SEED
    run_note: Optional[str] = None

    @classmethod
    def from_sources(
        cls,
        config: Optional["RuntimeTrainingConfig"] = None,
        config_dict: Optional[Mapping[str, Any]] = None,
        cli_args: Optional[argparse.Namespace] = None,
        **overrides: Any,
    ) -> "RuntimeTrainingConfig":
        values = asdict(cls())
        if config is not None:
            values.update(asdict(config))
        if config_dict is not None:
            values.update(_normalize_config_keys(config_dict))
        if cli_args is not None:
            values.update(_namespace_values(cli_args))
        values.update({key: value for key, value in overrides.items() if value is not None})
        values = _apply_named_profile(values)
        values["task_id"] = _resolve_task_id(values)
        return cls(**{key: value for key, value in values.items() if key in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def recurrent_model_kwargs(self) -> dict[str, Any]:
        return {
            "hidden": self.recurrent_hidden_size,
            "layers": self.recurrent_layer_count,
            "input_size": self.input_size,
        }

    def transformer_model_kwargs(self) -> dict[str, Any]:
        return {
            "d_model": self.transformer_d_model,
            "nhead": self.transformer_nhead,
            "num_layers": self.transformer_num_layers,
            "dropout": 0.1,
            "input_size": self.input_size,
        }

    def training_metadata(self) -> dict[str, Any]:
        metadata = {
            "lr": self.learning_rate,
            "batch_size": self.batch_size,
            "horizon": self.horizon,
            "data_source": self.data_source,
            "target_mode": self.target_mode,
            "target_smooth_window": self.target_smooth_window,
            "task_id": self.task_id,
            "epochs": self.epochs,
            "scheduler_type": self.scheduler_type,
            "random_seed": self.random_seed,
            "input_size": self.input_size,
            "seq_len": self.seq_len,
        }
        if self.run_note:
            metadata["run_note"] = self.run_note
        return metadata


def add_runtime_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--learning-rate", type=float, dest="learning_rate", help="Optimizer learning rate.")
    parser.add_argument("--batch-size", type=int, dest="batch_size", help="Training batch size.")
    parser.add_argument(
        "--recurrent-hidden-size",
        type=int,
        dest="recurrent_hidden_size",
        help="Hidden size for recurrent models.",
    )
    parser.add_argument(
        "--recurrent-layer-count",
        type=int,
        dest="recurrent_layer_count",
        help="Layer count for recurrent models.",
    )
    parser.add_argument("--d-model", type=int, dest="transformer_d_model", help="Transformer model width.")
    parser.add_argument(
        "--transformer-num-layers",
        type=int,
        dest="transformer_num_layers",
        help="Number of Transformer encoder layers.",
    )
    parser.add_argument("--nhead", type=int, dest="transformer_nhead", help="Transformer attention head count.")
    parser.add_argument("--input-size", type=int, dest="input_size", help="Number of input channels per sequence step.")
    parser.add_argument("--seq-len", type=int, dest="seq_len", help="Input lookback window length for sequence models.")
    parser.add_argument("--horizon", type=int, dest="horizon", help="Target horizon for horizon_return mode.")
    parser.add_argument(
        "--data-source",
        choices=["spy", "sine"],
        dest="data_source",
        help="Input data source for sequence features.",
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
        dest="target_mode",
        help="Training target definition.",
    )
    parser.add_argument(
        "--target-smooth-window",
        type=int,
        dest="target_smooth_window",
        help="Forward window used by rolling target modes (next_mean_return / next_volatility / next3_mean_return).",
    )
    parser.add_argument(
        "--task-id",
        dest="task_id",
        help="Strongly encouraged stable identifier for the forecast task. If omitted, a deterministic value is generated.",
    )
    parser.add_argument("--epochs", type=int, dest="epochs", help="Maximum training epochs.")
    parser.add_argument(
        "--scheduler-type",
        choices=["none", "plateau", "cosine"],
        dest="scheduler_type",
        help="Learning-rate scheduler strategy.",
    )
    parser.add_argument("--random-seed", type=int, dest="random_seed", help="Random seed for deterministic reruns.")
    parser.add_argument("--run-note", dest="run_note", help="Optional experiment tag or note.")
    return parser


_ALIAS_MAP = {
    "lr": "learning_rate",
    "learning_rate": "learning_rate",
    "batch_size": "batch_size",
    "hidden": "recurrent_hidden_size",
    "hidden_size": "recurrent_hidden_size",
    "recurrent_hidden_size": "recurrent_hidden_size",
    "layers": "recurrent_layer_count",
    "num_layers": "transformer_num_layers",
    "layer_count": "recurrent_layer_count",
    "recurrent_layer_count": "recurrent_layer_count",
    "d_model": "transformer_d_model",
    "transformer_d_model": "transformer_d_model",
    "nhead": "transformer_nhead",
    "transformer_nhead": "transformer_nhead",
    "transformer_num_layers": "transformer_num_layers",
    "input_size": "input_size",
    "seq_len": "seq_len",
    "horizon": "horizon",
    "data_source": "data_source",
    "target_mode": "target_mode",
    "target_smooth_window": "target_smooth_window",
    "task_id": "task_id",
    "epochs": "epochs",
    "scheduler_type": "scheduler_type",
    "random_seed": "random_seed",
    "run_note": "run_note",
    "experiment_tag": "run_note",
    "experiment_note": "run_note",
}


def _normalize_config_keys(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in config.items():
        target_key = _ALIAS_MAP.get(key)
        if target_key and value is not None:
            normalized[target_key] = value
    return normalized


def _namespace_values(namespace: argparse.Namespace) -> dict[str, Any]:
    if namespace is None:
        return {}
    return {key: value for key, value in vars(namespace).items() if value is not None}


def _apply_named_profile(values: dict[str, Any]) -> dict[str, Any]:
    profile_name = str(values.get("run_note") or "").strip().lower()
    if profile_name == SANITY_SINE_PROFILE_NAME:
        values.update(SANITY_SINE_PROFILE)
    return values


def _resolve_task_id(values: Mapping[str, Any]) -> str:
    explicit_task_id = str(values.get("task_id") or "").strip()
    if explicit_task_id:
        return explicit_task_id

    data_source = str(values.get("data_source") or "spy").strip().lower()
    horizon = int(values.get("horizon", HORIZON))
    target_mode = str(values.get("target_mode") or TARGET_MODE).strip().lower()
    target_smooth_window = int(values.get("target_smooth_window", TARGET_SMOOTH_WINDOW))
    return f"{data_source}_h{horizon}_{target_mode}_sw{target_smooth_window}"
