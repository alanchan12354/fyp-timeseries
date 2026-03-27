import argparse
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

from .config import BATCH_SIZE, HORIZON, LR, TARGET_MODE, TARGET_SMOOTH_WINDOW


@dataclass
class RuntimeTrainingConfig:
    learning_rate: float = LR
    batch_size: int = BATCH_SIZE
    recurrent_hidden_size: int = 64
    recurrent_layer_count: int = 2
    transformer_d_model: int = 64
    transformer_num_layers: int = 2
    transformer_nhead: int = 4
    horizon: int = HORIZON
    target_mode: str = TARGET_MODE
    target_smooth_window: int = TARGET_SMOOTH_WINDOW
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
        return cls(**{key: value for key, value in values.items() if key in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def recurrent_model_kwargs(self) -> dict[str, Any]:
        return {
            "hidden": self.recurrent_hidden_size,
            "layers": self.recurrent_layer_count,
        }

    def transformer_model_kwargs(self) -> dict[str, Any]:
        return {
            "d_model": self.transformer_d_model,
            "nhead": self.transformer_nhead,
            "num_layers": self.transformer_num_layers,
            "dropout": 0.1,
        }

    def training_metadata(self) -> dict[str, Any]:
        metadata = {
            "lr": self.learning_rate,
            "batch_size": self.batch_size,
            "horizon": self.horizon,
            "target_mode": self.target_mode,
            "target_smooth_window": self.target_smooth_window,
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
    parser.add_argument("--horizon", type=int, dest="horizon", help="Target horizon for horizon_return mode.")
    parser.add_argument(
        "--target-mode",
        choices=["horizon_return", "next_return", "next3_mean_return"],
        dest="target_mode",
        help="Training target definition.",
    )
    parser.add_argument(
        "--target-smooth-window",
        type=int,
        dest="target_smooth_window",
        help="Forward averaging window used by next3_mean_return mode.",
    )
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
    "horizon": "horizon",
    "target_mode": "target_mode",
    "target_smooth_window": "target_smooth_window",
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
