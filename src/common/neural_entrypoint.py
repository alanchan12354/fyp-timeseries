import argparse
from typing import Any, Callable, Optional

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .config import HORIZON, SEQ_LEN, TRAIN_RATIO, VAL_RATIO
from .data import SeqDataset, build_sequences, chronological_split, load_data
from .reporting import split_metadata
from .runtime_config import RuntimeTrainingConfig, add_runtime_config_args


def build_runtime_parser(model_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run the {model_name} training experiment.")
    add_runtime_config_args(parser)
    return parser


def resolve_runtime_config(
    *,
    config: Optional[RuntimeTrainingConfig] = None,
    config_dict: Optional[dict[str, Any]] = None,
    cli_args: Optional[argparse.Namespace] = None,
    **overrides: Any,
) -> RuntimeTrainingConfig:
    return RuntimeTrainingConfig.from_sources(
        config=config,
        config_dict=config_dict,
        cli_args=cli_args,
        **overrides,
    )


def prepare_sequence_training_data(batch_size: int):
    returns = load_data()
    X, y, idx = build_sequences(returns, SEQ_LEN, HORIZON)
    (X_tr, y_tr, _), (X_va, y_va, _), (X_te, y_te, idx_te) = chronological_split(
        X,
        y,
        idx,
        TRAIN_RATIO,
        VAL_RATIO,
    )

    scaler = StandardScaler()
    _, _, d_features = X_tr.shape
    X_tr_s = scaler.fit_transform(X_tr.reshape(-1, d_features)).reshape(X_tr.shape)
    X_va_s = scaler.transform(X_va.reshape(-1, d_features)).reshape(X_va.shape)
    X_te_s = scaler.transform(X_te.reshape(-1, d_features)).reshape(X_te.shape)

    tr_load = DataLoader(SeqDataset(X_tr_s, y_tr), batch_size=batch_size, shuffle=True)
    va_load = DataLoader(SeqDataset(X_va_s, y_va), batch_size=batch_size)
    split_meta = split_metadata(len(X_tr), len(X_va), len(X_te))
    return tr_load, va_load, (X_te_s, y_te), idx_te, split_meta
