from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .config import HORIZON, SEQ_LEN, TRAIN_RATIO, VAL_RATIO, TICKER, START
from .data import SeqDataset, build_sequences, chronological_split, load_data
from .reporting import (
    create_run_context,
    default_task_metadata,
    default_training_metadata,
    split_metadata,
)


@dataclass(frozen=True)
class PreparedSequenceRun:
    raw_returns: pd.Series
    sequences: np.ndarray
    targets: np.ndarray
    sequence_index: pd.DatetimeIndex
    train_data: tuple[np.ndarray, np.ndarray]
    val_data: tuple[np.ndarray, np.ndarray]
    test_data: tuple[np.ndarray, np.ndarray]
    train_index: pd.DatetimeIndex
    val_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex
    scaler: StandardScaler
    train_loader: DataLoader
    val_loader: DataLoader
    split_metadata: dict[str, Any]
    run_context: dict[str, Any]


def prepare_sequence_experiment_run(
    *,
    batch_size: int,
    experiment_name: str,
    run_note: str | None = None,
    training_metadata: dict[str, Any] | None = None,
    ticker: str | None = None,
    start: str | None = None,
    seq_len: int = SEQ_LEN,
    horizon: int = HORIZON,
    target_mode: str = "horizon_return",
    target_smooth_window: int = 3,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> PreparedSequenceRun:
    raw_returns = load_data(
    ticker=ticker or TICKER,
    start=start or START,
    )
    sequences, targets, sequence_index = build_sequences(
        raw_returns,
        seq_len,
        horizon,
        target_mode=target_mode,
        smooth_window=target_smooth_window,
    )
    (X_tr, y_tr, idx_tr), (X_va, y_va, idx_va), (X_te, y_te, idx_te) = chronological_split(
        sequences,
        targets,
        sequence_index,
        train_ratio,
        val_ratio,
    )

    scaler = StandardScaler()
    _, _, feature_count = X_tr.shape
    X_tr_scaled = scaler.fit_transform(X_tr.reshape(-1, feature_count)).reshape(X_tr.shape)
    X_va_scaled = scaler.transform(X_va.reshape(-1, feature_count)).reshape(X_va.shape)
    X_te_scaled = scaler.transform(X_te.reshape(-1, feature_count)).reshape(X_te.shape)

    train_loader = DataLoader(SeqDataset(X_tr_scaled, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(X_va_scaled, y_va), batch_size=batch_size)

    split_meta = split_metadata(len(X_tr_scaled), len(X_va_scaled), len(X_te_scaled))
    run_context = create_run_context(
        experiment_name,
        split_meta,
        task_meta=default_task_metadata(
            horizon=horizon,
            input_window=seq_len,
            target_mode=target_mode,
            target_smooth_window=target_smooth_window,
            ticker=ticker or TICKER,
            start_date=start or START,
        ),
        training_meta=default_training_metadata(**(training_metadata or {})),
        notes=run_note,
    )

    return PreparedSequenceRun(
        raw_returns=raw_returns,
        sequences=sequences,
        targets=targets,
        sequence_index=sequence_index,
        train_data=(X_tr_scaled, y_tr),
        val_data=(X_va_scaled, y_va),
        test_data=(X_te_scaled, y_te),
        train_index=idx_tr,
        val_index=idx_va,
        test_index=idx_te,
        scaler=scaler,
        train_loader=train_loader,
        val_loader=val_loader,
        split_metadata=split_meta,
        run_context=run_context,
    )
