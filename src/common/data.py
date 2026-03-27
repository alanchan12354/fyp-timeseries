import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch.utils.data import Dataset
from .config import START, TARGET_MODE, TARGET_SMOOTH_WINDOW, TICKER

def load_data(ticker=TICKER, start=START):
    df = yf.download(ticker, start=start, progress=False)
    if df.empty:
        raise RuntimeError("No data downloaded.")
    
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 1:
            close = close.iloc[:, 0]
        elif ticker in close.columns:
            close = close[ticker]
        else:
            close = close.iloc[:, 0]
            
    price = close.dropna()
    returns = np.log(price / price.shift(1)).dropna()
    return returns

def make_lag_features(returns: pd.Series, lags: int, horizon: int = 1):
    """
    For baseline models using lagged tabular features.

    returns: r_t
    target: y_t = r_{t+horizon}
    features at time t: [r_t, r_{t-1}, ..., r_{t-lags+1}]
    """
    df = pd.DataFrame({"r": returns})
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["r"].shift(i - 1)
    df["y"] = df["r"].shift(-horizon)
    df = df.dropna()
    
    X = df[[f"lag_{i}" for i in range(1, lags + 1)]].values
    y = df["y"].values
    return X, y, df.index

def build_sequences(
    returns: pd.Series,
    seq_len: int,
    horizon: int = 1,
    *,
    target_mode: str = TARGET_MODE,
    smooth_window: int = TARGET_SMOOTH_WINDOW,
):
    """
    For Neural Models
    Input:  [r_{t-seq+1}, ..., r_t]
    Target modes:
      - horizon_return: r_{t+horizon}
      - next_return: r_{t+1}
      - next3_mean_return: mean(r_{t+1}, ..., r_{t+smooth_window})
    """
    r = returns.values.astype(np.float64)
    dates = returns.index

    mode = (target_mode or "horizon_return").strip().lower()
    if mode == "horizon_return":
        offset = int(horizon)
        if offset < 1:
            raise ValueError("horizon must be >= 1 for horizon_return mode.")
    elif mode == "next_return":
        offset = 1
    elif mode == "next3_mean_return":
        offset = int(smooth_window)
        if offset < 1:
            raise ValueError("smooth_window must be >= 1 for next3_mean_return mode.")
    else:
        raise ValueError(
            f"Unsupported target_mode '{target_mode}'. "
            "Expected one of: horizon_return, next_return, next3_mean_return."
        )

    X_list, y_list, y_dates = [], [], []
    for t in range(seq_len - 1, len(r) - offset):
        X_list.append(r[t - seq_len + 1: t + 1])
        if mode == "next3_mean_return":
            forward_window = r[t + 1: t + offset + 1]
            y_list.append(float(np.mean(forward_window)))
        else:
            y_list.append(r[t + offset])
        y_dates.append(dates[t + offset])

    X = np.array(X_list)[:, :, None]  # (N, seq, 1)
    y = np.array(y_list)              # (N,)
    idx = pd.DatetimeIndex(y_dates)
    return X, y, idx

def chronological_split(X, y, idx, train_ratio, val_ratio):
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    return (
        (X[:n_train], y[:n_train], idx[:n_train]),
        (X[n_train:n_train+n_val], y[n_train:n_train+n_val], idx[n_train:n_train+n_val]),
        (X[n_train+n_val:], y[n_train+n_val:], idx[n_train+n_val:])
    )

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]
