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

def build_spy_feature_frame(ticker=TICKER, start=START):
    df = yf.download(ticker, start=start, progress=False)
    if df.empty:
        raise RuntimeError("No data downloaded.")

    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        elif ticker in df.columns.get_level_values(0):
            df = df.xs(ticker, axis=1, level=0)
        else:
            df.columns = df.columns.get_level_values(0)

    feature_source = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    feature_source = feature_source.apply(pd.to_numeric, errors="coerce")

    features = pd.DataFrame(index=feature_source.index)
    features["log_ret"] = np.log(feature_source["Close"] / feature_source["Close"].shift(1))
    features["oc_ret"] = np.log(feature_source["Close"] / feature_source["Open"])
    features["hl_range"] = (feature_source["High"] - feature_source["Low"]) / feature_source["Close"]
    features["vol_chg"] = np.log(feature_source["Volume"] / feature_source["Volume"].shift(1))
    features["ma_5_gap"] = (feature_source["Close"] / feature_source["Close"].rolling(5).mean()) - 1.0
    features["ma_20_gap"] = (feature_source["Close"] / feature_source["Close"].rolling(20).mean()) - 1.0
    features["volatility_5"] = features["log_ret"].rolling(5).std()
    features["volatility_20"] = features["log_ret"].rolling(20).std()

    return features.dropna()

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
    features: pd.DataFrame,
    seq_len: int,
    horizon: int = 1,
    *,
    target_mode: str = TARGET_MODE,
    smooth_window: int = TARGET_SMOOTH_WINDOW,
):
    """
    For Neural Models
    Input:  [[f1..f8]_{t-seq+1}, ..., [f1..f8]_t]
    Target modes:
      - horizon_return: r_{t+horizon}
      - next_return: r_{t+1}
      - next3_mean_return: mean(r_{t+1}, ..., r_{t+smooth_window})
    """
    if "log_ret" not in features.columns:
        raise ValueError("Expected features to include a 'log_ret' column for target construction.")

    feature_values = features.values.astype(np.float64)
    r = features["log_ret"].values.astype(np.float64)
    dates = features.index

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
        X_list.append(feature_values[t - seq_len + 1: t + 1, :])
        if mode == "next3_mean_return":
            forward_window = r[t + 1: t + offset + 1]
            y_list.append(float(np.mean(forward_window)))
        else:
            y_list.append(r[t + offset])
        y_dates.append(dates[t + offset])

    X = np.array(X_list)  # (N, seq, features)
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
