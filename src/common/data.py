import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch.utils.data import Dataset
from .config import TICKER, START

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

def build_sequences(returns: pd.Series, seq_len: int, horizon: int = 1):
    """
    For Neural Models
    Input:  [r_{t-seq+1}, ..., r_t]
    Target: r_{t+horizon}
    """
    r = returns.values.astype(np.float64)
    dates = returns.index
    
    X_list, y_list, y_dates = [], [], []
    for t in range(seq_len - 1, len(r) - horizon):
        X_list.append(r[t - seq_len + 1: t + 1])
        y_list.append(r[t + horizon])
        y_dates.append(dates[t + horizon])
        
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
