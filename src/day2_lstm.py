import os
import json
import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- config ----------
TICKER = "SPY"
START = "2010-01-01"
SEQ_LEN = 10

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
HIDDEN = 64
LAYERS = 1
PATIENCE = 6  # early stopping

OUT_DIR = "reports"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def directional_accuracy(y_true, y_pred):
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, seq, 1)
        self.y = torch.tensor(y, dtype=torch.float32)  # (N,)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]


class LSTMRegressor(nn.Module):
    def __init__(self, hidden=64, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, seq, 1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]          # (B, hidden)
        yhat = self.fc(last).squeeze(1)  # (B,)
        return yhat


def build_sequences(returns: pd.Series, seq_len: int):
    """
    Input:  [r_{t-seq+1}, ..., r_t]
    Target: r_{t+1}
    """
    r = returns.values.astype(np.float64)
    dates = returns.index

    X_list, y_list, y_dates = [], [], []
    for t in range(seq_len - 1, len(r) - 1):
        X_list.append(r[t - seq_len + 1: t + 1])
        y_list.append(r[t + 1])
        y_dates.append(dates[t + 1])

    X = np.array(X_list)[:, :, None]  # (N, seq, 1)
    y = np.array(y_list)              # (N,)
    idx = pd.DatetimeIndex(y_dates)
    return X, y, idx


def chronological_split(X, y, idx, train_ratio, val_ratio):
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    X_tr, y_tr, idx_tr = X[:n_train], y[:n_train], idx[:n_train]
    X_va, y_va, idx_va = X[n_train:n_train + n_val], y[n_train:n_train + n_val], idx[n_train:n_train + n_val]
    X_te, y_te, idx_te = X[n_train + n_val:], y[n_train + n_val:], idx[n_train + n_val:]
    return (X_tr, y_tr, idx_tr), (X_va, y_va, idx_va), (X_te, y_te, idx_te)


def main():
    # 1) download
    df = yf.download(TICKER, start=START, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("No data downloaded. Check internet or ticker.")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 1:
            close = close.iloc[:, 0]
        elif TICKER in close.columns:
            close = close[TICKER]
        else:
            close = close.iloc[:, 0]

    price = close.dropna()
    returns = np.log(price / price.shift(1)).dropna()

    # 2) sequences
    X, y, idx = build_sequences(returns, SEQ_LEN)

    # 3) split
    (X_tr, y_tr, idx_tr), (X_va, y_va, idx_va), (X_te, y_te, idx_te) = chronological_split(
        X, y, idx, TRAIN_RATIO, VAL_RATIO
    )

    # 4) scale (fit train only) — scale the single feature dimension
    scaler = StandardScaler()
    scaler.fit(X_tr.reshape(-1, 1))
    X_tr_s = scaler.transform(X_tr.reshape(-1, 1)).reshape(X_tr.shape)
    X_va_s = scaler.transform(X_va.reshape(-1, 1)).reshape(X_va.shape)
    X_te_s = scaler.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)

    # 5) loaders
    train_loader = DataLoader(SeqDataset(X_tr_s, y_tr), batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(SeqDataset(X_va_s, y_va), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(SeqDataset(X_te_s, y_te), batch_size=BATCH_SIZE, shuffle=False)

    # 6) model
    model = LSTMRegressor(hidden=HIDDEN, layers=LAYERS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_path = os.path.join(OUT_DIR, "lstm_best.pt")
    patience_left = PATIENCE

    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss_sum, tr_n = 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss_sum += loss.item() * len(xb)
            tr_n += len(xb)

        tr_loss = tr_loss_sum / tr_n
        train_losses.append(tr_loss)

        model.eval()
        va_loss_sum, va_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_loss_sum += loss.item() * len(xb)
                va_n += len(xb)

        va_loss = va_loss_sum / va_n
        val_losses.append(va_loss)

        print(f"Epoch {epoch:02d} | train MSE={tr_loss:.6f} | val MSE={va_loss:.6f}")

        # early stopping
        if va_loss < best_val - 1e-7:
            best_val = va_loss
            patience_left = PATIENCE
            torch.save(model.state_dict(), best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    # 7) test eval
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()

    yhat = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(DEVICE)
            pred = model(xb).cpu().numpy()
            yhat.append(pred)
    yhat = np.concatenate(yhat)

    metrics = {
        "ticker": TICKER,
        "start": START,
        "seq_len": SEQ_LEN,
        "split": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": 1 - TRAIN_RATIO - VAL_RATIO},
        "model": {"type": "LSTM", "hidden": HIDDEN, "layers": LAYERS, "lr": LR, "batch": BATCH_SIZE},
        "test": {
            "MAE": float(mean_absolute_error(y_te, yhat)),
            "RMSE": rmse(y_te, yhat),
            "DA": directional_accuracy(y_te, yhat),
        },
        "best_val_MSE": float(best_val),
        "device": DEVICE,
    }

    with open(os.path.join(OUT_DIR, "metrics_day2_lstm.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # plots
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.title("LSTM learning curve (MSE)")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "lstm_learning_curve.png"), dpi=200)
    plt.close()

    n_plot = min(200, len(y_te))
    t = idx_te[:n_plot]
    plt.figure()
    plt.plot(t, y_te[:n_plot], label="true")
    plt.plot(t, yhat[:n_plot], label="lstm")
    plt.legend()
    plt.title("LSTM: next-day log return (test slice)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "lstm_test_slice_true_vs_pred.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.scatter(y_te, yhat, s=8)
    plt.title("LSTM: true vs predicted (test)")
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "lstm_scatter.png"), dpi=200)
    plt.close()

    print("Done. Outputs:")
    print(" - reports/metrics_day2_lstm.json")
    print(" - reports/lstm_best.pt")
    print(" - reports/figures/lstm_learning_curve.png")
    print(" - reports/figures/lstm_test_slice_true_vs_pred.png")
    print(" - reports/figures/lstm_scatter.png")


if __name__ == "__main__":
    main()
