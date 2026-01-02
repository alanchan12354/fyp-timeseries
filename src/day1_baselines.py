import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- config ----------
TICKER = "SPY"
START = "2010-01-01"
LAGS = 10
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
OUT_DIR = "reports"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def directional_accuracy(y_true, y_pred):
    return float((np.sign(y_true) == np.sign(y_pred)).mean())

def make_lag_features(returns: pd.Series, lags: int):
    """
    returns: r_t
    target: y_t = r_{t+1}
    features at time t: [r_t, r_{t-1}, ..., r_{t-lags+1}]
    """
    df = pd.DataFrame({"r": returns})
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["r"].shift(i - 1)  # lag_1 = r_t
    df["y"] = df["r"].shift(-1)               # y = r_{t+1}
    df = df.dropna()
    X = df[[f"lag_{i}" for i in range(1, lags + 1)]].values
    y = df["y"].values
    idx = df.index
    return X, y, idx

def chronological_split(X, y, idx, train_ratio, val_ratio):
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    X_train, y_train, idx_train = X[:n_train], y[:n_train], idx[:n_train]
    X_val, y_val, idx_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val], idx[n_train:n_train + n_val]
    X_test, y_test, idx_test = X[n_train + n_val:], y[n_train + n_val:], idx[n_train + n_val:]
    return (X_train, y_train, idx_train), (X_val, y_val, idx_val), (X_test, y_test, idx_test)

def main():
    # 1) download data
    df = yf.download(TICKER, start=START, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("No data downloaded. Check internet or ticker.")

    close = df["Close"]

    # yfinance sometimes returns Close as a DataFrame (e.g., one column SPY) instead of a Series
    if isinstance(close, pd.DataFrame):
        # if there's exactly one column, take it
        if close.shape[1] == 1:
            close = close.iloc[:, 0]
        # otherwise try to select the ticker column
        elif TICKER in close.columns:
            close = close[TICKER]
        else:
            close = close.iloc[:, 0]

    price = close.dropna()
    returns = np.log(price / price.shift(1)).dropna()
    print(type(returns), getattr(returns, "shape", None))



    # 2) features/targets
    X, y, idx = make_lag_features(returns, LAGS)

    # 3) split (chronological)
    (X_tr, y_tr, idx_tr), (X_va, y_va, idx_va), (X_te, y_te, idx_te) = chronological_split(
        X, y, idx, TRAIN_RATIO, VAL_RATIO
    )

    # 4) scaler (fit train only)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    # ---------- baseline 1: persistence ----------
    # predict r_{t+1} using r_t == lag_1 (first feature column)
    yhat_persist_te = X_te[:, 0]

    # ---------- baseline 2: ridge regression ----------
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    best_alpha, best_val_rmse = None, float("inf")
    for a in alphas:
        model = Ridge(alpha=a, random_state=42)
        model.fit(X_tr_s, y_tr)
        yhat_va = model.predict(X_va_s)
        r = rmse(y_va, yhat_va)
        if r < best_val_rmse:
            best_val_rmse = r
            best_alpha = a

    ridge = Ridge(alpha=best_alpha, random_state=42)
    ridge.fit(np.vstack([X_tr_s, X_va_s]), np.hstack([y_tr, y_va]))
    yhat_ridge_te = ridge.predict(X_te_s)

    # ---------- metrics ----------
    metrics = {
        "ticker": TICKER,
        "start": START,
        "lags": LAGS,
        "split": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": 1 - TRAIN_RATIO - VAL_RATIO},
        "persistence": {
            "MAE": float(mean_absolute_error(y_te, yhat_persist_te)),
            "RMSE": rmse(y_te, yhat_persist_te),
            "DA": directional_accuracy(y_te, yhat_persist_te),
        },
        "ridge": {
            "alpha": float(best_alpha),
            "MAE": float(mean_absolute_error(y_te, yhat_ridge_te)),
            "RMSE": rmse(y_te, yhat_ridge_te),
            "DA": directional_accuracy(y_te, yhat_ridge_te),
            "val_RMSE_for_alpha": float(best_val_rmse),
        },
    }

    # save metrics
    with open(os.path.join(OUT_DIR, "metrics_day1.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # also a small csv
    rows = []
    for name in ["persistence", "ridge"]:
        rows.append({"model": name, **{k: metrics[name][k] for k in ["MAE", "RMSE", "DA"]}})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "metrics_day1.csv"), index=False)

    # ---------- plots ----------
    # plot 1: y_true vs y_pred (first 200 test points)
    n_plot = min(200, len(y_te))
    t = idx_te[:n_plot]

    plt.figure()
    plt.plot(t, y_te[:n_plot], label="true")
    plt.plot(t, yhat_persist_te[:n_plot], label="persist")
    plt.plot(t, yhat_ridge_te[:n_plot], label="ridge")
    plt.legend()
    plt.title("Next-day log return: true vs predicted (test slice)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "test_slice_true_vs_pred.png"), dpi=200)
    plt.close()

    # plot 2: scatter true vs ridge pred
    plt.figure()
    plt.scatter(y_te, yhat_ridge_te, s=8)
    plt.title("Ridge: true vs predicted (test)")
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ridge_scatter.png"), dpi=200)
    plt.close()

    print("Done. Outputs:")
    print(" - reports/metrics_day1.json")
    print(" - reports/metrics_day1.csv")
    print(" - reports/figures/test_slice_true_vs_pred.png")
    print(" - reports/figures/ridge_scatter.png")

if __name__ == "__main__":
    main()
