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

# ---------- CONFIG ----------
TICKER = "SPY"
START = "2010-01-01"

# Task Config (Sensitivity Analysis)
SEQ_LEN = 10      # "k" days lookback
HORIZON = 1       # 1 for next day, 10 for task 2

# Training Config
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
BATCH_SIZE = 64
EPOCHS = 100      # Sufficient for convergence curves
LR = 1e-3
PATIENCE = 10     # Relaxed patience

OUT_DIR = "reports/day3"
os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- METRICS ----------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def directional_accuracy(y_true, y_pred):
    return float((np.sign(y_true) == np.sign(y_pred)).mean())

# ---------- DATA ----------
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

def build_sequences(returns: pd.Series, seq_len: int, horizon: int = 1):
    r = returns.values.astype(np.float64)
    dates = returns.index
    
    X_list, y_list, y_dates = [], [], []
    for t in range(seq_len - 1, len(r) - horizon):
        X_list.append(r[t - seq_len + 1: t + 1])
        y_list.append(r[t + horizon])         # Look ahead by horizon
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

# ---------- MODELS ----------
class RNNModel(nn.Module):
    def __init__(self, hidden=64, layers=2):
        super().__init__()
        self.rnn = nn.RNN(1, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze(1)

class LSTMModel(nn.Module):
    def __init__(self, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)

class GRUModel(nn.Module):
    def __init__(self, hidden=64, layers=2):
        super().__init__()
        self.gru = nn.GRU(1, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(1)

class TransformerModel(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model) * 0.02) # Simple learnable pos enc
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, x):
        # x: (B, Seq, 1) -> (B, Seq, d_model)
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :x.size(1), :]
        out = self.transformer_encoder(x)
        # Take last time step
        return self.fc(out[:, -1, :]).squeeze(1)

# ---------- TRAIN LOOP ----------
def train_model(model_name, model_cls, train_loader, val_loader, test_data, test_idx):
    print(f"\nTraining {model_name}...")
    model = model_cls().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    
    best_val = float("inf")
    patience_left = PATIENCE
    train_losses, val_losses = [], []
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(train_loader.dataset)
        train_losses.append(tr_loss)
        
        model.eval()
        va_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                va_loss += loss_fn(model(xb), yb).item() * len(xb)
        va_loss /= len(val_loader.dataset)
        val_losses.append(va_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Tr: {tr_loss:.6f} | Va: {va_loss:.6f}")
            
        if va_loss < best_val - 1e-8:
            best_val = va_loss
            patience_left = PATIENCE
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{model_name}.pt"))
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}")
                break
                
    # Plot Learning Curve
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Test (Val) Loss") 
    plt.title(f"{model_name} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIG_DIR, f"loss_{model_name}.png"))
    plt.close()
    
    # Evaluate on Test
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, f"{model_name}.pt")))
    model.eval()
    X_te, y_te = test_data
    with torch.no_grad():
        preds = model(torch.tensor(X_te, dtype=torch.float32).to(DEVICE)).cpu().numpy()

    # Save per-model plots (learning curve already saved)
    # test slice
    try:
        n_plot = min(200, len(y_te))
        t = test_idx[:n_plot]
        plt.figure()
        plt.plot(t, y_te[:n_plot], label="true")
        plt.plot(t, preds[:n_plot], label=model_name)
        plt.legend()
        plt.title(f"{model_name}: next-day log return (test slice)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{model_name.lower()}_test_slice_true_vs_pred.png"))
        plt.close()

        plt.figure()
        plt.scatter(y_te, preds, s=8)
        plt.title(f"{model_name}: true vs predicted (test)")
        plt.xlabel("true")
        plt.ylabel("pred")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{model_name.lower()}_scatter.png"))
        plt.close()
    except Exception:
        pass

    return {
        "MAE": float(mean_absolute_error(y_te, preds)),
        "RMSE": rmse(y_te, preds),
        "DA": directional_accuracy(y_te, preds),
        "best_val_MSE": float(best_val)
    }

def main():
    # Load Data
    df = yf.download(TICKER, start=START, progress=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    price = close.dropna()
    returns = np.log(price / price.shift(1)).dropna()
    
    # Prepare Data
    X, y, idx = build_sequences(returns, SEQ_LEN, HORIZON)
    (X_tr, y_tr, _), (X_va, y_va, _), (X_te, y_te, idx_te) = chronological_split(X, y, idx, TRAIN_RATIO, VAL_RATIO)
    
    # Scale
    scaler = StandardScaler()
    N, T, D = X_tr.shape
    X_tr_s = scaler.fit_transform(X_tr.reshape(-1, D)).reshape(X_tr.shape)
    X_va_s = scaler.transform(X_va.reshape(-1, D)).reshape(X_va.shape)
    X_te_s = scaler.transform(X_te.reshape(-1, D)).reshape(X_te.shape)
    
    # Loaders
    tr_load = DataLoader(SeqDataset(X_tr_s, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    va_load = DataLoader(SeqDataset(X_va_s, y_va), batch_size=BATCH_SIZE)
    
    # Experiment Models
    results = []
    models = [
        ("RNN", RNNModel),
        ("LSTM", LSTMModel),
        ("GRU", GRUModel),
        ("Transformer", TransformerModel)
    ]
    
    for name, cls in models:
        metrics = train_model(name, cls, tr_load, va_load, (X_te_s, y_te), idx_te)
        results.append({"Model": name, **metrics})
        
    # Save Results
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT_DIR, f"metrics_h{HORIZON}.csv"), index=False)

    # Save JSON report similar to earlier days
    report = {
        "ticker": TICKER,
        "start": START,
        "seq_len": SEQ_LEN,
        "horizon": HORIZON,
        "split": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": 1 - TRAIN_RATIO - VAL_RATIO},
        "device": DEVICE,
        "models": {r["Model"]: {k: v for k, v in r.items() if k != "Model"} for r in results}
    }
    with open(os.path.join(OUT_DIR, f"metrics_h{HORIZON}.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\nFinal Results:")
    print(res_df)
    
    # Comparison Chart
    res_df.set_index("Model")[["MAE", "RMSE"]].plot(kind="bar", rot=0)
    plt.title(f"Model Comparison (Horizon={HORIZON})")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"comparison_h{HORIZON}.png"))
    # Print saved artifacts
    print(f"Saved artifacts to {OUT_DIR}")
    print(f" - {os.path.join(OUT_DIR, f'metrics_h{HORIZON}.json')}")
    print(f" - {os.path.join(OUT_DIR, f'metrics_h{HORIZON}.csv')}")
    for name, _ in models:
        print(f" - {os.path.join(OUT_DIR, name + '.pt')}")
        print(f" - {os.path.join(FIG_DIR, 'loss_' + name + '.png')}")
        print(f" - {os.path.join(FIG_DIR, name.lower() + '_test_slice_true_vs_pred.png')}")
        print(f" - {os.path.join(FIG_DIR, name.lower() + '_scatter.png')}")
    print(f" - {os.path.join(FIG_DIR, f'comparison_h{HORIZON}.png')}")

if __name__ == "__main__":
    main()
