import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from src.common.config import SEQ_LEN, HORIZON, TRAIN_RATIO, VAL_RATIO, BATCH_SIZE, REPORTS_DIR, FIGURES_DIR
from src.common.data import load_data, build_sequences, chronological_split, SeqDataset
from src.common.train import train_model

# Import Models
from src.rnn.model import RNNModel
from src.lstm.model import LSTMModel
from src.gru.model import GRUModel
from src.transformer.model import TransformerModel

def main():
    print("Running Model Comparison...")
    
    # 1. Data
    returns = load_data()
    X, y, idx = build_sequences(returns, SEQ_LEN, HORIZON)
    (X_tr, y_tr, _), (X_va, y_va, _), (X_te, y_te, idx_te) = chronological_split(X, y, idx, TRAIN_RATIO, VAL_RATIO)
    
    # 2. Scale
    scaler = StandardScaler()
    N, T, D = X_tr.shape
    X_tr_s = scaler.fit_transform(X_tr.reshape(-1, D)).reshape(X_tr.shape)
    X_va_s = scaler.transform(X_va.reshape(-1, D)).reshape(X_va.shape) 
    X_te_s = scaler.transform(X_te.reshape(-1, D)).reshape(X_te.shape)
    
    # 3. Loaders
    tr_load = DataLoader(SeqDataset(X_tr_s, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    va_load = DataLoader(SeqDataset(X_va_s, y_va), batch_size=BATCH_SIZE)
    
    results = []
    models = [
        ("RNN", RNNModel, {"hidden": 64}),
        ("LSTM", LSTMModel, {"hidden": 64}),
        ("GRU", GRUModel, {"hidden": 64}),
        ("Transformer", TransformerModel, {"d_model": 64})
    ]
    
    for name, cls, kwargs in models:
        metrics = train_model(name, cls, tr_load, va_load, (X_te_s, y_te), idx_te, **kwargs)
        results.append({"Model": name, **metrics})
        
    res_df = pd.DataFrame(results)
    print("\nComparison Results:")
    print(res_df)
    
    res_df.to_csv(os.path.join(REPORTS_DIR, "metrics_comparison.csv"), index=False)
    
    # Bar Chart
    res_df.set_index("Model")[["MAE", "RMSE"]].plot(kind="bar", rot=0)
    plt.title(f"Model Comparison (Horizon={HORIZON})")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "comparison_all.png"))
    print(f"Comparison plot saved to {os.path.join(FIGURES_DIR, 'comparison_all.png')}")

if __name__ == "__main__":
    main()
