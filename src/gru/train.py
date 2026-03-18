import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json
import os

from src.common.config import SEQ_LEN, HORIZON, TRAIN_RATIO, VAL_RATIO, BATCH_SIZE, REPORTS_DIR
from src.common.data import load_data, build_sequences, chronological_split, SeqDataset
from src.common.reporting import create_run_context, split_metadata
from src.common.train import train_model
from .model import GRUModel

def main():
    print("Running GRU Experiment...")
    
    # 1. Data
    returns = load_data()
    X, y, idx = build_sequences(returns, SEQ_LEN, HORIZON)
    (X_tr, y_tr, _), (X_va, y_va, _), (X_te, y_te, idx_te) = chronological_split(X, y, idx, TRAIN_RATIO, VAL_RATIO)
    
    run_context = create_run_context(
        "gru_experiment",
        split_metadata(len(X_tr), len(X_va), len(X_te)),
        notes="Single-model training run for GRU.",
    )

    # 2. Scale
    scaler = StandardScaler()
    N, T, D = X_tr.shape
    X_tr_s = scaler.fit_transform(X_tr.reshape(-1, D)).reshape(X_tr.shape)
    X_va_s = scaler.transform(X_va.reshape(-1, D)).reshape(X_va.shape) 
    X_te_s = scaler.transform(X_te.reshape(-1, D)).reshape(X_te.shape)
    
    # 3. Loaders
    tr_load = DataLoader(SeqDataset(X_tr_s, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    va_load = DataLoader(SeqDataset(X_va_s, y_va), batch_size=BATCH_SIZE)
    
    # 4. Train
    metrics = train_model("GRU", GRUModel, tr_load, va_load, (X_te_s, y_te), idx_te, experiment_context=run_context, tuning_notes="Single-model default configuration.", hidden=64, layers=2)
    
    print(metrics)
    with open(os.path.join(REPORTS_DIR, "metrics_gru.json"), "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
