import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np

from .config import DEVICE, EPOCHS, LR, PATIENCE, MIN_EPOCHS, FIGURES_DIR, REPORTS_DIR
from .metrics import evaluate_preds

def train_model(model_name, model_cls, train_loader, val_loader, test_data, test_idx, **model_kwargs):
    print(f"\nTraining {model_name}...")
    model = model_cls(**model_kwargs).to(DEVICE)
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
        
        in_warmup = epoch < MIN_EPOCHS
        if epoch % 10 == 0 or in_warmup:
            warmup_msg = f" | Warmup: {epoch}/{MIN_EPOCHS}" if in_warmup else " | Warmup: done"
            print(f"Epoch {epoch:03d} | Tr: {tr_loss:.6f} | Va: {va_loss:.6f}{warmup_msg}")

        if va_loss < best_val - 1e-8:
            best_val = va_loss
            if epoch >= MIN_EPOCHS:
                patience_left = PATIENCE
            torch.save(model.state_dict(), os.path.join(REPORTS_DIR, f"{model_name}.pt"))
        elif epoch >= MIN_EPOCHS:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}")
                break
                
    # Plot Learning Curve
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss") 
    plt.title(f"{model_name} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_path = os.path.join(FIGURES_DIR, f"loss_{model_name}.png")
    plt.savefig(loss_path)
    plt.close()
    
    # Evaluate on Test
    model.load_state_dict(torch.load(os.path.join(REPORTS_DIR, f"{model_name}.pt"), map_location=DEVICE))
    model.eval()
    X_te, y_te = test_data
    
    # Ensure X_te is tensor
    if not torch.is_tensor(X_te):
        X_te_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
    else:
        X_te_t = X_te.to(DEVICE)
        
    with torch.no_grad():
        preds = model(X_te_t).cpu().numpy()

    # Save plots
    try:
        n_plot = min(200, len(y_te))
        t = test_idx[:n_plot]
        plt.figure()
        plt.plot(t, y_te[:n_plot], label="true")
        plt.plot(t, preds[:n_plot], label=model_name)
        plt.legend()
        plt.title(f"{model_name}: Test Prediction Slice")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"{model_name.lower()}_pred_slice.png"))
        plt.close()

        plt.figure()
        plt.scatter(y_te, preds, s=8, alpha=0.5)
        plt.title(f"{model_name}: True vs Predicted")
        plt.xlabel("True")
        plt.ylabel("Pred")
        plt.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], 'r--') # identify line
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"{model_name.lower()}_scatter.png"))
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")

    metrics = evaluate_preds(y_te, preds)
    metrics["best_val_MSE"] = float(best_val)
    return metrics
