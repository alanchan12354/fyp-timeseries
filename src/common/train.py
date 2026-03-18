import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os
import numpy as np

from .config import (
    DEVICE,
    EPOCHS,
    LR,
    PATIENCE,
    MIN_DELTA,
    VAL_LOSS_SMOOTH_WINDOW,
    MIN_EPOCHS,
    FIGURES_DIR,
    REPORTS_DIR,
    TRAIN_LOG_EVERY,
)
from .metrics import evaluate_preds
from .reporting import append_experiment_record, build_experiment_record

def train_model(
    model_name,
    model_cls,
    train_loader,
    val_loader,
    test_data,
    test_idx,
    *,
    record_experiment=True,
    experiment_context=None,
    selection_metric="best_val_MSE",
    selection_split="validation",
    tuning_notes=None,
    artifact_paths=None,
    **model_kwargs,
):
    print(f"\nTraining {model_name}...")
    train_log_every = int(model_kwargs.pop("train_log_every", TRAIN_LOG_EVERY))
    train_log_every = max(1, train_log_every)
    model_hyperparameters = dict(model_kwargs)

    model = model_cls(**model_kwargs).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_train_loss = float("nan")
    best_test_loss = float("nan")
    best_epoch = 0
    stop_epoch = EPOCHS

    patience_left = PATIENCE
    train_losses, val_losses = [], []
    smooth_val_losses = []
    epoch_diagnostics = []

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

        # Optional smoothing for validation-loss based early stopping.
        # Falls back to raw loss when the configured window is <= 1.
        if VAL_LOSS_SMOOTH_WINDOW and VAL_LOSS_SMOOTH_WINDOW > 1:
            window = min(VAL_LOSS_SMOOTH_WINDOW, len(val_losses))
            va_loss_for_stop = float(np.mean(val_losses[-window:]))
        else:
            va_loss_for_stop = va_loss
        smooth_val_losses.append(va_loss_for_stop)

        in_warmup = epoch < MIN_EPOCHS
        delta = (va_loss_for_stop - best_val) if np.isfinite(best_val) else float("nan")
        improved = va_loss_for_stop < best_val - MIN_DELTA

        if improved:
            best_val = va_loss_for_stop
            best_epoch = epoch
            best_train_loss = tr_loss

            X_te_raw, y_te_raw = test_data
            if not torch.is_tensor(X_te_raw):
                X_te_eval = torch.tensor(X_te_raw, dtype=torch.float32).to(DEVICE)
            else:
                X_te_eval = X_te_raw.to(DEVICE)

            if not torch.is_tensor(y_te_raw):
                y_te_eval = torch.tensor(y_te_raw, dtype=torch.float32).to(DEVICE)
            else:
                y_te_eval = y_te_raw.to(DEVICE)

            with torch.no_grad():
                best_test_loss = loss_fn(model(X_te_eval), y_te_eval).item()

            if epoch >= MIN_EPOCHS:
                patience_left = PATIENCE

            torch.save(model.state_dict(), os.path.join(REPORTS_DIR, f"{model_name}.pt"))

        elif epoch >= MIN_EPOCHS:
            patience_left -= 1

        epoch_diag = {
            "epoch": epoch,
            "train_loss": float(tr_loss),
            "val_loss": float(va_loss),
            "val_loss_for_stop": float(va_loss_for_stop),
            "best_val": float(best_val),
            "delta": float(delta),
            "patience_left": int(patience_left),
            "improved": bool(improved),
        }
        epoch_diagnostics.append(epoch_diag)

        should_log = train_log_every <= 1 or epoch % train_log_every == 0 or in_warmup
        if should_log:
            warmup_msg = f" | Warmup: {epoch}/{MIN_EPOCHS}" if in_warmup else " | Warmup: done"
            smoothed_msg = (
                f" | VaSmooth: {va_loss_for_stop:.6f}"
                if VAL_LOSS_SMOOTH_WINDOW and VAL_LOSS_SMOOTH_WINDOW > 1
                else ""
            )
            print(
                f"Epoch {epoch:03d} | Tr: {tr_loss:.6f} | Va: {va_loss:.6f}"
                f"{smoothed_msg} | best_val: {best_val:.6f} | delta: {delta:+.6f}"
                f" | patience_left: {patience_left}{warmup_msg}"
            )

        if epoch >= MIN_EPOCHS and patience_left <= 0:
            stop_epoch = epoch
            print(f"Early stopping at epoch {epoch}")
            break

        stop_epoch = epoch

    print(
        f"Training stop summary | stop_epoch: {stop_epoch}"
        f" | best_epoch: {best_epoch} | best_val: {best_val:.6f}"
    )

    # Plot Learning Curve
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    if VAL_LOSS_SMOOTH_WINDOW and VAL_LOSS_SMOOTH_WINDOW > 1:
        plt.plot(smooth_val_losses, label=f"Val Loss (MA{VAL_LOSS_SMOOTH_WINDOW})")
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

    if not torch.is_tensor(X_te):
        X_te_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
    else:
        X_te_t = X_te.to(DEVICE)

    with torch.no_grad():
        preds = model(X_te_t).cpu().numpy()

    diagnostics_payload = {
        "model_name": model_name,
        "stop_epoch": int(stop_epoch),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "min_epochs": int(MIN_EPOCHS),
        "patience": int(PATIENCE),
        "min_delta": float(MIN_DELTA),
        "val_loss_smooth_window": int(VAL_LOSS_SMOOTH_WINDOW),
        "train_log_every": int(train_log_every),
        "epochs": epoch_diagnostics,
    }
    diagnostics_path = os.path.join(REPORTS_DIR, f"{model_name.lower()}_diagnostics.json")
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics_payload, f, indent=2)

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
        plt.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], "r--")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"{model_name.lower()}_scatter.png"))
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")

    metrics = evaluate_preds(y_te, preds)
    metrics["best_val_MSE"] = float(best_val)
    metrics["best_train_MSE"] = float(best_train_loss)
    metrics["best_test_MSE"] = float(best_test_loss)

    if record_experiment:
        record = build_experiment_record(
            model_name=model_name,
            record_type="neural_model",
            metrics=metrics,
            hyperparameters=model_hyperparameters,
            context=experiment_context,
            selection_metric=selection_metric,
            selection_split=selection_split,
            tuning={
                "best_epoch": int(best_epoch),
                "stop_epoch": int(stop_epoch),
                "train_log_every": int(train_log_every),
                "tuning_notes": tuning_notes,
            },
            artifacts={
                "checkpoint": os.path.join(REPORTS_DIR, f"{model_name}.pt"),
                "diagnostics": diagnostics_path,
                "loss_curve": loss_path,
                **(artifact_paths or {}),
            },
        )
        append_experiment_record(record)

    return metrics