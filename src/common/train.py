import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os
import re
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
    SCHEDULER_TYPE,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    SCHEDULER_MIN_LR,
)
from .metrics import evaluate_preds
from .reporting import append_experiment_record, build_experiment_record


def _build_scheduler(optimizer, *, scheduler_type, epochs, scheduler_factor, scheduler_patience, scheduler_min_lr):
    scheduler_type = (scheduler_type or "none").strip().lower()
    if scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr,
        )
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=scheduler_min_lr,
        )
    return None


def _step_scheduler(scheduler, scheduler_type, metric):
    if scheduler is None:
        return
    if scheduler_type == "plateau":
        scheduler.step(metric)
    else:
        scheduler.step()


def _slugify_path_component(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip())
    text = re.sub(r"-+", "-", text).strip("-._")
    return text or "run"


def _build_figure_stem(model_name, run_id=None):
    parts = [_slugify_path_component(model_name)]
    if run_id:
        parts.append(_slugify_path_component(run_id))
    return "_".join(parts)


def _format_plot_hyperparameters(model_hyperparameters, training_hyperparameters):
    combined = {**training_hyperparameters, **model_hyperparameters}
    if not combined:
        return "Hyperparameters: none"
    ordered_items = [f"{key}={combined[key]}" for key in sorted(combined)]
    return "Hyperparameters: " + ", ".join(ordered_items)


def _annotate_figure_with_hyperparameters(fig, text):
    fig.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))


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
    notes_metadata=None,
    artifact_paths=None,
    **model_kwargs,
):
    print(f"\nTraining {model_name}...")
    learning_rate = float(model_kwargs.pop("learning_rate", LR))
    epochs = int(model_kwargs.pop("epochs", EPOCHS))
    patience = int(model_kwargs.pop("patience", PATIENCE))
    min_delta = float(model_kwargs.pop("min_delta", MIN_DELTA))
    val_loss_smooth_window = int(model_kwargs.pop("val_loss_smooth_window", VAL_LOSS_SMOOTH_WINDOW))
    min_epochs = int(model_kwargs.pop("min_epochs", MIN_EPOCHS))
    train_log_every = int(model_kwargs.pop("train_log_every", TRAIN_LOG_EVERY))
    scheduler_type = model_kwargs.pop("scheduler_type", SCHEDULER_TYPE)
    scheduler_factor = float(model_kwargs.pop("scheduler_factor", SCHEDULER_FACTOR))
    scheduler_patience = int(model_kwargs.pop("scheduler_patience", SCHEDULER_PATIENCE))
    scheduler_min_lr = float(model_kwargs.pop("scheduler_min_lr", SCHEDULER_MIN_LR))
    train_log_every = max(1, train_log_every)
    model_hyperparameters = dict(model_kwargs)
    training_hyperparameters = {
        "batch_size": getattr(train_loader, "batch_size", None),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "min_delta": min_delta,
        "min_epochs": min_epochs,
        "patience": patience,
        "scheduler_factor": scheduler_factor,
        "scheduler_min_lr": scheduler_min_lr,
        "scheduler_patience": scheduler_patience,
        "scheduler_type": scheduler_type,
        "val_loss_smooth_window": val_loss_smooth_window,
    }
    training_hyperparameters = {k: v for k, v in training_hyperparameters.items() if v is not None}
    plot_hyperparameters_text = _format_plot_hyperparameters(model_hyperparameters, training_hyperparameters)

    model = model_cls(**model_kwargs).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = _build_scheduler(
        opt,
        scheduler_type=scheduler_type,
        epochs=epochs,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
        scheduler_min_lr=scheduler_min_lr,
    )
    scheduler_type = (scheduler_type or "none").strip().lower()
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_train_loss = float("nan")
    best_test_loss = float("nan")
    best_epoch = 0
    stop_epoch = epochs

    patience_left = patience
    train_losses, val_losses = [], []
    smooth_val_losses = []
    epoch_diagnostics = []

    for epoch in range(1, epochs + 1):
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

        if val_loss_smooth_window and val_loss_smooth_window > 1:
            window = min(val_loss_smooth_window, len(val_losses))
            va_loss_for_stop = float(np.mean(val_losses[-window:]))
        else:
            va_loss_for_stop = va_loss
        smooth_val_losses.append(va_loss_for_stop)

        _step_scheduler(scheduler, scheduler_type, va_loss_for_stop)
        current_lr = float(opt.param_groups[0]["lr"])

        in_warmup = epoch < min_epochs
        delta = (va_loss_for_stop - best_val) if np.isfinite(best_val) else float("nan")
        improved = va_loss_for_stop < best_val - min_delta

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

            if epoch >= min_epochs:
                patience_left = patience

            torch.save(model.state_dict(), os.path.join(REPORTS_DIR, f"{model_name}.pt"))

        elif epoch >= min_epochs:
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
            "learning_rate": current_lr,
        }
        epoch_diagnostics.append(epoch_diag)

        should_log = train_log_every <= 1 or epoch % train_log_every == 0 or in_warmup
        if should_log:
            warmup_msg = f" | Warmup: {epoch}/{min_epochs}" if in_warmup else " | Warmup: done"
            smoothed_msg = (
                f" | VaSmooth: {va_loss_for_stop:.6f}"
                if val_loss_smooth_window and val_loss_smooth_window > 1
                else ""
            )
            scheduler_msg = f" | lr: {current_lr:.2e}"
            print(
                f"Epoch {epoch:03d} | Tr: {tr_loss:.6f} | Va: {va_loss:.6f}"
                f"{smoothed_msg} | best_val: {best_val:.6f} | delta: {delta:+.6f}"
                f" | patience_left: {patience_left}{scheduler_msg}{warmup_msg}"
            )

        if epoch >= min_epochs and patience_left <= 0:
            stop_epoch = epoch
            print(f"Early stopping at epoch {epoch}")
            break

        stop_epoch = epoch

    print(
        f"Training stop summary | stop_epoch: {stop_epoch}"
        f" | best_epoch: {best_epoch} | best_val: {best_val:.6f}"
    )

    run_id = (experiment_context or {}).get("run_id")
    figure_stem = _build_figure_stem(model_name, run_id)

    fig = plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    if val_loss_smooth_window and val_loss_smooth_window > 1:
        plt.plot(smooth_val_losses, label=f"Val Loss (MA{val_loss_smooth_window})")
    plt.title(f"{model_name} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _annotate_figure_with_hyperparameters(fig, plot_hyperparameters_text)
    loss_path = os.path.join(FIGURES_DIR, f"loss_{figure_stem}.png")
    plt.savefig(loss_path)
    plt.close(fig)

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
        "min_epochs": int(min_epochs),
        "patience": int(patience),
        "min_delta": float(min_delta),
        "val_loss_smooth_window": int(val_loss_smooth_window),
        "train_log_every": int(train_log_every),
        "scheduler_type": scheduler_type,
        "scheduler_factor": float(scheduler_factor),
        "scheduler_patience": int(scheduler_patience),
        "scheduler_min_lr": float(scheduler_min_lr),
        "figure_hyperparameters": plot_hyperparameters_text,
        "epochs": epoch_diagnostics,
    }
    diagnostics_path = os.path.join(REPORTS_DIR, f"{model_name.lower()}_diagnostics.json")
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics_payload, f, indent=2)

    try:
        n_plot = min(200, len(y_te))
        t = test_idx[:n_plot]

        fig = plt.figure()
        plt.plot(t, y_te[:n_plot], label="true")
        plt.plot(t, preds[:n_plot], label=model_name)
        plt.legend()
        plt.title(f"{model_name}: Test Prediction Slice")
        plt.xticks(rotation=45)
        _annotate_figure_with_hyperparameters(fig, plot_hyperparameters_text)
        pred_slice_path = os.path.join(FIGURES_DIR, f"{figure_stem.lower()}_pred_slice.png")
        plt.savefig(pred_slice_path)
        plt.close(fig)

        fig = plt.figure()
        plt.scatter(y_te, preds, s=8, alpha=0.5)
        plt.title(f"{model_name}: True vs Predicted")
        plt.xlabel("True")
        plt.ylabel("Pred")
        plt.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], "r--")
        _annotate_figure_with_hyperparameters(fig, plot_hyperparameters_text)
        scatter_path = os.path.join(FIGURES_DIR, f"{figure_stem.lower()}_scatter.png")
        plt.savefig(scatter_path)
        plt.close(fig)
    except Exception as e:
        print(f"Plotting failed: {e}")
        pred_slice_path = None
        scatter_path = None

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
                "scheduler_type": scheduler_type,
                "scheduler_factor": float(scheduler_factor),
                "scheduler_patience": int(scheduler_patience),
                "scheduler_min_lr": float(scheduler_min_lr),
                "tuning_notes": tuning_notes,
                "notes_metadata": notes_metadata or {},
            },
            artifacts={
                **(artifact_paths or {}),
                "checkpoint": os.path.join(REPORTS_DIR, f"{model_name}.pt"),
                "diagnostics": diagnostics_path,
                "loss_curve": loss_path,
                "prediction_slice": pred_slice_path,
                "scatter": scatter_path,
            },
        )
        append_experiment_record(record)

    return metrics
