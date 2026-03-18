import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.common.config import (
    BATCH_SIZE,
    FIGURES_DIR,
    HORIZON,
    REPORTS_DIR,
    SEQ_LEN,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.common.data import SeqDataset, build_sequences, chronological_split, load_data
from src.common.metrics import directional_accuracy
from src.common.reporting import (
    append_experiment_record,
    build_experiment_record,
    create_run_context,
    split_metadata,
    write_model_comparison_record,
)
from src.common.train import train_model

# Import Models
from src.gru.model import GRUModel
from src.lstm.model import LSTMModel
from src.rnn.model import RNNModel
from src.transformer.model import TransformerModel


def _plot_loss_comparison(df: pd.DataFrame) -> None:
    loss_cols = ["best_train_MSE", "best_test_MSE", "best_val_MSE"]
    titles = ["Training Loss (MSE)", "Testing Loss (MSE)", "Validation Loss (MSE)"]

    y_max = float(df[loss_cols].to_numpy().max()) * 1.05
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, col, title in zip(axes, loss_cols, titles):
        ax.bar(df["Model"], df[col])
        ax.set_title(title)
        ax.set_xlabel("Model")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(0, y_max)

    axes[0].set_ylabel("MSE")
    fig.suptitle(f"Loss Comparison Across Models (Horizon={HORIZON})")
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "comparison_losses.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Comparison loss plots saved to {out_path}")


def main():
    print("Running Model Comparison...")

    # 1. Data
    returns = load_data()
    X, y, idx = build_sequences(returns, SEQ_LEN, HORIZON)
    (X_tr, y_tr, _), (X_va, y_va, _), (X_te, y_te, idx_te) = chronological_split(
        X, y, idx, TRAIN_RATIO, VAL_RATIO
    )

    run_context = create_run_context(
        "model_comparison",
        split_metadata(len(X_tr), len(X_va), len(X_te)),
        comparison_group="shared_split_comparison",
        notes="Shared neural/baseline comparison record for the FYP report.",
    )

    comparison_records = []

    # 2. Scale
    scaler = StandardScaler()
    _, _, D = X_tr.shape
    X_tr_s = scaler.fit_transform(X_tr.reshape(-1, D)).reshape(X_tr.shape)
    X_va_s = scaler.transform(X_va.reshape(-1, D)).reshape(X_va.shape)
    X_te_s = scaler.transform(X_te.reshape(-1, D)).reshape(X_te.shape)

    # 3. Loaders
    tr_load = DataLoader(SeqDataset(X_tr_s, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    va_load = DataLoader(SeqDataset(X_va_s, y_va), batch_size=BATCH_SIZE)

    # 4. Baseline (Linear Regression) losses and metrics
    X_tr_flat = X_tr_s.reshape(len(X_tr_s), -1)
    X_va_flat = X_va_s.reshape(len(X_va_s), -1)
    X_te_flat = X_te_s.reshape(len(X_te_s), -1)

    baseline = LinearRegression()
    baseline.fit(X_tr_flat, y_tr)

    yhat_tr = baseline.predict(X_tr_flat)
    yhat_va = baseline.predict(X_va_flat)
    yhat_te = baseline.predict(X_te_flat)

    results = [
        {
            "Model": "Baseline-LR",
            "MAE": float((abs(y_te - yhat_te)).mean()),
            "RMSE": float(mean_squared_error(y_te, yhat_te) ** 0.5),
            "best_train_MSE": float(mean_squared_error(y_tr, yhat_tr)),
            "best_val_MSE": float(mean_squared_error(y_va, yhat_va)),
            "best_test_MSE": float(mean_squared_error(y_te, yhat_te)),
            "DA": directional_accuracy(y_te, yhat_te),
        }
    ]

    baseline_record = build_experiment_record(
        model_name="Baseline-LR",
        record_type="baseline_model",
        metrics={k: v for k, v in results[0].items() if k != "Model"},
        hyperparameters={"model": "LinearRegression", "flattened_sequence": True, "lookback": SEQ_LEN},
        context=run_context,
        tuning={"best_epoch": None, "stop_epoch": None, "tuning_notes": "Default sklearn LinearRegression on flattened sequence inputs."},
        artifacts={},
    )
    append_experiment_record(baseline_record)
    comparison_records.append(baseline_record)

    models = [
        ("RNN", RNNModel, {"hidden": 64}),
        ("LSTM", LSTMModel, {"hidden": 64}),
        ("GRU", GRUModel, {"hidden": 64}),
        ("Transformer", TransformerModel, {"d_model": 64}),
    ]

    for name, cls, kwargs in models:
        metrics = train_model(
            name,
            cls,
            tr_load,
            va_load,
            (X_te_s, y_te),
            idx_te,
            experiment_context=run_context,
            tuning_notes="Final shared-comparison configuration.",
            artifact_paths={
                "prediction_slice": os.path.join(FIGURES_DIR, f"{name.lower()}_pred_slice.png"),
                "scatter": os.path.join(FIGURES_DIR, f"{name.lower()}_scatter.png"),
            },
            **kwargs,
        )
        results.append({"Model": name, **metrics})

    res_df = pd.DataFrame(results)
    print("\nComparison Results:")
    print(res_df)

    res_df.to_csv(os.path.join(REPORTS_DIR, "metrics_comparison.csv"), index=False)
    with open(os.path.join(REPORTS_DIR, "metrics_comparison.json"), "w") as f:
        json.dump(res_df.to_dict(orient="records"), f, indent=2)

    _plot_loss_comparison(res_df)

    comparison_json = os.path.join(REPORTS_DIR, "metrics_comparison.json")
    comparison_csv = os.path.join(REPORTS_DIR, "metrics_comparison.csv")

    for row in results[1:]:
        comparison_records.append(
            build_experiment_record(
                model_name=row["Model"],
                record_type="comparison_summary",
                metrics={k: v for k, v in row.items() if k != "Model"},
                hyperparameters=next(kwargs for model, _cls, kwargs in models if model == row["Model"]),
                context=run_context,
                tuning={"best_epoch": None, "stop_epoch": None, "tuning_notes": "Summary row for shared comparison; epoch-level details are in the experiment log and diagnostics JSON."},
                artifacts={
                    "comparison_csv": comparison_csv,
                    "comparison_json": comparison_json,
                    "comparison_plot": os.path.join(FIGURES_DIR, "comparison_losses.png"),
                    "diagnostics": os.path.join(REPORTS_DIR, f"{row['Model'].lower()}_diagnostics.json"),
                },
            )
        )

    best_row = res_df.sort_values("best_val_MSE", ascending=True).iloc[0].to_dict()
    write_model_comparison_record(
        comparison_records,
        summary={
            "selection_metric": "best_val_MSE",
            "winner": best_row.get("Model"),
            "winner_best_val_MSE": float(best_row.get("best_val_MSE")),
            "num_models": int(len(res_df)),
            "comparison_csv": comparison_csv,
            "comparison_json": comparison_json,
            "comparison_plot": os.path.join(FIGURES_DIR, "comparison_losses.png"),
        },
    )


if __name__ == "__main__":
    main()
