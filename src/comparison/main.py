import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from src.common.config import FIGURES_DIR, REPORTS_DIR, SEQ_LEN
from src.common.data import SeqDataset
from src.common.experiment import prepare_sequence_experiment_run
from src.common.metrics import directional_accuracy
from src.common.reporting import (
    append_experiment_record,
    build_experiment_record,
    default_task_metadata,
    write_model_comparison_record,
)
from src.common.runtime_config import RuntimeTrainingConfig, add_runtime_config_args
from src.common.train import train_model

# Import Models
from src.gru.model import GRUModel
from src.lstm.model import LSTMModel
from src.rnn.model import RNNModel
from src.transformer.model import TransformerModel


def _slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip()) or "task"


def _task_artifact_paths(task_id: str) -> dict[str, str]:
    safe_task_id = _slugify(task_id)
    return {
        "comparison_plot": os.path.join(FIGURES_DIR, f"comparison_losses_{safe_task_id}.png"),
        "comparison_csv": os.path.join(REPORTS_DIR, f"metrics_comparison_{safe_task_id}.csv"),
        "comparison_json": os.path.join(REPORTS_DIR, f"metrics_comparison_{safe_task_id}.json"),
    }


def _plot_loss_comparison(df: pd.DataFrame, *, horizon: int, out_path: str, task_id: str) -> None:
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
    fig.suptitle(f"Loss Comparison Across Models (task={task_id}, horizon={horizon})")
    plt.tight_layout()

    plt.savefig(out_path)
    plt.close(fig)
    print(f"Comparison loss plots saved to {out_path}")


def _build_runtime_config(cli_args: argparse.Namespace | None = None, **overrides) -> RuntimeTrainingConfig:
    config_values = {**({} if cli_args is None else vars(cli_args)), **overrides}
    return RuntimeTrainingConfig.from_sources(config_dict=config_values)


def main(cli_args=None, **overrides):
    runtime_config = _build_runtime_config(cli_args=cli_args, **overrides)
    task_meta = default_task_metadata(
        task_id=runtime_config.task_id,
        data_source=runtime_config.data_source,
        horizon=runtime_config.horizon,
        input_window=SEQ_LEN,
        target_mode=runtime_config.target_mode,
        target_smooth_window=runtime_config.target_smooth_window,
    )
    task_id = task_meta["task_id"]

    print(
        "Running Model Comparison "
        f"for task_id={task_id} (source={runtime_config.data_source}, "
        f"target_mode={runtime_config.target_mode}, horizon={runtime_config.horizon}, "
        f"smooth_window={runtime_config.target_smooth_window})..."
    )

    prepared_run = prepare_sequence_experiment_run(
        batch_size=runtime_config.batch_size,
        experiment_name="model_comparison",
        run_note=runtime_config.run_note
        or "Task-aware shared neural/baseline comparison record for the FYP report.",
        training_metadata=runtime_config.training_metadata(),
        horizon=runtime_config.horizon,
        data_source=runtime_config.data_source,
        target_mode=runtime_config.target_mode,
        target_smooth_window=runtime_config.target_smooth_window,
        task_id=task_id,
    )

    X_tr_s, y_tr = prepared_run.train_data
    X_va_s, y_va = prepared_run.val_data
    X_te_s, y_te = prepared_run.test_data
    idx_te = prepared_run.test_index
    run_context = prepared_run.run_context

    artifacts = _task_artifact_paths(task_id)

    comparison_records = []

    # Baseline (Linear Regression) losses and metrics
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
            "MSE": float(mean_squared_error(y_te, yhat_te)),
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
        hyperparameters={
            "model": "LinearRegression",
            "flattened_sequence": True,
            "flattened_multifeature": True,
            "lookback": SEQ_LEN,
        },
        context=run_context,
        tuning={
            "best_epoch": None,
            "stop_epoch": None,
            "tuning_notes": "Default sklearn LinearRegression on flattened multi-feature sequence inputs.",
        },
        artifacts={},
    )
    append_experiment_record(baseline_record)
    comparison_records.append(baseline_record)

    _, _, feature_dim = X_tr_s.shape
    tr_load = DataLoader(SeqDataset(X_tr_s, y_tr), batch_size=runtime_config.batch_size, shuffle=True)
    va_load = DataLoader(SeqDataset(X_va_s, y_va), batch_size=runtime_config.batch_size)

    models = [
        (
            "RNN",
            RNNModel,
            {
                "hidden": runtime_config.recurrent_hidden_size,
                "layers": runtime_config.recurrent_layer_count,
                "input_size": feature_dim,
            },
        ),
        (
            "LSTM",
            LSTMModel,
            {
                "hidden": runtime_config.recurrent_hidden_size,
                "layers": runtime_config.recurrent_layer_count,
                "input_size": feature_dim,
            },
        ),
        (
            "GRU",
            GRUModel,
            {
                "hidden": runtime_config.recurrent_hidden_size,
                "layers": runtime_config.recurrent_layer_count,
                "input_size": feature_dim,
            },
        ),
        (
            "Transformer",
            TransformerModel,
            {
                "d_model": runtime_config.transformer_d_model,
                "nhead": runtime_config.transformer_nhead,
                "num_layers": runtime_config.transformer_num_layers,
                "dropout": 0.1,
                "input_size": feature_dim,
            },
        ),
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
            tuning_notes=(
                "Task-aware shared-comparison configuration "
                f"(task_id={task_id}, target_mode={runtime_config.target_mode}, "
                f"horizon={runtime_config.horizon}, smooth_window={runtime_config.target_smooth_window})."
            ),
            learning_rate=runtime_config.learning_rate,
            epochs=runtime_config.epochs,
            scheduler_type=runtime_config.scheduler_type,
            artifact_paths={
                "prediction_slice": os.path.join(FIGURES_DIR, f"{name.lower()}_{_slugify(task_id)}_pred_slice.png"),
                "scatter": os.path.join(FIGURES_DIR, f"{name.lower()}_{_slugify(task_id)}_scatter.png"),
            },
            **kwargs,
        )
        results.append({"Model": name, **metrics})

    res_df = pd.DataFrame(results)
    print("\nComparison Results:")
    print(res_df)

    res_df.to_csv(artifacts["comparison_csv"], index=False)
    with open(artifacts["comparison_json"], "w", encoding="utf-8") as f:
        json.dump(res_df.to_dict(orient="records"), f, indent=2)

    _plot_loss_comparison(
        res_df,
        horizon=runtime_config.horizon,
        out_path=artifacts["comparison_plot"],
        task_id=task_id,
    )

    for row in results[1:]:
        comparison_records.append(
            build_experiment_record(
                model_name=row["Model"],
                record_type="comparison_summary",
                metrics={k: v for k, v in row.items() if k != "Model"},
                hyperparameters=next(kwargs for model, _cls, kwargs in models if model == row["Model"]),
                context=run_context,
                tuning={
                    "best_epoch": None,
                    "stop_epoch": None,
                    "tuning_notes": (
                        "Summary row for task-aware shared comparison; "
                        "epoch-level details are in the experiment log and diagnostics JSON."
                    ),
                },
                artifacts={
                    "comparison_csv": artifacts["comparison_csv"],
                    "comparison_json": artifacts["comparison_json"],
                    "comparison_plot": artifacts["comparison_plot"],
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
            "task_id": task_id,
            "target_mode": runtime_config.target_mode,
            "horizon": int(runtime_config.horizon),
            "target_smooth_window": int(runtime_config.target_smooth_window),
            "comparison_csv": artifacts["comparison_csv"],
            "comparison_json": artifacts["comparison_json"],
            "comparison_plot": artifacts["comparison_plot"],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the task-aware model comparison pipeline.")
    add_runtime_config_args(parser)
    args = parser.parse_args()
    main(cli_args=args)
