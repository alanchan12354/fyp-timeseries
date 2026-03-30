import json
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.common.config import REPORTS_DIR
from src.common.metrics import directional_accuracy
from src.common.neural_entrypoint import (
    build_runtime_parser,
    prepare_runtime_sequence_run,
    resolve_runtime_config,
)
from src.common.reporting import append_experiment_record, build_experiment_record


def main(config=None, config_dict=None, cli_args=None, prepared_run=None, **overrides):
    print("Running Baseline-LR Experiment...")
    runtime_config = resolve_runtime_config(
        config=config,
        config_dict=config_dict,
        cli_args=cli_args,
        **overrides,
    )
    prepared_run = prepared_run or prepare_runtime_sequence_run(
        experiment_name="baseline_lr_experiment",
        batch_size=runtime_config.batch_size,
        run_note=runtime_config.run_note or "Single-model training run for Baseline-LR.",
        training_metadata=runtime_config.training_metadata(),
        seq_len=runtime_config.seq_len,
        horizon=runtime_config.horizon,
        data_source=runtime_config.data_source,
        target_mode=runtime_config.target_mode,
        target_smooth_window=runtime_config.target_smooth_window,
        task_id=runtime_config.task_id,
        random_seed=runtime_config.random_seed,
    )

    X_tr, y_tr = prepared_run.train_data
    X_va, y_va = prepared_run.val_data
    X_te, y_te = prepared_run.test_data

    X_tr_flat = X_tr.reshape(len(X_tr), -1)
    X_va_flat = X_va.reshape(len(X_va), -1)
    X_te_flat = X_te.reshape(len(X_te), -1)

    model = LinearRegression()
    model.fit(X_tr_flat, y_tr)

    yhat_tr = model.predict(X_tr_flat)
    yhat_va = model.predict(X_va_flat)
    yhat_te = model.predict(X_te_flat)

    metrics = {
        "MAE": float((abs(y_te - yhat_te)).mean()),
        "MSE": float(mean_squared_error(y_te, yhat_te)),
        "best_train_MSE": float(mean_squared_error(y_tr, yhat_tr)),
        "best_val_MSE": float(mean_squared_error(y_va, yhat_va)),
        "best_test_MSE": float(mean_squared_error(y_te, yhat_te)),
        "DA": directional_accuracy(y_te, yhat_te),
    }

    record = build_experiment_record(
        model_name="Baseline-LR",
        record_type="baseline_model",
        metrics=metrics,
        hyperparameters={
            "model": "LinearRegression",
            "flattened_sequence": True,
            "flattened_multifeature": True,
            "lookback": runtime_config.seq_len,
        },
        context=prepared_run.run_context,
        tuning={
            "best_epoch": None,
            "stop_epoch": None,
            "tuning_notes": runtime_config.run_note
            or "Default sklearn LinearRegression on flattened multi-feature sequence inputs.",
        },
        artifacts={},
    )
    append_experiment_record(record)

    with open(os.path.join(REPORTS_DIR, "metrics_baseline_lr.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = build_runtime_parser("Baseline-LR")
    args = parser.parse_args()
    main(cli_args=args)
