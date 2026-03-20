import json
import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.common.config import HORIZON, REPORTS_DIR, SEQ_LEN, TRAIN_RATIO, VAL_RATIO
from src.common.data import chronological_split, load_data, make_lag_features
from src.common.metrics import evaluate_preds
from src.common.reporting import (
    append_experiment_record,
    build_experiment_record,
    create_run_context,
    default_task_metadata,
    default_training_metadata,
    split_metadata,
)


def main():
    print("Running Baselines...")
    # 1. Data
    returns = load_data()
    X, y, idx = make_lag_features(returns, SEQ_LEN, HORIZON)

    # 2. Split
    (X_tr, y_tr, _), (X_va, y_va, _), (X_te, y_te, idx_te) = chronological_split(
        X, y, idx, TRAIN_RATIO, VAL_RATIO
    )

    run_context = create_run_context(
        "baselines",
        split_metadata(len(X_tr), len(X_va), len(X_te)),
        task_meta={
            **default_task_metadata(),
            "target": f"log return at t + {HORIZON} (aligned baseline setup)",
            "input_description": f"previous {SEQ_LEN} daily log returns",
            "prediction_horizon": HORIZON,
            "input_window": SEQ_LEN,
        },
        training_meta=default_training_metadata(
            lr=None, epochs=None, patience=None, min_delta=None, min_epochs=None
        ),
        notes="Baseline benchmark record aligned to the shared neural-model horizon and lookback.",
    )

    # 3. Scale
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    results = []

    # --- Persistence ---
    # yhat = r_t (first column of X, since X = [r_t, r_{t-1}, ...])
    yhat_persist = X_te[:, 0]
    m_persist = evaluate_preds(y_te, yhat_persist)
    m_persist["best_train_MSE"] = float(mean_squared_error(y_tr, X_tr[:, 0]))
    m_persist["best_val_MSE"] = float(mean_squared_error(y_va, X_va[:, 0]))
    m_persist["best_test_MSE"] = float(mean_squared_error(y_te, yhat_persist))
    results.append({"Model": "Persistence", **m_persist})

    # --- Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_tr_s, y_tr)
    yhat_lr = lr.predict(X_te_s)
    m_lr = evaluate_preds(y_te, yhat_lr)
    m_lr["best_train_MSE"] = float(mean_squared_error(y_tr, lr.predict(X_tr_s)))
    m_lr["best_val_MSE"] = float(mean_squared_error(y_va, lr.predict(X_va_s)))
    m_lr["best_test_MSE"] = float(mean_squared_error(y_te, yhat_lr))
    results.append({"Model": "LinearRegression", **m_lr})

    append_experiment_record(
        build_experiment_record(
            model_name="Persistence",
            record_type="baseline_model",
            metrics=m_persist,
            hyperparameters={
                "rule": "y_hat = latest observed return",
                "lookback": SEQ_LEN,
                "prediction_horizon": HORIZON,
            },
            context=run_context,
            selection_metric="best_test_MSE",
            selection_split="test",
            tuning={
                "best_epoch": None,
                "stop_epoch": None,
                "tuning_notes": "No tuning for persistence baseline.",
            },
        )
    )
    append_experiment_record(
        build_experiment_record(
            model_name="LinearRegression",
            record_type="baseline_model",
            metrics=m_lr,
            hyperparameters={
                "model": "LinearRegression",
                "lookback": SEQ_LEN,
                "prediction_horizon": HORIZON,
                "scaled_inputs": True,
            },
            context=run_context,
            selection_metric="best_val_MSE",
            selection_split="validation",
            tuning={
                "best_epoch": None,
                "stop_epoch": None,
                "tuning_notes": "Default sklearn LinearRegression baseline aligned to the shared horizon/lookback task.",
            },
        )
    )

    # Save
    df = pd.DataFrame(results)
    print(df)
    df.to_csv(os.path.join(REPORTS_DIR, "metrics_baselines.csv"), index=False)

    with open(os.path.join(REPORTS_DIR, "metrics_baselines.json"), "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)


if __name__ == "__main__":
    main()
