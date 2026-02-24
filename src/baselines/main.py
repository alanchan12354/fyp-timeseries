import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os
import json

from src.common.config import LAGS, TRAIN_RATIO, VAL_RATIO, REPORTS_DIR, FIGURES_DIR
from src.common.data import load_data, make_lag_features, chronological_split
from src.common.metrics import evaluate_preds

def main():
    print("Running Baselines...")
    # 1. Data
    returns = load_data()
    X, y, idx = make_lag_features(returns, LAGS)
    
    # 2. Split
    (X_tr, y_tr, _), (X_va, y_va, _), (X_te, y_te, idx_te) = chronological_split(X, y, idx, TRAIN_RATIO, VAL_RATIO)
    
    # 3. Scale
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)
    
    results = []
    
    # --- Persistence ---
    # yhat = r_t (first column of X, since X = [r_t, r_{t-1}...])
    yhat_persist = X_te[:, 0]
    m_persist = evaluate_preds(y_te, yhat_persist)
    results.append({"Model": "Persistence", **m_persist})
    
    # --- Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_tr_s, y_tr)
    yhat_lr = lr.predict(X_te_s)
    m_lr = evaluate_preds(y_te, yhat_lr)
    results.append({"Model": "LinearRegression", **m_lr})
    
    # Save
    df = pd.DataFrame(results)
    print(df)
    df.to_csv(os.path.join(REPORTS_DIR, "metrics_baselines.csv"), index=False)
    
    with open(os.path.join(REPORTS_DIR, "metrics_baselines.json"), "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

if __name__ == "__main__":
    main()
