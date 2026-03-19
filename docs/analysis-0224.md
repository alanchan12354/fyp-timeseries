# Model Analysis Report

**Date:** February 24, 2026
**Project:** SPY Daily Returns Forecasting
**Metrics Source:** `reports/metrics_comparison.csv`, `reports/metrics_baselines.csv`

## 1. Overview
This report analyzes the performance of various time series forecasting models on SPY daily returns. The models range from simple baselines (Persistence, Linear Regression) to deep learning models (RNN, LSTM, GRU, Transformer).

## 2. Methodology
- **Target**: Next-day log return ($r_{t+1}$).
- **Input**: Sequence of past 10 days' log returns.
- **Data Period**: 2010-01-01 to present.
- **Split**: 70% Train, 15% Validation, 15% Test.
- **Loss Function**: Mean Squared Error (MSE).

## 3. Results Summary

### Neural Models (Test Set Metrics)

| Model | MAE | MSE | Directional Accuracy (DA) | Best Val MSE |
| :--- | :--- | :--- | :--- | :--- |
| **RNN** | 0.00680 | 1.02e-4 | 47.37% | 1.40e-4 |
| **LSTM** | 0.00662 | **9.78e-5** | 46.55% | **1.34e-4** |
| **GRU** | **0.00657** | 1.01e-4 | **54.28%** | 1.38e-4 |
| **Transformer** | 0.00900 | 1.62e-4 | 46.71% | 1.96e-4 |

*Note: Best Val MSE is the Mean Squared Error on the validation set during the epoch where the model performed best.*

### Baselines (Test Set Metrics)

| Model | MAE | MSE | Directional Accuracy (DA) |
| :--- | :--- | :--- | :--- |
| **Persistence** | 0.00952 | 2.08e-4 | 52.63% |
| **Linear Regression** | 0.00665 | 9.92e-5 | 52.14% |

## 4. Analysis

### Training & Validation Loss (Generalization)
- **LSTM** achieved the lowest validation MSE ($1.34 \times 10^{-4}$), indicating it generalized the best among the neural models continuously.
- **Transformer** struggled significantly, with a validation MSE ($1.96 \times 10^{-4}$) nearly 46% higher than the LSTM. This suggests the Transformer model may require more data, better positional encoding tuning, or is overfitting to noise in this small-data regime.

### Test Set Performance (MAE & MSE)
- **Error Magnitude**: The **LSTM** and **Linear Regression** performed best in terms of minimizing error (MSE ~1.0e-4).
- **Baselines**: The simple **Linear Regression** is extremely competitive, beating the RNN, GRU, and Transformer on MSE. This is a common finding in financial time series where the "signal" is very weak compared to noise.
- **Deep Learning**: The **GRU** had the lowest Mean Absolute Error (MAE) of 0.00657, marginally beating the LSTM and Linear Regression.

### Directional Accuracy (DA)
- **Best Directional Predictor**: **GRU** achieved the highest DA at **54.28%**, which is the only neural model to decisively beat the random walk/persistence baseline (52.63%).
- **Poor Directional Performance**: RNN, LSTM, and Transformer all hovered around 46-47%, meaning they were worse than a coin flip for predicting the *direction* of the move, even if their magnitude error was low (for LSTM).
- **Interpretation**: A low MSE but low DA (like LSTM) implies the model is predicting close to the mean (0) conservatively, effectively minimizing squared error but failing to capture the swing direction. The GRU managed to capture some trend dynamics.

## 5. Conclusion
1.  **Linear Regression is a tough baseline**: It performs similarly to the LSTM in terms of MSE and beat most neural models in Directional Accuracy.
2.  **GRU is the most promising Neural Model**: It provided the best combination of low MAE and highest Directional Accuracy (>54%).
3.  **Transformer Underperformance**: The current Transformer architecture is likely too complex or ill-suited for this specific univariate, small-window task without further regularization or pre-training.

**Recommendation**: Proceed with **GRU** for further tuning if the goal is trading strategy (DA matters), or **LSTM/Linear Regression** if the goal is volatility/risk estimation (MSE matters).
