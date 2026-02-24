import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def directional_accuracy(y_true, y_pred):
    return float((np.sign(y_true) == np.sign(y_pred)).mean())

def evaluate_preds(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "DA": directional_accuracy(y_true, y_pred)
    }
