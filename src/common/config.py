import os
import torch
from datetime import datetime, timezone

# Data Config
TICKER = "SPY"
START = "2005-01-01"

# Task Config
SEQ_LEN = 30      # Lookback window
HORIZON = 1       # Prediction horizon
TARGET_MODE = "horizon_return"  # options: "horizon_return", "next_return", "next3_mean_return", "next_mean_return", "next_volatility", "sine_next_day"
TARGET_SMOOTH_WINDOW = 3
LAGS = 30         # For baseline models

# Training Config
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
BATCH_SIZE = 64
EPOCHS = 100
# fine tuning: try different learning rates for different models
LR = 1e-3
PATIENCE = 25
# fine tuning: min delta and min epochs for early stopping to avoid underfitting
MIN_DELTA = 1e-5
VAL_LOSS_SMOOTH_WINDOW = 3
MIN_EPOCHS = 20
TRAIN_LOG_EVERY = 1
RANDOM_SEED = 42

# Learning-rate scheduler config
SCHEDULER_TYPE = "plateau"  # options: "none", "plateau", "cosine"
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR = 1e-6

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def _build_report_session_name() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    safe_timestamp = timestamp.replace(".", "-")
    return f"run_{safe_timestamp}"


def _resolve_reports_dir() -> str:
    explicit_dir = os.environ.get("FYP_REPORTS_DIR")
    if explicit_dir:
        return explicit_dir

    if os.environ.get("FYP_REPORTS_DISABLE_SESSION_DIR", "").lower() in {"1", "true", "yes"}:
        return BASE_REPORTS_DIR

    sessions_root = os.path.join(BASE_REPORTS_DIR, "sessions")
    return os.path.join(sessions_root, _build_report_session_name())


REPORTS_DIR = _resolve_reports_dir()
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
