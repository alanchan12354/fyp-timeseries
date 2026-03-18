import os
import torch

# Data Config
TICKER = "SPY"
START = "2010-01-01"

# Task Config
SEQ_LEN = 30      # Lookback window
HORIZON = 10      # Prediction horizon
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

# Learning-rate scheduler config
SCHEDULER_TYPE = "plateau"  # options: "none", "plateau", "cosine"
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR = 1e-6

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
