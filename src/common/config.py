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
LR = 1e-3
PATIENCE = 10
MIN_EPOCHS = 20

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
