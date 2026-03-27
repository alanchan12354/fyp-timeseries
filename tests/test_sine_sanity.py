import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.lstm.model import LSTMModel


class LSTMSineSanityTests(unittest.TestCase):
    """Sanity test: LSTM should quickly fit a tiny deterministic sine dataset."""

    SEED = 7

    @classmethod
    def setUpClass(cls):
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def _build_sine_sequences(seq_len: int = 20, n_points: int = 260):
        x = np.linspace(0, 16 * np.pi, n_points, dtype=np.float32)
        y = np.sin(x)

        X, targets = [], []
        for i in range(n_points - seq_len):
            X.append(y[i : i + seq_len, None])
            targets.append(y[i + seq_len])

        X = np.asarray(X, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        return X, targets

    @staticmethod
    def _mean_mse(model: torch.nn.Module, loader: DataLoader, loss_fn: torch.nn.Module) -> float:
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in loader:
                pred = model(xb)
                total_loss += float(loss_fn(pred, yb).item()) * len(xb)
        return total_loss / len(loader.dataset)

    def test_lstm_sine_sanity_small_deterministic_run(self):
        X, y = self._build_sine_sequences(seq_len=20, n_points=260)

        n_total = len(X)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)

        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
        X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
            batch_size=16,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.SEED),
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
            batch_size=64,
            shuffle=False,
        )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
            batch_size=64,
            shuffle=False,
        )

        model = LSTMModel(hidden=16, layers=1, input_size=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = torch.nn.MSELoss()

        for _ in range(15):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()

        val_mse = self._mean_mse(model, val_loader, loss_fn)
        test_mse = self._mean_mse(model, test_loader, loss_fn)

        threshold = 0.03
        failure_message = (
            "LSTM sine sanity check failed. This usually indicates a data pipeline mismatch, "
            "target misalignment, tensor shape issue, or a training loop regression. "
            f"Observed val_mse={val_mse:.6f}, test_mse={test_mse:.6f}, expected <= {threshold:.6f}."
        )

        self.assertLessEqual(val_mse, threshold, failure_message)
        self.assertLessEqual(test_mse, threshold, failure_message)


if __name__ == "__main__":
    unittest.main()
