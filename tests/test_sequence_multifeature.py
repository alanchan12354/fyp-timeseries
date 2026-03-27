import unittest

import numpy as np
import pandas as pd

from src.common.data import build_sequences, chronological_split


class MultiFeatureSequenceTests(unittest.TestCase):
    def _feature_frame(self, periods: int = 80) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=periods, freq="D")
        base = np.linspace(100.0, 130.0, periods)
        close = pd.Series(base, index=idx)
        log_ret = np.log(close / close.shift(1))
        frame = pd.DataFrame(index=idx)
        frame["log_ret"] = log_ret
        frame["oc_ret"] = log_ret * 0.8
        frame["hl_range"] = 0.01 + (np.arange(periods) % 5) * 0.001
        frame["vol_chg"] = log_ret * 0.5
        frame["ma_5_gap"] = (close / close.rolling(5).mean()) - 1.0
        frame["ma_20_gap"] = (close / close.rolling(20).mean()) - 1.0
        frame["volatility_5"] = log_ret.rolling(5).std()
        frame["volatility_20"] = log_ret.rolling(20).std()
        return frame.dropna()

    def test_build_sequences_outputs_eight_channels_without_nans(self):
        features = self._feature_frame()
        X, y, idx = build_sequences(features, seq_len=10, horizon=1)

        self.assertEqual(X.shape[2], 8)
        self.assertFalse(np.isnan(X).any())
        self.assertFalse(np.isnan(y).any())
        self.assertEqual(len(idx), len(y))

    def test_chronological_split_sizes_follow_existing_ratio_rules(self):
        features = self._feature_frame()
        X, y, idx = build_sequences(features, seq_len=10, horizon=1)
        (X_tr, _, _), (X_va, _, _), (X_te, _, _) = chronological_split(X, y, idx, 0.7, 0.15)

        expected_train = int(len(X) * 0.7)
        expected_val = int(len(X) * 0.15)
        expected_test = len(X) - expected_train - expected_val

        self.assertEqual(len(X_tr), expected_train)
        self.assertEqual(len(X_va), expected_val)
        self.assertEqual(len(X_te), expected_test)


if __name__ == "__main__":
    unittest.main()
