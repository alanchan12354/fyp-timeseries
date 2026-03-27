import unittest

import numpy as np
import pandas as pd

from src.common.data import build_sequences


class AdditionalTargetModesTests(unittest.TestCase):
    def _feature_frame(self) -> pd.DataFrame:
        index = pd.date_range("2024-01-01", periods=12, freq="B")
        log_ret = np.arange(12, dtype=np.float64) / 100.0
        return pd.DataFrame(
            {
                "log_ret": log_ret,
                "oc_ret": log_ret * 0.8,
                "hl_range": np.full(12, 0.01),
                "vol_chg": log_ret * 0.2,
                "ma_5_gap": log_ret * 0.1,
                "ma_20_gap": log_ret * 0.05,
                "volatility_5": np.full(12, 0.005),
                "volatility_20": np.full(12, 0.007),
            },
            index=index,
        )

    def test_next_mean_return_matches_forward_window_mean(self):
        features = self._feature_frame()
        seq_len = 3
        window = 5

        _, y, _ = build_sequences(
            features,
            seq_len=seq_len,
            horizon=1,
            target_mode="next_mean_return",
            smooth_window=window,
        )

        expected_first = np.mean(features["log_ret"].values[3:8])
        self.assertAlmostEqual(y[0], expected_first)

    def test_next_volatility_matches_forward_window_std(self):
        features = self._feature_frame()
        seq_len = 3
        window = 5

        _, y, _ = build_sequences(
            features,
            seq_len=seq_len,
            horizon=1,
            target_mode="next_volatility",
            smooth_window=window,
        )

        expected_first = np.std(features["log_ret"].values[3:8], ddof=0)
        self.assertAlmostEqual(y[0], expected_first)

    def test_sine_next_day_alias_matches_next_return(self):
        features = self._feature_frame()

        _, y_alias, _ = build_sequences(
            features,
            seq_len=3,
            horizon=1,
            target_mode="sine_next_day",
            smooth_window=5,
        )
        _, y_next, _ = build_sequences(
            features,
            seq_len=3,
            horizon=1,
            target_mode="next_return",
            smooth_window=5,
        )

        np.testing.assert_allclose(y_alias, y_next)


if __name__ == "__main__":
    unittest.main()
