import importlib.util
import unittest


HAS_BASELINE_TEST_DEPS = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("pandas", "yfinance", "torch")
)

if HAS_BASELINE_TEST_DEPS:
    import pandas as pd

    from src.common.data import make_lag_features


@unittest.skipUnless(HAS_BASELINE_TEST_DEPS, "baseline alignment test requires pandas, yfinance, and torch")
class BaselineAlignmentTests(unittest.TestCase):
    def test_make_lag_features_respects_configured_horizon(self):
        returns = pd.Series(
            [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            index=pd.date_range("2024-01-01", periods=6, freq="D"),
        )

        X, y, idx = make_lag_features(returns, lags=3, horizon=2)

        self.assertEqual(X.tolist(), [[0.03, 0.02, 0.01], [0.04, 0.03, 0.02]])
        self.assertEqual(y.tolist(), [0.05, 0.06])
        self.assertEqual(list(idx), list(returns.index[2:4]))


if __name__ == "__main__":
    unittest.main()
