import importlib
import sys
import types
import unittest
from unittest import mock


def _load_train_module_with_stubs():
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "0.0-test"
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    fake_torch.device = lambda name: name

    fake_nn = types.ModuleType("torch.nn")
    fake_nn.Module = object
    fake_torch.nn = fake_nn

    fake_pyplot = types.ModuleType("matplotlib.pyplot")
    fake_pyplot.figure = lambda *args, **kwargs: object()
    fake_pyplot.plot = lambda *args, **kwargs: None
    fake_pyplot.title = lambda *args, **kwargs: None
    fake_pyplot.xlabel = lambda *args, **kwargs: None
    fake_pyplot.ylabel = lambda *args, **kwargs: None
    fake_pyplot.legend = lambda *args, **kwargs: None
    fake_pyplot.grid = lambda *args, **kwargs: None
    fake_pyplot.savefig = lambda *args, **kwargs: None
    fake_pyplot.close = lambda *args, **kwargs: None
    fake_pyplot.scatter = lambda *args, **kwargs: None
    fake_pyplot.xticks = lambda *args, **kwargs: None

    fake_matplotlib = types.ModuleType("matplotlib")
    fake_matplotlib.pyplot = fake_pyplot


    fake_numpy = types.ModuleType("numpy")
    fake_numpy.mean = lambda values: sum(values) / len(values) if values else 0.0
    fake_numpy.isfinite = lambda value: True

    fake_metrics = types.ModuleType("src.common.metrics")
    fake_metrics.evaluate_preds = lambda y_true, y_pred: {"MAE": 0.0, "MSE": 0.0, "DA": 0.0}

    fake_reporting = types.ModuleType("src.common.reporting")
    fake_reporting.append_experiment_record = lambda record: None
    fake_reporting.build_experiment_record = lambda **kwargs: kwargs

    with mock.patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "torch.nn": fake_nn,
            "matplotlib": fake_matplotlib,
            "matplotlib.pyplot": fake_pyplot,
            "numpy": fake_numpy,
            "src.common.metrics": fake_metrics,
            "src.common.reporting": fake_reporting,
        },
    ):
        sys.modules.pop("src.common.train", None)
        return importlib.import_module("src.common.train")


class TrainFigureArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_module = _load_train_module_with_stubs()

    def test_build_figure_stem_includes_run_id_to_avoid_overwrite(self):
        stem = self.train_module._build_figure_stem("LSTM", "lstm_experiment-20260319T090151Z")

        self.assertEqual(stem, "LSTM_lstm_experiment-20260319T090151Z")

    def test_format_plot_hyperparameters_includes_training_and_model_values(self):
        text = self.train_module._format_plot_hyperparameters(
            {"hidden": 64, "layers": 2},
            {"learning_rate": 0.001, "batch_size": 32},
        )

        self.assertIn("hidden=64", text)
        self.assertIn("layers=2", text)
        self.assertIn("learning_rate=0.001", text)
        self.assertIn("batch_size=32", text)
        self.assertTrue(text.startswith("Hyperparameters: "))


if __name__ == "__main__":
    unittest.main()
