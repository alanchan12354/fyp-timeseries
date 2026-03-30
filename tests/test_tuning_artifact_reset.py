import contextlib
import importlib
import io
import os
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from unittest import mock


class _FakeCuda:
    @staticmethod
    def is_available():
        return False


class _FakeTorch(types.SimpleNamespace):
    pass


def _install_fake_modules():
    fake_torch = _FakeTorch(cuda=_FakeCuda(), __version__="0.0-test")
    sys.modules.setdefault("torch", fake_torch)

    for module_name in ("src.gru.train", "src.lstm.train", "src.rnn.train", "src.transformer.train", "src.baseline_lr.train"):
        module = types.ModuleType(module_name)
        module.main = lambda **kwargs: {"best_val_MSE": 0.123}
        sys.modules[module_name] = module


class TuningArtifactResetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_fake_modules()
        cls.reporting = importlib.import_module("src.common.reporting")
        cls.tuning_main = importlib.import_module("src.tuning.main")

    def test_reset_tuning_artifacts_removes_expected_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            figures_dir = os.path.join(tmpdir, "figures")
            os.makedirs(figures_dir, exist_ok=True)
            targets = [
                "experiment_log.csv",
                "experiment_log.jsonl",
                "lstm_diagnostics.json",
                "model.pt",
                "tuning_runs.csv",
                "tuning_summary_latest.csv",
            ]
            for name in targets:
                with open(os.path.join(tmpdir, name), "w", encoding="utf-8") as handle:
                    handle.write("x")
            with open(os.path.join(figures_dir, "loss_lstm.png"), "w", encoding="utf-8") as handle:
                handle.write("x")
            with open(os.path.join(tmpdir, "keep.txt"), "w", encoding="utf-8") as handle:
                handle.write("keep")

            summary = self.reporting.reset_tuning_artifacts(reports_dir=tmpdir)

            self.assertEqual(summary["removed_count"], 7)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "keep.txt")))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "experiment_log.csv")))
            self.assertFalse(os.path.exists(os.path.join(figures_dir, "loss_lstm.png")))

    def test_main_reset_mode_reports_removed_files_before_run(self):
        args = Namespace(
            model="lstm",
            plan_file=None,
            plan_json=None,
            session_mode="reset",
            clear_outputs=False,
            keep_outputs=False,
            dry_run=False,
            horizon=None,
            data_source=None,
            target_mode=None,
            target_smooth_window=None,
            task_id=None,
            seq_len=None,
            epochs=None,
            scheduler_type=None,
            random_seed=None,
        )
        with mock.patch.object(self.tuning_main, "reset_tuning_artifacts", return_value={
            "reports_dir": "/tmp/reports",
            "removed_count": 2,
            "removed_files": ["/tmp/reports/experiment_log.csv", "/tmp/reports/figures/loss_lstm.png"],
        }) as reset_mock, mock.patch.object(self.tuning_main, "tune_model", return_value={"model": "lstm", "final_config": {}, "winners": []}):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                summaries = self.tuning_main.main(args)

        reset_mock.assert_called_once_with()
        self.assertEqual(len(summaries), 1)
        text = output.getvalue()
        self.assertIn("Session mode: reset", text)
        self.assertIn("removed experiment_log.csv", text)
        self.assertIn("removed figures/loss_lstm.png", text)

    def test_format_structured_notes_is_csv_friendly(self):
        note = self.reporting.format_structured_notes({
            "model": "LSTM",
            "stage": "lr_sweep",
            "candidate_param": "lr",
            "candidates": [1e-3, 5e-4, 1e-4],
            "fixed": {"hidden": 64, "layers": 2, "batch": 64},
            "selection": "best_val_MSE",
        })
        self.assertEqual(
            note,
            "model=LSTM;stage=lr_sweep;candidate_param=lr;candidates=0.001|0.0005|0.0001;fixed=batch:64|hidden:64|layers:2;selection=best_val_MSE",
        )

    def test_tune_model_writes_structured_notes_for_candidate_runs(self):
        captured_notes = []

        def _fake_train_entrypoint(*, config_dict=None, **kwargs):
            captured_notes.append(config_dict["run_note"])
            return {"best_val_MSE": 0.123}

        spec = self.tuning_main.MODEL_SPECS["lstm"]
        patched_spec = self.tuning_main.ModelSpec(
            cli_name=spec.cli_name,
            display_name=spec.display_name,
            train_entrypoint=_fake_train_entrypoint,
            baseline=spec.baseline,
            stage_order=["lr"],
            default_plan={"lr": [1e-3, 5e-4]},
        )

        with mock.patch.dict(self.tuning_main.MODEL_SPECS, {"lstm": patched_spec}, clear=False), mock.patch.object(
            self.tuning_main,
            "_append_csv",
        ):
            summary = self.tuning_main.tune_model("lstm", {"lr": [1e-3, 5e-4]})

        self.assertEqual(summary["final_config"]["lr"], 1e-3)
        self.assertEqual(
            captured_notes[0],
            "model=LSTM;stage=lr_sweep;candidate_param=lr;candidates=0.001|0.0005;fixed=batch:64|hidden:64|layers:2|seq_len:30;selection=best_val_MSE;hidden=64;layers=2;lr=0.001;batch=64;seq_len=30",
        )
        self.assertEqual(
            captured_notes[1],
            "model=LSTM;stage=lr_sweep;candidate_param=lr;candidates=0.001|0.0005;fixed=batch:64|hidden:64|layers:2|seq_len:30;selection=best_val_MSE;hidden=64;layers=2;lr=0.0005;batch=64;seq_len=30",
        )


if __name__ == "__main__":
    unittest.main()
