import unittest
import types
import sys

sys.modules.setdefault("torch", types.SimpleNamespace(__version__="0.0-test", cuda=types.SimpleNamespace(is_available=lambda: False), backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))))
from src.common.reporting import build_experiment_record, create_run_context, default_task_metadata, deterministic_task_id
from src.common.runtime_config import RuntimeTrainingConfig


class TaskIdPropagationTests(unittest.TestCase):
    def test_runtime_config_generates_deterministic_task_id_when_missing(self):
        runtime = RuntimeTrainingConfig.from_sources(
            config_dict={
                "data_source": "sine",
                "horizon": 5,
                "target_mode": "next_return",
                "target_smooth_window": 7,
            }
        )

        self.assertEqual(runtime.task_id, "sine_h5_next_return_sw7")
        self.assertEqual(runtime.training_metadata()["task_id"], "sine_h5_next_return_sw7")

    def test_default_task_metadata_includes_optional_group_and_version(self):
        meta = default_task_metadata(task_group="ablation", task_version="v2")

        self.assertIn("task_id", meta)
        self.assertEqual(meta["task_group"], "ablation")
        self.assertEqual(meta["task_version"], "v2")

    def test_build_experiment_record_exposes_task_id_top_level(self):
        task_id = deterministic_task_id(data_source="spy", horizon=1, target_mode="horizon_return", target_smooth_window=3)
        context = create_run_context(
            "unit_test",
            {"train_samples": 1, "val_samples": 1, "test_samples": 1},
            task_meta=default_task_metadata(task_id=task_id),
        )

        record = build_experiment_record(
            model_name="LSTM",
            record_type="neural_model",
            metrics={"MSE": 0.1, "MAE": 0.1, "DA": 0.5, "best_val_MSE": 0.1},
            context=context,
        )

        self.assertEqual(record["task_id"], task_id)
        self.assertEqual(record["task"]["task_id"], task_id)


if __name__ == "__main__":
    unittest.main()
