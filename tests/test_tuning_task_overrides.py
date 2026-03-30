import unittest

from src.tuning import main as tuning_main


class TuningTaskOverridesTests(unittest.TestCase):
    def test_parser_accepts_task_runtime_overrides(self):
        parser = tuning_main.build_parser()
        args = parser.parse_args(
            [
                "--model",
                "all",
                "--task-id",
                "spy_next5_volatility",
                "--target-mode",
                "next_volatility",
                "--target-smooth-window",
                "5",
                "--horizon",
                "1",
                "--data-source",
                "spy",
                "--epochs",
                "20",
                "--scheduler-type",
                "none",
                "--random-seed",
                "123",
                "--dry-run",
            ]
        )

        self.assertEqual(args.task_id, "spy_next5_volatility")
        self.assertEqual(args.target_mode, "next_volatility")
        self.assertEqual(args.target_smooth_window, 5)
        self.assertEqual(args.horizon, 1)
        self.assertEqual(args.data_source, "spy")
        self.assertEqual(args.epochs, 20)
        self.assertEqual(args.scheduler_type, "none")
        self.assertEqual(args.random_seed, 123)

    def test_runtime_overrides_are_applied_into_candidate_config(self):
        runtime = tuning_main._config_to_runtime_dict(  # noqa: SLF001
            "gru",
            {"lr": 1e-4, "hidden": 32, "layers": 1, "batch_size": 16},
            "unit_test",
            runtime_overrides={
                "task_id": "sine_next_day",
                "target_mode": "sine_next_day",
                "target_smooth_window": 1,
                "horizon": 1,
                "data_source": "sine",
                "random_seed": 7,
            },
        )

        self.assertEqual(runtime["task_id"], "sine_next_day")
        self.assertEqual(runtime["target_mode"], "sine_next_day")
        self.assertEqual(runtime["target_smooth_window"], 1)
        self.assertEqual(runtime["horizon"], 1)
        self.assertEqual(runtime["data_source"], "sine")
        self.assertEqual(runtime["random_seed"], 7)


if __name__ == "__main__":
    unittest.main()
