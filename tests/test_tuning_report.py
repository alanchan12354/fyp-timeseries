import unittest

from src.tuning import report


class TuningReportTests(unittest.TestCase):
    def test_collect_best_runs_keeps_lowest_validation_loss_per_model(self):
        rows = [
            {
                "model_name": "LSTM",
                "record_type": "neural_model",
                "notes_metadata": {"stage": "lr_sweep"},
                "metrics": {"best_val_MSE": 0.2, "best_train_MSE": 0.1, "best_test_MSE": 0.3, "MAE": 0.1, "DA": 0.5},
            },
            {
                "model_name": "LSTM",
                "record_type": "neural_model",
                "notes_metadata": {"stage": "hidden_sweep"},
                "metrics": {"best_val_MSE": 0.1, "best_train_MSE": 0.08, "best_test_MSE": 0.2, "MAE": 0.09, "DA": 0.6},
            },
        ]

        best = report._collect_best_runs(rows)

        self.assertEqual(best["LSTM"]["metrics"]["best_val_MSE"], 0.1)

    def test_collect_stage_impacts_describes_improvement_direction(self):
        rows = [
            {"model": "lstm", "stage_index": "1", "stage_name": "LSTM lr sweep", "parameter_group": "lr", "winning_value_json": "0.001", "best_val_MSE": "0.2"},
            {"model": "lstm", "stage_index": "2", "stage_name": "LSTM hidden sweep", "parameter_group": "hidden", "winning_value_json": "64", "best_val_MSE": "0.1"},
        ]

        impacts = report._collect_stage_impacts(rows)

        self.assertEqual(impacts["LSTM"][1]["change_from_previous"], "improved by 0.1")


if __name__ == "__main__":
    unittest.main()
