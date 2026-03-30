import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.modules.setdefault("torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))

from src.common.runtime_config import RuntimeTrainingConfig
from src.comparison.best_configs import BestConfigError, load_best_configs
from src.comparison.best_tuned_main import build_markdown_report, build_report_row, run_best_tuned_comparison

from src.comparison.best_tuned_charts import (
    BestTunedChartError,
    ComparisonRow,
    build_svg_chart,
    generate_best_tuned_svg_charts,
)


class BestConfigsTests(unittest.TestCase):
    def test_load_best_configs_from_winners_normalizes_each_model(self):
        csv_text = """model,stage_index,stage_name,parameter_group,winning_value_json,best_val_MSE,frozen_config_json
lstm,1,LSTM lr sweep,lr,0.0005,0.1,"{""batch_size"": 32, ""hidden"": 128, ""layers"": 1, ""lr"": 0.0005}"
gru,1,GRU lr sweep,lr,0.001,0.2,"{""batch_size"": 64, ""hidden"": 32, ""layers"": 3, ""lr"": 0.001}"
rnn,1,RNN lr sweep,lr,0.0001,0.3,"{""batch_size"": 32, ""hidden"": 64, ""layers"": 2, ""lr"": 0.0001}"
transformer,1,Transformer lr sweep,lr,0.001,0.4,"{""batch_size"": 32, ""d_model"": 32, ""lr"": 0.001, ""nhead"": 8, ""num_layers"": 1}"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tuning_winners.csv"
            path.write_text(csv_text, encoding="utf-8")

            selection = load_best_configs("tuning_winners", source_path=path)

        self.assertEqual(selection.source_key, "tuning_winners")
        self.assertEqual(selection.configs["lstm"], {"batch_size": 32, "hidden": 128, "layers": 1, "lr": 0.0005})
        self.assertEqual(selection.configs["transformer"], {"batch_size": 32, "d_model": 32, "lr": 0.001, "nhead": 8, "num_layers": 1})

    def test_load_best_configs_from_best_configs_backfills_lr_and_batch_size_from_notes(self):
        csv_text = """model,run_id,timestamp,hyperparameters,hidden_size,d_model,nhead,num_layers,best_val_MSE,test_MSE,MAE,DA,best_epoch,notes,selection_reason
LSTM,lstm-run,2026-03-19T09:02:41+00:00,"{""hidden"":64,""layers"":2}",64,,,2,0.1,0.2,0.3,0.4,5,model=LSTM;hidden=64;layers=2;lr=0.0005;batch=32,selected
GRU,gru-run,2026-03-19T09:02:41+00:00,"{""hidden"":32,""layers"":3}",32,,,3,0.11,0.2,0.3,0.4,5,model=GRU;hidden=32;layers=3;lr=0.001;batch=64,selected
RNN,rnn-run,2026-03-19T09:02:41+00:00,"{""hidden"":64,""layers"":2}",64,,,2,0.12,0.2,0.3,0.4,5,model=RNN;hidden=64;layers=2;lr=0.0001;batch=32,selected
Transformer,transformer-run,2026-03-19T09:02:41+00:00,"{""d_model"":32,""dropout"":0.1,""nhead"":8,""num_layers"":1}",,32,8,1,0.13,0.2,0.3,0.4,5,model=Transformer;d_model=32;layers=1;nhead=8;lr=0.001;batch=64,selected
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tuning_best_configs.csv"
            path.write_text(csv_text, encoding="utf-8")

            selection = load_best_configs("tuning_best_configs", source_path=path)

        self.assertEqual(selection.configs["lstm"]["lr"], 0.0005)
        self.assertEqual(selection.configs["lstm"]["batch_size"], 32)
        self.assertEqual(selection.configs["transformer"]["num_layers"], 1)
        self.assertEqual(selection.configs["transformer"]["nhead"], 8)

    def test_load_best_configs_raises_for_missing_incomplete_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.csv"
            with self.assertRaisesRegex(BestConfigError, "not found"):
                load_best_configs("tuning_winners", source_path=missing)

            incomplete = Path(tmpdir) / "tuning_winners.csv"
            incomplete.write_text(
                "model,stage_index,stage_name,parameter_group,winning_value_json,best_val_MSE,frozen_config_json\n"
                'lstm,1,LSTM lr sweep,lr,0.0005,0.1,"{""hidden"": 64, ""layers"": 2, ""lr"": 0.0005}"\n'
                'gru,1,GRU lr sweep,lr,0.001,0.2,"{""batch_size"": 64, ""hidden"": 32, ""layers"": 3, ""lr"": 0.001}"\n'
                'rnn,1,RNN lr sweep,lr,0.0001,0.3,"{""batch_size"": 32, ""hidden"": 64, ""layers"": 2, ""lr"": 0.0001}"\n'
                'transformer,1,Transformer lr sweep,lr,0.001,0.4,"{""batch_size"": 32, ""d_model"": 32, ""lr"": 0.001, ""nhead"": 8, ""num_layers"": 1}"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(BestConfigError, "missing batch_size"):
                load_best_configs("tuning_winners", source_path=incomplete)

    def test_runtime_config_aliases_accept_normalized_best_configs(self):
        recurrent_runtime = RuntimeTrainingConfig.from_sources(
            config_dict={"lr": 0.0005, "batch_size": 32, "hidden": 128, "layers": 3}
        )
        transformer_runtime = RuntimeTrainingConfig.from_sources(
            config_dict={"lr": 0.001, "batch_size": 64, "d_model": 32, "num_layers": 1, "nhead": 8}
        )

        self.assertEqual(recurrent_runtime.learning_rate, 0.0005)
        self.assertEqual(recurrent_runtime.recurrent_hidden_size, 128)
        self.assertEqual(recurrent_runtime.recurrent_layer_count, 3)
        self.assertEqual(transformer_runtime.transformer_d_model, 32)
        self.assertEqual(transformer_runtime.transformer_num_layers, 1)
        self.assertEqual(transformer_runtime.transformer_nhead, 8)


class BestTunedComparisonReportTests(unittest.TestCase):
    def test_build_report_row_contains_expected_fields(self):
        row = build_report_row(
            model_name="LSTM",
            config_source="tuning_winners.csv",
            tuned_config={"lr": 0.0005, "batch_size": 32, "hidden": 128, "layers": 1},
            metrics={"best_train_MSE": 0.05, "best_val_MSE": 0.1, "best_test_MSE": 0.2, "MSE": 0.2, "MAE": 0.3, "DA": 0.4},
            run_id="best_tuned_lstm_comparison-20260320T000000Z",
        )

        self.assertEqual(
            set(row),
            {"model", "tuned_hyperparameters", "best_train_MSE", "best_val_MSE", "best_test_MSE", "MSE", "MAE", "DA", "run_id", "config_source"},
        )
        self.assertEqual(row["config_source"], "tuning_winners.csv")

    def test_run_best_tuned_comparison_uses_each_model_config_and_generates_rows(self):
        selection = mock.Mock()
        selection.source_key = "tuning_winners"
        selection.source_path = Path("reports/tuning_winners.csv")
        selection.note = "Uses the final frozen staged winners from sequential tuning for each model."
        selection.configs = {
            "lstm": {"lr": 0.0005, "batch_size": 32, "hidden": 128, "layers": 1},
            "gru": {"lr": 0.001, "batch_size": 64, "hidden": 32, "layers": 3},
            "rnn": {"lr": 0.0001, "batch_size": 32, "hidden": 64, "layers": 2},
            "transformer": {"lr": 0.001, "batch_size": 64, "d_model": 32, "num_layers": 1, "nhead": 8},
        }

        prepared_runs = {model: mock.Mock(run_context={"run_id": f"{model}-run"}) for model in selection.configs}
        captured_calls = []

        def fake_run(entrypoint, *, config_dict, prepared_run):
            captured_calls.append((entrypoint, config_dict, prepared_run.run_context["run_id"]))
            return {"best_train_MSE": 0.05, "best_val_MSE": 0.1, "best_test_MSE": 0.2, "MSE": 0.2, "MAE": 0.3, "DA": 0.4}

        with mock.patch("src.comparison.best_tuned_main._prepare_runs", return_value=prepared_runs), mock.patch(
            "src.comparison.best_tuned_main._run_entrypoint", side_effect=fake_run
        ), mock.patch(
            "src.comparison.best_tuned_main._build_baseline_row",
            return_value={
                "model": "Baseline-LR",
                "tuned_hyperparameters": "{\"flattened_sequence\": true, \"model\": \"LinearRegression\"}",
                "best_train_MSE": 0.07,
                "best_val_MSE": 0.15,
                "best_test_MSE": 0.16,
                "MSE": 0.16,
                "MAE": 0.25,
                "DA": 0.45,
                "run_id": "baseline-run",
                "config_source": "tuning_winners.csv",
            },
        ):
            with mock.patch("src.comparison.best_tuned_main._load_entrypoints", return_value={model: mock.Mock() for model in selection.configs}):
                results = run_best_tuned_comparison(selection=selection)

        self.assertEqual([row["model"] for row in results], ["GRU", "LSTM", "RNN", "Transformer", "Baseline-LR"])
        self.assertEqual(len(captured_calls), 4)
        transformer_call = next(call for call in captured_calls if call[2] == "transformer-run")
        self.assertEqual(transformer_call[1]["d_model"], 32)
        self.assertEqual(transformer_call[1]["num_layers"], 1)
        self.assertEqual(transformer_call[1]["nhead"], 8)
        lstm_call = next(call for call in captured_calls if call[2] == "lstm-run")
        self.assertEqual(lstm_call[1]["hidden"], 128)
        self.assertEqual(lstm_call[1]["layers"], 1)
        self.assertIn("run_note", lstm_call[1])

    def test_build_markdown_report_summarizes_rankings_and_source_note(self):
        selection = mock.Mock(source_path=Path("reports/tuning_winners.csv"), note="Uses the final frozen staged winners from sequential tuning for each model.")
        results = [
                build_report_row(
                    model_name="LSTM",
                    config_source="tuning_winners.csv",
                    tuned_config={"lr": 0.0005, "batch_size": 32, "hidden": 128, "layers": 1},
                    metrics={"best_train_MSE": 0.08, "best_val_MSE": 0.1, "best_test_MSE": 0.3, "MSE": 0.3, "MAE": 0.2, "DA": 0.6},
                    run_id="lstm-run",
                ),
                build_report_row(
                    model_name="Baseline-LR",
                    config_source="tuning_winners.csv",
                    tuned_config={"model": "LinearRegression", "flattened_sequence": True},
                    metrics={"best_train_MSE": 0.09, "best_val_MSE": 0.15, "best_test_MSE": 0.11, "MSE": 0.11, "MAE": 0.25, "DA": 0.52},
                    run_id="baseline-run",
                ),
                build_report_row(
                    model_name="GRU",
                    config_source="tuning_winners.csv",
                    tuned_config={"lr": 0.001, "batch_size": 64, "hidden": 32, "layers": 3},
                    metrics={"best_train_MSE": 0.07, "best_val_MSE": 0.2, "best_test_MSE": 0.1, "MSE": 0.1, "MAE": 0.3, "DA": 0.5},
                    run_id="gru-run",
                ),
            ]

        markdown = build_markdown_report(results, selection=selection)

        self.assertIn("Best model by validation MSE: **LSTM**", markdown)
        self.assertIn("Best model by test MSE: **GRU**", markdown)
        self.assertIn("Baseline: shared flattened-sequence linear regression on the same split", markdown)
        self.assertIn("Train MSE", markdown)
        self.assertIn("Ranking by validation MSE", markdown)
        self.assertIn("This comparison uses the final frozen staged winners", markdown)


class BestTunedChartTests(unittest.TestCase):
    def test_generate_best_tuned_svg_charts_creates_expected_svg_files(self):
        csv_text = """model,tuned_hyperparameters,best_train_MSE,best_val_MSE,best_test_MSE,MSE,MAE,DA,run_id,config_source
LSTM,{},0.2,0.1,0.3,0.3,0.2,0.6,lstm-run,tuning_winners.csv
GRU,{},0.1,0.15,0.2,0.2,0.3,0.5,gru-run,tuning_winners.csv
Baseline-LR,{},0.05,0.12,0.11,0.11,0.25,0.52,baseline-run,tuning_winners.csv
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "best_tuned_comparison.csv"
            output_dir = Path(tmpdir) / "figures"
            csv_path.write_text(csv_text, encoding="utf-8")

            artifacts = generate_best_tuned_svg_charts(csv_path, output_dir=output_dir)

            self.assertEqual(len(artifacts), 3)
            self.assertEqual([artifact.metric for artifact in artifacts], ["best_train_MSE", "best_test_MSE", "best_val_MSE"])
            for artifact in artifacts:
                self.assertTrue(artifact.output_path.exists())
                self.assertEqual(artifact.output_path.suffix, ".svg")
                self.assertIn("<svg", artifact.output_path.read_text(encoding="utf-8"))

    def test_generate_best_tuned_svg_charts_raises_for_missing_metric_column(self):
        csv_text = """model,best_train_MSE,best_val_MSE
LSTM,0.1,0.2
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "best_tuned_comparison.csv"
            csv_path.write_text(csv_text, encoding="utf-8")

            with self.assertRaisesRegex(BestTunedChartError, "best_test_MSE"):
                generate_best_tuned_svg_charts(csv_path, output_dir=Path(tmpdir) / "figures")

    def test_build_svg_chart_uses_scientific_notation_for_tiny_values(self):
        rows = [
            ComparisonRow(model="Baseline-LR", metrics={"best_val_MSE": 2.7e-8}),
            ComparisonRow(model="GRU", metrics={"best_val_MSE": 8.0e-6}),
        ]

        svg = build_svg_chart(rows, metric="best_val_MSE", title="Validation Loss (MSE)", bar_color="#54A24B")

        self.assertIn("2.70e-08", svg)
        self.assertNotIn(">0.000000<", svg)


if __name__ == "__main__":
    unittest.main()
