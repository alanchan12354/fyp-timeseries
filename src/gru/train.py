import json
import os

from src.common.config import REPORTS_DIR
from src.common.neural_entrypoint import (
    build_runtime_parser,
    prepare_runtime_sequence_run,
    resolve_runtime_config,
)
from src.common.train import train_model

from .model import GRUModel


def main(config=None, config_dict=None, cli_args=None, prepared_run=None, **overrides):
    print("Running GRU Experiment...")

    runtime_config = resolve_runtime_config(
        config=config,
        config_dict=config_dict,
        cli_args=cli_args,
        **overrides,
    )
    prepared_run = prepared_run or prepare_runtime_sequence_run(
        experiment_name="gru_experiment",
        batch_size=runtime_config.batch_size,
        run_note=runtime_config.run_note or "Single-model training run for GRU.",
        training_metadata=runtime_config.training_metadata(),
        horizon=runtime_config.horizon,
        data_source=runtime_config.data_source,
        target_mode=runtime_config.target_mode,
        target_smooth_window=runtime_config.target_smooth_window,
        task_id=runtime_config.task_id,
        random_seed=runtime_config.random_seed,
    )

    metrics = train_model(
        "GRU",
        GRUModel,
        prepared_run.train_loader,
        prepared_run.val_loader,
        prepared_run.test_data,
        prepared_run.test_index,
        experiment_context=prepared_run.run_context,
        tuning_notes=runtime_config.run_note or "Runtime-configurable single-model training run.",
        learning_rate=runtime_config.learning_rate,
        epochs=runtime_config.epochs,
        scheduler_type=runtime_config.scheduler_type,
        random_seed=runtime_config.random_seed,
        **runtime_config.recurrent_model_kwargs(),
    )

    print(metrics)
    with open(os.path.join(REPORTS_DIR, "metrics_gru.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = build_runtime_parser("GRU")
    args = parser.parse_args()
    main(cli_args=args)
