import argparse
from typing import Any, Optional

from .experiment import PreparedSequenceRun, prepare_sequence_experiment_run
from .runtime_config import RuntimeTrainingConfig, add_runtime_config_args


def build_runtime_parser(model_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run the {model_name} training experiment.")
    add_runtime_config_args(parser)
    return parser


def resolve_runtime_config(
    *,
    config: Optional[RuntimeTrainingConfig] = None,
    config_dict: Optional[dict[str, Any]] = None,
    cli_args: Optional[argparse.Namespace] = None,
    **overrides: Any,
) -> RuntimeTrainingConfig:
    return RuntimeTrainingConfig.from_sources(
        config=config,
        config_dict=config_dict,
        cli_args=cli_args,
        **overrides,
    )


def prepare_sequence_training_data(batch_size: int) -> tuple[Any, Any, tuple[Any, Any], Any, dict[str, Any]]:
    prepared_run = prepare_sequence_experiment_run(
        batch_size=batch_size,
        experiment_name="sequence_training",
        random_seed=RuntimeTrainingConfig().random_seed,
    )
    return (
        prepared_run.train_loader,
        prepared_run.val_loader,
        prepared_run.test_data,
        prepared_run.test_index,
        prepared_run.split_metadata,
    )


def prepare_runtime_sequence_run(
    *,
    experiment_name: str,
    batch_size: int,
    run_note: str | None = None,
    training_metadata: dict[str, Any] | None = None,
    seq_len: int,
    horizon: int,
    data_source: str,
    target_mode: str,
    target_smooth_window: int,
    task_id: str | None,
    random_seed: int,
) -> PreparedSequenceRun:
    return prepare_sequence_experiment_run(
        batch_size=batch_size,
        experiment_name=experiment_name,
        run_note=run_note,
        training_metadata=training_metadata,
        seq_len=seq_len,
        horizon=horizon,
        data_source=data_source,
        target_mode=target_mode,
        target_smooth_window=target_smooth_window,
        task_id=task_id,
        random_seed=random_seed,
    )
