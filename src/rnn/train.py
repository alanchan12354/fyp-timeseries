import json
import os

from src.common.config import REPORTS_DIR
from src.common.neural_entrypoint import (
    build_runtime_parser,
    prepare_sequence_training_data,
    resolve_runtime_config,
)
from src.common.reporting import create_run_context, default_training_metadata
from src.common.train import train_model

from .model import RNNModel


def main(config=None, config_dict=None, cli_args=None, **overrides):
    print("Running RNN Experiment...")

    runtime_config = resolve_runtime_config(
        config=config,
        config_dict=config_dict,
        cli_args=cli_args,
        **overrides,
    )
    tr_load, va_load, test_data, idx_te, split_meta = prepare_sequence_training_data(
        runtime_config.batch_size
    )

    run_context = create_run_context(
        "rnn_experiment",
        split_meta,
        training_meta=default_training_metadata(**runtime_config.training_metadata()),
        notes=runtime_config.run_note or "Single-model training run for RNN.",
    )

    metrics = train_model(
        "RNN",
        RNNModel,
        tr_load,
        va_load,
        test_data,
        idx_te,
        experiment_context=run_context,
        tuning_notes=runtime_config.run_note or "Runtime-configurable single-model training run.",
        learning_rate=runtime_config.learning_rate,
        **runtime_config.recurrent_model_kwargs(),
    )

    print(metrics)
    with open(os.path.join(REPORTS_DIR, "metrics_rnn.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = build_runtime_parser("RNN")
    args = parser.parse_args()
    main(cli_args=args)
