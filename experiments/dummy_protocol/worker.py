"""Минимальный worker для проверки протокола без реального датасета."""

from __future__ import annotations

from typing import Any

from context_detection.config import ExperimentConfig
from context_detection.experiment import ExperimentRun


def run_dummy(
    experiment: ExperimentRun,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Записать детерминированную smoke-метрику.

    Args:
        experiment: Активный запуск протокола.
        config: Полностью скомпонованный Hydra-конфиг.

    Returns:
        Итоговое резюме запуска.
    """
    experiment.log_metrics(
        {"protocol_ok": 1.0},
        step=0,
        split="smoke",
    )
    return {
        "protocol_ok": True,
        "seed": config.train.seed,
        "dataset_config": config.data.config_path,
    }
