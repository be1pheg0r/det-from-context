"""Быстрые конфигурационные проверки MeMOT без импорта torch."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from context_detection.config import ExperimentConfig


def test_memot_rejects_short_history_larger_than_memory() -> None:
    with pytest.raises(ValidationError, match="short_memory_length"):
        ExperimentConfig.model_validate(
            {
                "data": {"clip_len": 2},
                "context": {
                    "name": "memot",
                    "memory_length": 2,
                    "short_memory_length": 3,
                },
            }
        )


def test_memot_lifecycle_parameters_are_bounded() -> None:
    with pytest.raises(ValidationError, match="motion_momentum"):
        ExperimentConfig.model_validate(
            {
                "data": {"clip_len": 2},
                "context": {"name": "memot", "motion_momentum": 1.0},
            }
        )


def test_memot_config_exposes_association_and_lifecycle() -> None:
    config = ExperimentConfig.model_validate(
        {
            "data": {"clip_len": 2},
            "context": {
                "name": "memot",
                "max_missed": 7,
                "association_iou_threshold": 0.2,
                "association_cosine_threshold": 0.6,
                "association_appearance_weight": 0.3,
                "motion_momentum": 0.75,
            },
        }
    )
    assert config.context.max_missed == 7
    assert config.context.association_iou_threshold == 0.2
    assert config.context.association_cosine_threshold == 0.6
    assert config.context.association_appearance_weight == 0.3
    assert config.context.motion_momentum == 0.75
