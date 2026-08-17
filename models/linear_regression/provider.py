"""Self-contained model provider для одномерной линейной регрессии."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from context_detection.config import ExperimentConfig


class ModelSettings(BaseModel):
    """Архитектурный конфиг линейной модели."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["linear_regression"]
    in_features: int = Field(gt=0)
    out_features: int = Field(gt=0)
    bias: bool = True


class LinearRegressor(nn.Module):
    """Линейный regression endpoint."""

    def __init__(self, settings: ModelSettings) -> None:
        super().__init__()
        self.linear: nn.Linear = nn.Linear(
            settings.in_features,
            settings.out_features,
            bias=settings.bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class LinearRegressionProtocol:
    """Строит nn.Module из config.yaml этой component directory."""

    def build(self, config: ExperimentConfig) -> nn.Module:
        if config.detector.config_path is None:
            raise ValueError("linear_regression требует detector.config_path")
        raw: Any = OmegaConf.to_container(
            OmegaConf.load(Path(config.detector.config_path)),
            resolve=True,
        )
        settings: ModelSettings = ModelSettings.model_validate(raw)
        return LinearRegressor(settings)


PROTOCOL: LinearRegressionProtocol = LinearRegressionProtocol()
