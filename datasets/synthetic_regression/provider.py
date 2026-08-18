"""Self-contained dataset provider для синтетической регрессии."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from context_detection.data.protocols import DatasetSplit
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from context_detection.config import ExperimentConfig


class DatasetSettings(BaseModel):
    """Параметры генератора ``y = slope * x + intercept + noise``."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["synthetic_regression"]
    samples: int = Field(gt=4)
    x_min: float
    x_max: float
    slope: float
    intercept: float
    noise_std: float = Field(ge=0.0)
    validation_fraction: float = Field(gt=0.0, lt=0.5)

    @model_validator(mode="after")
    def _check_range(self) -> DatasetSettings:
        if self.x_min >= self.x_max:
            raise ValueError("x_min должен быть меньше x_max")
        return self


class SyntheticRegressionProtocol:
    """Строит DataLoader и сохраняет воспроизводимый tensor artifact split."""

    def build(
        self,
        config: ExperimentConfig,
        split: DatasetSplit,
    ) -> DataLoader[Any]:
        settings: DatasetSettings = self._load_settings(config.data.config_path)
        inputs, targets = self._generate(settings, split, config.train.seed)
        self._save_artifact(config, split, settings, inputs, targets)
        batch_size: int = (
            config.train.batch_size
            if split is DatasetSplit.TRAIN
            else config.validation.batch_size
        )
        generator: torch.Generator = torch.Generator().manual_seed(
            config.train.seed + 1
        )
        return DataLoader(
            TensorDataset(inputs, targets),
            batch_size=batch_size,
            shuffle=split is DatasetSplit.TRAIN,
            num_workers=config.train.num_workers,
            generator=generator,
        )

    @staticmethod
    def _load_settings(path: str | Path) -> DatasetSettings:
        raw: Any = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True)
        return DatasetSettings.model_validate(raw)

    @staticmethod
    def _generate(
        settings: DatasetSettings,
        split: DatasetSplit,
        seed: int,
    ) -> tuple[Tensor, Tensor]:
        generator: torch.Generator = torch.Generator().manual_seed(seed)
        inputs: Tensor = torch.linspace(
            settings.x_min,
            settings.x_max,
            settings.samples,
        ).unsqueeze(1)
        noise: Tensor = torch.randn(inputs.shape, generator=generator)
        targets: Tensor = (
            inputs * settings.slope + settings.intercept + noise * settings.noise_std
        )
        indices: Tensor = torch.randperm(settings.samples, generator=generator)
        validation_size: int = round(settings.samples * settings.validation_fraction)
        selected: Tensor = (
            indices[validation_size:]
            if split is DatasetSplit.TRAIN
            else indices[:validation_size]
        )
        return inputs[selected], targets[selected]

    @staticmethod
    def _save_artifact(
        config: ExperimentConfig,
        split: DatasetSplit,
        settings: DatasetSettings,
        inputs: Tensor,
        targets: Tensor,
    ) -> None:
        if config.data.component_path is None:
            raise ValueError("synthetic_regression требует data.component_path")
        artifact_path: Path = (
            Path(config.data.component_path) / "artifacts" / f"{split.value}.pt"
        )
        torch.save(
            {
                "inputs": inputs,
                "targets": targets,
                "settings": settings.model_dump(mode="json"),
                "seed": config.train.seed,
                "split": split.value,
            },
            artifact_path,
        )


PROTOCOL: SyntheticRegressionProtocol = SyntheticRegressionProtocol()
