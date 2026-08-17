"""Обучение линейной регрессии для сквозной проверки протокола."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from context_detection.config import ExperimentConfig, OptimizerName
from context_detection.experiment import ExperimentRun


class RegressionDatasetConfig(BaseModel):
    """Строгая схема синтетического regression-датасета."""

    model_config = ConfigDict(extra="forbid")

    name: str
    samples: int = Field(gt=4)
    x_min: float
    x_max: float
    slope: float
    intercept: float
    noise_std: float = Field(ge=0.0)
    validation_fraction: float = Field(gt=0.0, lt=0.5)


class LinearRegressor(nn.Module):
    """Одномерная линейная модель ``y = wx + b``."""

    def __init__(self) -> None:
        super().__init__()
        self.linear: nn.Linear = nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Вычислить прогноз для батча признаков."""
        return self.linear(inputs)


class RegressionExperiment:
    """Конкретный worker, использующий общий lifecycle эксперимента."""

    def __init__(
        self,
        experiment: ExperimentRun,
        config: ExperimentConfig,
        project_root: Path,
    ) -> None:
        self.experiment: ExperimentRun = experiment
        self.config: ExperimentConfig = config
        self.project_root: Path = project_root
        self.dataset_config: RegressionDatasetConfig = self._load_dataset_config()
        torch.manual_seed(config.train.seed)
        self.generator: torch.Generator = torch.Generator().manual_seed(
            config.train.seed
        )
        self.model: LinearRegressor = LinearRegressor()
        self.criterion: nn.MSELoss = nn.MSELoss()

    def run(self) -> dict[str, Any]:
        """Обучить модель, сохранить веса и вернуть итоговые показатели."""
        train_x, train_y, validation_x, validation_y = self._make_dataset()
        optimizer: torch.optim.Optimizer = self._make_optimizer()

        for epoch in range(1, self.config.train.epochs + 1):
            train_mse: float = self._train_epoch(train_x, train_y, optimizer)
            should_log: bool = (
                epoch == 1
                or epoch == self.config.train.epochs
                or epoch % self.config.logging.every_n_steps == 0
            )
            if should_log:
                validation_mse: float = self._evaluate(validation_x, validation_y)
                self.experiment.log_metrics(
                    {"mse": train_mse},
                    step=epoch,
                    split="train",
                )
                self.experiment.log_metrics(
                    {"mse": validation_mse},
                    step=epoch,
                    split="validation",
                )

        final_mse: float = self._evaluate(validation_x, validation_y)
        checkpoint_path: Path = self.experiment.checkpoints_dir / "model.pt"
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "train_config": self.config.train.model_dump(mode="json"),
                "dataset_config": self.dataset_config.model_dump(mode="json"),
            },
            checkpoint_path,
        )
        self.experiment.save_artifact("model.pt", checkpoint_path)

        learned_slope: float = float(self.model.linear.weight.item())
        learned_intercept: float = float(self.model.linear.bias.item())
        return {
            "validation_mse": final_mse,
            "learned_slope": learned_slope,
            "learned_intercept": learned_intercept,
            "target_slope": self.dataset_config.slope,
            "target_intercept": self.dataset_config.intercept,
            "samples": self.dataset_config.samples,
        }

    def _load_dataset_config(self) -> RegressionDatasetConfig:
        path: Path = self.project_root / self.config.data.config_path
        raw: Any = OmegaConf.to_container(
            OmegaConf.load(path),
            resolve=True,
        )
        return RegressionDatasetConfig.model_validate(raw)

    def _make_dataset(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dataset: RegressionDatasetConfig = self.dataset_config
        inputs: torch.Tensor = torch.linspace(
            dataset.x_min,
            dataset.x_max,
            dataset.samples,
        ).unsqueeze(1)
        noise: torch.Tensor = (
            torch.randn(
                inputs.shape,
                generator=self.generator,
            )
            * dataset.noise_std
        )
        targets: torch.Tensor = inputs * dataset.slope + dataset.intercept + noise
        indices: torch.Tensor = torch.randperm(
            dataset.samples,
            generator=self.generator,
        )
        validation_size: int = round(dataset.samples * dataset.validation_fraction)
        validation_indices: torch.Tensor = indices[:validation_size]
        train_indices: torch.Tensor = indices[validation_size:]
        return (
            inputs[train_indices],
            targets[train_indices],
            inputs[validation_indices],
            targets[validation_indices],
        )

    def _make_optimizer(self) -> torch.optim.Optimizer:
        parameters = self.model.parameters()
        if self.config.train.optimizer is OptimizerName.SGD:
            return torch.optim.SGD(parameters, lr=self.config.train.lr)
        return torch.optim.AdamW(
            parameters,
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay,
        )

    def _train_epoch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.model.train()
        order: torch.Tensor = torch.randperm(
            inputs.shape[0],
            generator=self.generator,
        )
        total_loss: float = 0.0
        for start in range(0, inputs.shape[0], self.config.train.batch_size):
            batch_indices: torch.Tensor = order[
                start : start + self.config.train.batch_size
            ]
            predictions: torch.Tensor = self.model(inputs[batch_indices])
            loss: torch.Tensor = self.criterion(predictions, targets[batch_indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch_indices.numel()
        return total_loss / inputs.shape[0]

    def _evaluate(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            predictions: torch.Tensor = self.model(inputs)
            return float(self.criterion(predictions, targets).item())


def run_regression(
    experiment: ExperimentRun,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Запустить синтетическую линейную регрессию."""
    project_root: Path = Path(__file__).resolve().parents[2]
    return RegressionExperiment(experiment, config, project_root).run()
