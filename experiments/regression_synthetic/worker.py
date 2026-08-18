"""Обучение регрессии через directory-backed component protocols."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from context_detection.components import ComponentKind
from context_detection.config import ExperimentConfig, OptimizerName
from context_detection.experiment import ExperimentComponents, ExperimentRun


class RegressionExperiment:
    """Worker, который получает готовые DataLoader и nn.Module endpoints."""

    def __init__(
        self,
        experiment: ExperimentRun,
        config: ExperimentConfig,
        components: ExperimentComponents,
    ) -> None:
        self.experiment: ExperimentRun = experiment
        self.config: ExperimentConfig = config
        self.components: ExperimentComponents = components
        self.model: nn.Module = components.model
        self.train_loader: DataLoader[Any] = components.loader("train")
        self.validation_loader: DataLoader[Any] = components.loader("validation")
        self.criterion: nn.MSELoss = nn.MSELoss()

    def run(self) -> dict[str, Any]:
        """Обучить модель, оставить artifact в её папке и вернуть summary."""
        optimizer: torch.optim.Optimizer = self._make_optimizer()
        for epoch in range(1, self.config.train.epochs + 1):
            train_mse: float = self._train_epoch(optimizer)
            should_log: bool = (
                epoch == 1
                or epoch == self.config.train.epochs
                or epoch % self.config.logging.every_n_steps == 0
            )
            if should_log:
                validation_mse: float = self._evaluate()
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

        final_mse: float = self._evaluate()
        model_artifacts: Path = self.components.artifacts(ComponentKind.MODEL)
        checkpoint_path: Path = model_artifacts / "model.pt"
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "train_config": self.config.train.model_dump(mode="json"),
            },
            checkpoint_path,
        )

        linear: nn.Linear = self._linear_layer()
        dataset_config: DictConfig = OmegaConf.load(self.config.data.config_path)
        return {
            "validation_mse": final_mse,
            "learned_slope": float(linear.weight.item()),
            "learned_intercept": float(linear.bias.item()),
            "target_slope": float(dataset_config.slope),
            "target_intercept": float(dataset_config.intercept),
            "samples": int(dataset_config.samples),
            "dataset_endpoint": type(self.train_loader).__name__,
            "model_endpoint": type(self.model).__name__,
        }

    def _make_optimizer(self) -> torch.optim.Optimizer:
        parameters = self.model.parameters()
        if self.config.train.optimizer is OptimizerName.SGD:
            return torch.optim.SGD(parameters, lr=self.config.train.lr)
        return torch.optim.AdamW(
            parameters,
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay,
        )

    def _train_epoch(self, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        total_loss: float = 0.0
        total_samples: int = 0
        for inputs, targets in self.train_loader:
            predictions: torch.Tensor = self.model(inputs)
            loss: torch.Tensor = self.criterion(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size: int = inputs.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
        return total_loss / total_samples

    def _evaluate(self) -> float:
        self.model.eval()
        total_loss: float = 0.0
        total_samples: int = 0
        with torch.no_grad():
            for inputs, targets in self.validation_loader:
                predictions: torch.Tensor = self.model(inputs)
                batch_size: int = inputs.shape[0]
                total_loss += float(self.criterion(predictions, targets).item()) * (
                    batch_size
                )
                total_samples += batch_size
        return total_loss / total_samples

    def _linear_layer(self) -> nn.Linear:
        linear: Any = getattr(self.model, "linear", None)
        if not isinstance(linear, nn.Linear):
            raise TypeError("linear_regression model обязан содержать nn.Linear")
        return linear


def run_regression(
    experiment: ExperimentRun,
    config: ExperimentConfig,
    components: ExperimentComponents,
) -> dict[str, Any]:
    """Запустить синтетическую линейную регрессию."""
    return RegressionExperiment(experiment, config, components).run()
