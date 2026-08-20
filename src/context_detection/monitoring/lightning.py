"""PyTorch Lightning integration for project logs, ClearML, and diagnostics."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import Tensor

from ..experiment import ExperimentRun
from .detection import (
    render_confidence_iou_diagnostics,
    render_confusion_matrix,
    render_dataset_diagnostics,
    render_metric_history,
    render_precision_recall,
    render_prediction_grid,
)


class MetricHistory:
    """In-memory scalar history shared by the Lightning logger and plot callback."""

    def __init__(self) -> None:
        self._values: dict[str, list[tuple[int, float]]] = defaultdict(list)

    def add(self, name: str, step: int, value: float) -> None:
        """Append or replace a scalar at one global step."""
        points = self._values[name]
        if points and points[-1][0] == step:
            points[-1] = (step, value)
        else:
            points.append((step, value))

    def snapshot(self) -> dict[str, list[tuple[int, float]]]:
        """Return a copy safe for plotting while training continues."""
        return {name: list(points) for name, points in self._values.items()}


class ExperimentLightningLogger(Logger):
    """Route native RF-DETR Lightning scalars through :class:`ExperimentRun`."""

    def __init__(
        self,
        run: ExperimentRun,
        history: MetricHistory | None = None,
    ) -> None:
        super().__init__()
        self.run = run
        self.history = history or MetricHistory()

    @property
    def name(self) -> str:
        """Return the experiment name used by Lightning."""
        return self.run.config.name

    @property
    def version(self) -> str:
        """Use the isolated run directory name as logger version."""
        return self.run.root.name

    @property
    def experiment(self) -> ExperimentRun:
        """Expose the project run as the underlying logger experiment."""
        return self.run

    @property
    def save_dir(self) -> str:
        """Return the local run root for Lightning-generated files."""
        return str(self.run.root)

    @rank_zero_only
    def log_hyperparams(self, params: Any) -> None:
        """Store resolved Lightning hyperparameters without secrets."""
        if hasattr(params, "items"):
            normalized = dict(params.items())
        elif hasattr(params, "__dict__"):
            normalized = dict(vars(params))
        else:
            normalized = {"value": str(params)}
        self.run.record_metadata("lightning_hyperparameters", normalized)

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        """Group finite Lightning scalars by split and report them everywhere."""
        global_step = int(step or 0)
        grouped: dict[str, dict[str, float]] = defaultdict(dict)
        for raw_name, raw_value in metrics.items():
            if raw_name in {"epoch", "step"}:
                continue
            value = _finite_scalar(raw_value)
            if value is None:
                continue
            split, name = _metric_parts(raw_name)
            grouped[split][name] = value
            self.history.add(f"{split}/{name}", global_step, value)
        for split, values in grouped.items():
            if values:
                self.run.log_metrics(values, step=global_step, split=split)

    @rank_zero_only
    def log_runtime(self, metrics: Mapping[str, float], step: int) -> None:
        """Record callback-derived timing, memory, and gradient diagnostics."""
        normalized = {name: float(value) for name, value in metrics.items()}
        for name, value in normalized.items():
            self.history.add(f"runtime/{name}", step, value)
        self.run.log_metrics(normalized, step=step, split="runtime")

    def save(self) -> None:
        """All values are persisted eagerly by :class:`ExperimentRun`."""

    @rank_zero_only
    def finalize(self, status: str) -> None:
        """Write Lightning's final status to the combined console/file log."""
        self.run.log_message(f"Lightning trainer finalized with status={status}")


class RFDetrMonitoringCallback(Callback):
    """Collect bounded RF-DETR runtime and validation diagnostics."""

    def __init__(
        self,
        run: ExperimentRun,
        logger: ExperimentLightningLogger,
        *,
        class_names: list[str],
        every_n_steps: int,
        visualize_every_n_epochs: int,
        max_visual_images: int,
        max_diagnostic_images: int,
        score_threshold: float,
    ) -> None:
        super().__init__()
        self.run = run
        self.logger = logger
        self.class_names = class_names
        self.every_n_steps = every_n_steps
        self.visualize_every_n_epochs = visualize_every_n_epochs
        self.max_visual_images = max_visual_images
        self.max_diagnostic_images = max_diagnostic_images
        self.score_threshold = score_threshold
        self._batch_started = 0.0
        self._images: list[Tensor] = []
        self._predictions: list[dict[str, Tensor]] = []
        self._targets: list[dict[str, Any]] = []

    @rank_zero_only
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Announce the full native stack and dataset sizes in live logs."""
        datamodule = trainer.datamodule
        self.run.log_message(
            "RF-DETR native fit started: "
            f"module={type(pl_module).__name__}, precision={trainer.precision}, "
            f"train_batches={trainer.num_training_batches}, "
            f"validation_batches={trainer.num_val_batches}"
        )
        if datamodule is not None:
            self.run.record_metadata(
                "dataset_runtime",
                {
                    "train_samples": len(datamodule.train_dataloader().dataset),
                    "validation_samples": len(datamodule.val_dataloader().dataset),
                    "class_count": len(self.class_names),
                },
            )

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Start wall-clock timing for one training batch."""
        del trainer, pl_module, batch, batch_idx
        self._batch_started = perf_counter()

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Report throughput, batch latency, and accelerator memory."""
        del pl_module, outputs
        if (batch_idx + 1) % self.every_n_steps:
            return
        elapsed = max(perf_counter() - self._batch_started, 1e-9)
        samples = batch[0].tensors.shape[0]
        metrics = {
            "batch_seconds": elapsed,
            "samples_per_second": samples / elapsed,
        }
        if torch.cuda.is_available():
            metrics.update(
                {
                    "gpu_allocated_gb": torch.cuda.memory_allocated() / 2**30,
                    "gpu_reserved_gb": torch.cuda.memory_reserved() / 2**30,
                    "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
                }
            )
        self.logger.log_runtime(metrics, int(trainer.global_step))

    @rank_zero_only
    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Report a global L2 gradient norm before the upstream optimizer step."""
        del optimizer
        squared = torch.zeros((), device=pl_module.device)
        for parameter in pl_module.parameters():
            if parameter.grad is not None:
                squared += parameter.grad.detach().float().square().sum()
        self.logger.log_runtime(
            {"grad_norm": float(squared.sqrt().cpu())}, int(trainer.global_step)
        )

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Reset bounded validation caches outside Lightning's sanity check."""
        del pl_module
        if trainer.sanity_checking:
            return
        self._images.clear()
        self._predictions.clear()
        self._targets.clear()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Cache CPU predictions/targets for bounded post-epoch diagnostics."""
        del pl_module, batch_idx, dataloader_idx
        if trainer.sanity_checking or not isinstance(outputs, Mapping):
            return
        results = outputs.get("results")
        targets = outputs.get("targets")
        if not isinstance(results, list) or not isinstance(targets, list):
            return
        samples = batch[0].tensors.detach().cpu()
        remaining = self.max_diagnostic_images - len(self._targets)
        for image, prediction, target in zip(
            samples[:remaining], results[:remaining], targets[:remaining], strict=True
        ):
            if len(self._images) < self.max_visual_images:
                self._images.append(image)
            self._predictions.append(_cpu_mapping(prediction))
            self._targets.append(_cpu_mapping(target))

    @rank_zero_only
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Publish diverse validation diagnostics to ClearML and local artifacts."""
        del pl_module
        if trainer.sanity_checking:
            return
        epoch = int(trainer.current_epoch) + 1
        if epoch % self.visualize_every_n_epochs and epoch != trainer.max_epochs:
            return
        if not self._targets:
            self.run.log_message(
                f"validation epoch {epoch}: no diagnostic samples captured",
                level=logging.WARNING,
            )
            return
        directory = self.run.root / "logs" / "diagnostics"
        renderers: list[tuple[str, str, Any]] = [
            (
                "prediction-errors",
                "GT, true positives, false positives, and false negatives",
                lambda path: render_prediction_grid(
                    self._images,
                    self._predictions[: len(self._images)],
                    self._targets[: len(self._images)],
                    self.class_names,
                    path,
                    score_threshold=self.score_threshold,
                    max_images=self.max_visual_images,
                ),
            ),
            (
                "confidence-iou",
                "confidence, localization, and error composition",
                lambda path: render_confidence_iou_diagnostics(
                    self._predictions,
                    self._targets,
                    path,
                    score_threshold=self.score_threshold,
                ),
            ),
            (
                "confusion-matrix",
                "class confusion including background",
                lambda path: render_confusion_matrix(
                    self._predictions,
                    self._targets,
                    self.class_names,
                    path,
                    score_threshold=self.score_threshold,
                ),
            ),
            (
                "precision-recall",
                "global PR and threshold sweep",
                lambda path: render_precision_recall(
                    self._predictions, self._targets, path
                ),
            ),
            (
                "metric-history",
                "loss, AP, LR, gradient, throughput, and memory",
                lambda path: render_metric_history(
                    self.logger.history.snapshot(), path
                ),
            ),
        ]
        if epoch == 1:
            renderers.append(
                (
                    "dataset-distributions",
                    "class, density, area, and aspect distributions",
                    lambda path: render_dataset_diagnostics(
                        self._targets, self.class_names, path
                    ),
                )
            )
        for name, description, renderer in renderers:
            path = directory / f"epoch-{epoch:03d}-{name}.png"
            self._publish_figure(name, description, epoch, path, renderer)

    def _publish_figure(
        self,
        name: str,
        description: str,
        epoch: int,
        path: Path,
        renderer: Any,
    ) -> None:
        """Render and publish one non-fatal diagnostic figure."""
        try:
            renderer(path)
            self.run.log_image("RF-DETR diagnostics", description, epoch, path)
            self.run.save_artifact(path.name, path)
            self.run.log_message(f"published diagnostic {name}: {path.name}")
        except Exception as error:
            self.run.log_message(
                f"diagnostic {name} failed: {type(error).__name__}: {error}",
                level=logging.WARNING,
            )


def _metric_parts(name: str) -> tuple[str, str]:
    normalized = name.replace("_step", "").replace("_epoch", "")
    if "/" in normalized:
        split, series = normalized.split("/", 1)
        return split, series
    return "trainer", normalized


def _finite_scalar(value: Any) -> float | None:
    if isinstance(value, bool | int | float):
        result = float(value)
        return result if math.isfinite(result) else None
    try:
        tensor = torch.as_tensor(value).detach().cpu()
    except (TypeError, ValueError):
        return None
    if tensor.numel() != 1 or not torch.isfinite(tensor):
        return None
    return float(tensor)


def _cpu_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: item.detach().cpu() if isinstance(item, Tensor) else item
        for name, item in value.items()
    }
