"""Two-epoch RF-DETR training worker with ClearML-ready artifacts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")

import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.utils.data import DataLoader

from context_detection.config import ExperimentConfig
from context_detection.contracts import ContextBatch, DetectionBatch, DetectorOutput
from context_detection.evaluation import coco_ap, detector_output_to_predictions
from context_detection.experiment import ExperimentComponents, ExperimentRun
from context_detection.models.losses import SetCriterion
from context_detection.models.rfdetr import RFDetrAdapter


class RFDetrImageExperiment:
    """Train the directory-backed RF-DETR model on image_dataloader."""

    def __init__(
        self,
        experiment: ExperimentRun,
        config: ExperimentConfig,
        components: ExperimentComponents,
    ) -> None:
        self.experiment = experiment
        self.config = config
        self.model: nn.Module = components.model
        self.train_loader: DataLoader[Any] = components.loader("train")
        self.validation_loader: DataLoader[Any] = components.loader("validation")
        self.components = components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("RF-DETR Datasphere experiment requires a CUDA GPU")
        detector = getattr(self.model, "detector", None)
        if not isinstance(detector, RFDetrAdapter):
            raise TypeError("RF-DETR experiment requires RFDetrAdapter")
        detector.freeze_for_class_adaptation()
        self.model.to(self.device)
        self.criterion = SetCriterion(config.detector.num_classes).to(self.device)
        self.amp_enabled = config.train.amp
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.history: dict[str, list[float]] = defaultdict(list)

    def run(self) -> dict[str, Any]:
        optimizer = self._make_optimizer()
        self.experiment.record_metadata(
            "runtime",
            {
                "device": str(self.device),
                "gpu_name": torch.cuda.get_device_name(self.device),
                "gpu_memory_gb": round(
                    torch.cuda.get_device_properties(self.device).total_memory / 2**30,
                    2,
                ),
                "amp": self.amp_enabled,
                "gradient_accumulation": self.config.train.grad_accum,
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in self.model.parameters()
                    if parameter.requires_grad
                ),
                "adaptation": "class_embed_only",
            },
        )
        best_map = float("-inf")
        for epoch in range(1, self.config.train.epochs + 1):
            train_metrics = self._train_epoch(optimizer, epoch)
            validation_metrics, visual_batch, visual_output = self._evaluate()
            self.experiment.log_metrics(train_metrics, epoch, "train")
            self.experiment.log_metrics(validation_metrics, epoch, "validation")
            self._remember(train_metrics, validation_metrics)
            if validation_metrics["map"] > best_map:
                best_map = validation_metrics["map"]
                self._save_checkpoint("best.pt", optimizer, epoch, validation_metrics)
            self._save_checkpoint(
                f"epoch-{epoch}.pt", optimizer, epoch, validation_metrics
            )
            if visual_batch is not None and visual_output is not None:
                self._save_prediction_visualization(epoch, visual_batch, visual_output)

        self._save_curve_visualization()
        return {
            "epochs": self.config.train.epochs,
            "best_map": best_map,
            "train_samples": len(self.train_loader.dataset),
            "validation_samples": len(self.validation_loader.dataset),
            "model_endpoint": type(self.model).__name__,
            "dataset_endpoint": type(self.train_loader).__name__,
        }

    def _make_optimizer(self) -> torch.optim.Optimizer:
        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        try:
            return torch.optim.AdamW(
                parameters,
                lr=self.config.train.lr,
                weight_decay=self.config.train.weight_decay,
                fused=True,
            )
        except (RuntimeError, TypeError):
            return torch.optim.AdamW(
                parameters,
                lr=self.config.train.lr,
                weight_decay=self.config.train.weight_decay,
            )

    def _train_epoch(
        self, optimizer: torch.optim.Optimizer, epoch: int
    ) -> dict[str, float]:
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        started = perf_counter()
        for step, raw_batch in enumerate(self.train_loader, start=1):
            batch, context = _move_batch(raw_batch, self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp_enabled):
                output, _ = self.model(batch, context)
                losses = self.criterion(output.logits, output.boxes, batch.targets)
                loss = losses["loss_total"] / self.config.train.grad_accum
            self.scaler.scale(loss).backward()
            if step % self.config.train.grad_accum == 0 or step == len(
                self.train_loader
            ):
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)
            for name, value in losses.items():
                totals[name] += float(value.detach())
            batches += 1
            if step % self.config.logging.every_n_steps == 0:
                self.experiment.log_metrics(
                    {name: value / batches for name, value in totals.items()},
                    step=(epoch - 1) * len(self.train_loader) + step,
                    split="train_step",
                )
        elapsed = perf_counter() - started
        metrics = {name: value / batches for name, value in totals.items()}
        metrics["samples_per_second"] = len(self.train_loader.dataset) / elapsed
        metrics["gpu_memory_gb"] = torch.cuda.max_memory_allocated(self.device) / 2**30
        torch.cuda.reset_peak_memory_stats(self.device)
        return metrics

    @torch.no_grad()
    def _evaluate(
        self,
    ) -> tuple[dict[str, float], DetectionBatch | None, DetectorOutput | None]:
        self.model.eval()
        losses: dict[str, float] = defaultdict(float)
        predictions: list[dict[str, Tensor | int]] = []
        targets: list[dict[str, Any]] = []
        visual_batch: DetectionBatch | None = None
        visual_output: DetectorOutput | None = None
        batches = 0
        for raw_batch in self.validation_loader:
            batch, context = _move_batch(raw_batch, self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp_enabled):
                output, _ = self.model(batch, context)
                batch_losses = self.criterion(
                    output.logits, output.boxes, batch.targets
                )
            for name, value in batch_losses.items():
                losses[name] += float(value)
            predictions.extend(
                detector_output_to_predictions(
                    output,
                    [int(target["image_id"]) for target in batch.targets],
                    score_threshold=0.05,
                )
            )
            targets.extend(_cpu_targets(batch.targets))
            if visual_batch is None:
                visual_batch, visual_output = batch, output
            batches += 1
        metrics = {name: value / batches for name, value in losses.items()}
        metrics.update(coco_ap(predictions, targets))
        return metrics, visual_batch, visual_output

    def _save_checkpoint(
        self,
        name: str,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        path = self.experiment.checkpoints_dir / name
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "config": self.config.model_dump(mode="json"),
            },
            path,
        )
        self.experiment.save_artifact(name, path)

    def _save_prediction_visualization(
        self, epoch: int, batch: DetectionBatch, output: DetectorOutput
    ) -> None:
        path = self.experiment.root / "logs" / f"predictions-epoch-{epoch}.png"
        _render_predictions(batch, output, path)
        self.experiment.save_artifact(path.name, path)

    def _save_curve_visualization(self) -> None:
        path = self.experiment.root / "logs" / "training-curves.png"
        figure, axis = plt.subplots(figsize=(8, 5))
        for name, values in self.history.items():
            axis.plot(range(1, len(values) + 1), values, marker="o", label=name)
        axis.set(xlabel="epoch", ylabel="value", title="RF-DETR Datasphere smoke run")
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        self.experiment.save_artifact(path.name, path)

    def _remember(
        self, train_metrics: dict[str, float], validation_metrics: dict[str, float]
    ) -> None:
        for name in ("loss_total",):
            self.history[f"train/{name}"].append(train_metrics[name])
            self.history[f"validation/{name}"].append(validation_metrics[name])
        self.history["validation/map"].append(validation_metrics["map"])


def _move_batch(
    raw_batch: tuple[DetectionBatch, ContextBatch], device: torch.device
) -> tuple[DetectionBatch, ContextBatch]:
    batch, context = raw_batch
    targets = [
        {
            key: value.to(device, non_blocking=True)
            if isinstance(value, Tensor)
            else value
            for key, value in target.items()
        }
        for target in batch.targets
    ]
    moved_batch = DetectionBatch(
        images=batch.images.to(device, non_blocking=True),
        targets=targets,
        sequence_id=batch.sequence_id,
        frame_id=batch.frame_id.to(device, non_blocking=True),
        timestamp=batch.timestamp.to(device, non_blocking=True),
        is_sequence_start=batch.is_sequence_start.to(device, non_blocking=True),
    )
    return moved_batch, ContextBatch(
        images=(
            context.images.to(device, non_blocking=True)
            if context.images is not None
            else None
        ),
        valid_mask=context.valid_mask.to(device, non_blocking=True),
        time_offsets=context.time_offsets.to(device, non_blocking=True),
        targets=context.targets,
        extras=context.extras,
    )


def _cpu_targets(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value.detach().cpu() if isinstance(value, Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def _render_predictions(
    batch: DetectionBatch, output: DetectorOutput, path: Path
) -> None:
    image = batch.images[0].detach().float().cpu().permute(1, 2, 0).clamp(0, 1)
    height, width = image.shape[:2]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.imshow(image)
    for box in batch.targets[0]["boxes"].detach().cpu():
        _draw_box(axis, box, width, height, "lime", "ground truth")
    score, label = output.logits[0].detach().sigmoid().max(dim=-1)
    for box, confidence, class_id in zip(
        output.boxes[0].detach().cpu(), score.cpu(), label.cpu(), strict=True
    ):
        if float(confidence) >= 0.25:
            _draw_box(
                axis,
                box,
                width,
                height,
                "red",
                f"{int(class_id)}: {float(confidence):.2f}",
            )
    axis.set(title="Green: target; red: RF-DETR prediction", xticks=[], yticks=[])
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _draw_box(
    axis: Any, box: Tensor, width: int, height: int, color: str, label: str
) -> None:
    cx, cy, box_width, box_height = (float(value) for value in box)
    x = (cx - box_width / 2) * width
    y = (cy - box_height / 2) * height
    rectangle = plt.Rectangle(
        (x, y),
        box_width * width,
        box_height * height,
        fill=False,
        edgecolor=color,
        linewidth=1.5,
    )
    axis.add_patch(rectangle)
    axis.text(
        x,
        max(y - 2, 0),
        label,
        color=color,
        fontsize=7,
        bbox={"facecolor": "black", "alpha": 0.35, "pad": 1},
    )


def run_rfdetr_image(
    experiment: ExperimentRun,
    config: ExperimentConfig,
    components: ExperimentComponents,
) -> dict[str, Any]:
    return RFDetrImageExperiment(experiment, config, components).run()
