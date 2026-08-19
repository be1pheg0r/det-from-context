"""Two-epoch RF-DETR training worker with ClearML-ready artifacts."""

from __future__ import annotations

import os
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
            raise RuntimeError(
                "RF-DETR Datasphere experiment requires CUDA, but PyTorch cannot "
                "access it: "
                f"torch={torch.__version__}, torch_cuda={torch.version.cuda}, "
                "CUDA_VISIBLE_DEVICES="
                f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
            )
        detector = getattr(self.model, "detector", None)
        if not isinstance(detector, RFDetrAdapter):
            raise TypeError("RF-DETR experiment requires RFDetrAdapter")
        detector.freeze_for_class_adaptation()
        self._configure_trainable_blocks()
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
                "adaptation": "frozen_backbone",
            },
        )
        best_map = float("-inf")
        for epoch in range(1, self.config.train.epochs + 1):
            train_metrics = self._train_epoch(optimizer, epoch)
            validation_metrics, visual_batch, visual_output = self._evaluate()
            self.experiment.log_metrics(train_metrics, epoch, "train")
            self.experiment.log_metrics(validation_metrics, epoch, "validation")
            self._remember(train_metrics, validation_metrics)
            self._save_curve_visualizations(epoch)
            if validation_metrics["map"] > best_map:
                best_map = validation_metrics["map"]
                self._save_checkpoint("best.pt", optimizer, epoch, validation_metrics)
            self._save_checkpoint(
                f"epoch-{epoch}.pt", optimizer, epoch, validation_metrics
            )
            if visual_batch is not None and visual_output is not None:
                self._save_prediction_visualization(epoch, visual_batch, visual_output)

        return {
            "epochs": self.config.train.epochs,
            "best_map": best_map,
            "train_samples": len(self.train_loader.dataset),
            "validation_samples": len(self.validation_loader.dataset),
            "model_endpoint": type(self.model).__name__,
            "dataset_endpoint": type(self.train_loader).__name__,
        }

    def _configure_trainable_blocks(self) -> None:
        freeze_backbone = self.config.detector.freeze_backbone
        freeze_decoder = self.config.detector.freeze_decoder

        for name, parameter in self.model.named_parameters():
            lowered = name.lower()
            if "class_embed" in lowered:
                parameter.requires_grad = True
            elif "decoder" in lowered:
                parameter.requires_grad = not freeze_decoder
            elif "backbone" in lowered or "encoder" in lowered:
                parameter.requires_grad = not freeze_backbone

    def _make_optimizer(self) -> torch.optim.Optimizer:
        base_lr = self.config.train.lr
        grouped_parameters: dict[str, list[nn.Parameter]] = {
            "backbone": [],
            "decoder": [],
            "head": [],
            "other": [],
        }

        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            lowered = name.lower()
            if "class_embed" in lowered:
                group = "head"
            elif "decoder" in lowered:
                group = "decoder"
            elif "backbone" in lowered or "encoder" in lowered:
                group = "backbone"
            else:
                group = "other"
            grouped_parameters[group].append(parameter)

        learning_rates = {
            "backbone": base_lr * self.config.train.backbone_lr_multiplier,
            "decoder": base_lr * self.config.train.decoder_lr_multiplier,
            "head": base_lr * self.config.train.head_lr_multiplier,
            "other": base_lr,
        }
        parameter_groups = [
            {"params": parameters, "lr": learning_rates[name], "name": name}
            for name, parameters in grouped_parameters.items()
            if parameters
        ]

        optimizer_kwargs = {
            "lr": base_lr,
            "weight_decay": self.config.train.weight_decay,
        }
        try:
            return torch.optim.AdamW(
                parameter_groups,
                fused=True,
                **optimizer_kwargs,
            )
        except (RuntimeError, TypeError):
            return torch.optim.AdamW(
                parameter_groups,
                **optimizer_kwargs,
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
        self.experiment.log_image(
            "validation predictions",
            "top-k vs ground truth",
            epoch,
            path,
        )
        self.experiment.save_artifact(path.name, path)

    def _save_curve_visualizations(self, epoch: int) -> None:
        self._save_loss_curves(epoch)
        self._save_detection_metric_curves(epoch)

    def _save_loss_curves(self, epoch: int) -> None:
        path = self.experiment.root / "logs" / "loss-curves.png"
        loss_names = ("loss_total", "loss_cls", "loss_bbox", "loss_giou")
        figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        for axis, name in zip(axes.flat, loss_names, strict=True):
            for split, color in (("train", "tab:blue"), ("validation", "tab:orange")):
                values = self.history.get(f"{split}/{name}", [])
                if values:
                    axis.plot(
                        range(1, len(values) + 1),
                        values,
                        marker="o",
                        color=color,
                        label=split,
                    )
            axis.set(title=name, ylabel="loss")
            axis.grid(True, alpha=0.25)
            axis.legend()
        for axis in axes[-1]:
            axis.set_xlabel("epoch")
        figure.suptitle("RF-DETR training and validation losses")
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        self.experiment.log_image("training curves", "losses", epoch, path)
        self.experiment.save_artifact(path.name, path)

    def _save_detection_metric_curves(self, epoch: int) -> None:
        metric_names = (
            ("map", "mAP"),
            ("map50", "mAP@50"),
            ("map75", "mAP@75"),
            ("mar", "mAR"),
            ("mar_100", "mAR@100"),
        )
        available = [
            (name, title)
            for name, title in metric_names
            if self.history.get(f"validation/{name}")
        ]
        if not available:
            return
        path = self.experiment.root / "logs" / "validation-detection-metrics.png"
        figure, axis = plt.subplots(figsize=(9, 5))
        for name, title in available:
            values = self.history[f"validation/{name}"]
            axis.plot(range(1, len(values) + 1), values, marker="o", label=title)
        axis.set(
            xlabel="epoch",
            ylabel="score",
            title="RF-DETR validation detection metrics",
            ylim=(0, 1),
        )
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        self.experiment.log_image("validation curves", "detection metrics", epoch, path)
        self.experiment.save_artifact(path.name, path)

    def _remember(
        self, train_metrics: dict[str, float], validation_metrics: dict[str, float]
    ) -> None:
        for name in ("loss_total", "loss_cls", "loss_bbox", "loss_giou"):
            if name in train_metrics:
                self.history[f"train/{name}"].append(train_metrics[name])
            if name in validation_metrics:
                self.history[f"validation/{name}"].append(validation_metrics[name])
        for name in ("map", "map50", "map75", "mar", "mar_100"):
            if name in validation_metrics:
                self.history[f"validation/{name}"].append(validation_metrics[name])


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
    batch: DetectionBatch, output: DetectorOutput, path: Path, top_k: int = 12
) -> None:
    image = batch.images[0].detach().float().cpu().permute(1, 2, 0).clamp(0, 1)
    target_boxes = batch.targets[0]["boxes"].detach().cpu()
    target_labels = batch.targets[0].get("labels")
    height, width = image.shape[:2]
    scores, labels = output.logits[0].detach().sigmoid().max(dim=-1)
    top_indices = scores.topk(min(top_k, len(scores))).indices.cpu()
    boxes = output.boxes[0].detach().cpu()[top_indices]
    confidences = scores.cpu()[top_indices]
    classes = labels.cpu()[top_indices]
    best_ious = _best_ious(boxes, target_boxes)

    figure, (axis, score_axis) = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        gridspec_kw={"width_ratios": (3, 1)},
    )
    axis.imshow(image)
    for index, box in enumerate(target_boxes):
        label = "GT"
        if isinstance(target_labels, Tensor):
            label = f"GT {int(target_labels[index])}"
        _draw_box(axis, box, width, height, "lime", label)
    for rank, (box, confidence, class_id, best_iou) in enumerate(
        zip(boxes, confidences, classes, best_ious, strict=True), start=1
    ):
        _draw_box(
            axis,
            box,
            width,
            height,
            "red",
            f"#{rank} {int(class_id)} "
            f"{float(confidence):.2f} IoU {float(best_iou):.2f}",
        )
    mean_iou = float(best_ious.mean()) if len(best_ious) else 0.0
    matched = int((best_ious >= 0.5).sum())
    axis.set(
        title=(
            f"Green: ground truth; red: top-{len(boxes)} predictions\n"
            f"best-IoU mean={mean_iou:.3f}, IoU@0.50={matched}/{len(boxes)}"
        ),
        xticks=[],
        yticks=[],
    )
    ranks = list(range(len(confidences), 0, -1))
    score_axis.barh(ranks, confidences.numpy(), color="tab:red", alpha=0.8)
    score_axis.set(
        title="Top-K confidence",
        xlabel="confidence",
        ylabel="rank",
        xlim=(0, 1),
        yticks=ranks,
    )
    score_axis.grid(True, axis="x", alpha=0.25)
    for rank, confidence, class_id, best_iou in zip(
        ranks, confidences, classes, best_ious, strict=True
    ):
        score_axis.text(
            min(float(confidence) + 0.02, 0.86),
            rank,
            f"c{int(class_id)} / {float(best_iou):.2f}",
            va="center",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _best_ious(predictions: Tensor, targets: Tensor) -> Tensor:
    if len(predictions) == 0 or len(targets) == 0:
        return torch.zeros(len(predictions))
    prediction_xyxy = _cxcywh_to_xyxy(predictions)
    target_xyxy = _cxcywh_to_xyxy(targets)
    top_left = torch.maximum(prediction_xyxy[:, None, :2], target_xyxy[None, :, :2])
    bottom_right = torch.minimum(prediction_xyxy[:, None, 2:], target_xyxy[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    prediction_area = (
        (prediction_xyxy[:, 2:] - prediction_xyxy[:, :2]).clamp(min=0).prod(dim=-1)
    )
    target_area = (target_xyxy[:, 2:] - target_xyxy[:, :2]).clamp(min=0).prod(dim=-1)
    union = prediction_area[:, None] + target_area[None, :] - intersection
    return (intersection / union.clamp_min(1e-6)).max(dim=1).values


def _cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


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
