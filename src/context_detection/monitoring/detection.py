"""Bounded Matplotlib diagnostics for transformed detection batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from torch import Tensor

DetectionRecord = Mapping[str, Any]


def render_prediction_grid(
    images: Sequence[Tensor],
    predictions: Sequence[DetectionRecord],
    targets: Sequence[DetectionRecord],
    class_names: Sequence[str],
    path: Path,
    *,
    score_threshold: float,
    max_images: int,
) -> None:
    """Render GT and matched/unmatched predictions on fixed validation images."""
    count = min(len(images), len(predictions), len(targets), max_images)
    if count == 0:
        raise ValueError("prediction grid requires at least one image")
    columns = min(2, count)
    rows = ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(8 * columns, 6 * rows))
    flat_axes = np.asarray(axes, dtype=object).reshape(-1)
    for axis, image, prediction, target in zip(
        flat_axes, images[:count], predictions[:count], targets[:count], strict=False
    ):
        displayed = _denormalize_image(image)
        height, width = displayed.shape[:2]
        axis.imshow(displayed)
        target_boxes = _target_boxes_xyxy(target, width, height)
        target_labels = _labels(target)
        pred_boxes, pred_scores, pred_labels = _prediction_boxes_xyxy(
            prediction, target, width, height, score_threshold
        )
        pred_matched, target_matched, best_ious = _greedy_matches(
            pred_boxes, pred_scores, pred_labels, target_boxes, target_labels
        )
        for index, (box, label) in enumerate(
            zip(target_boxes, target_labels, strict=True)
        ):
            color = "lime" if target_matched[index] else "gold"
            _draw_box(axis, box, color, f"GT {_class_name(label, class_names)}")
        for index, (box, score, label) in enumerate(
            zip(pred_boxes, pred_scores, pred_labels, strict=True)
        ):
            color = "deepskyblue" if pred_matched[index] else "red"
            _draw_box(
                axis,
                box,
                color,
                f"{_class_name(label, class_names)} {float(score):.2f} "
                f"IoU {float(best_ious[index]):.2f}",
            )
        axis.set_title(
            f"matched={int(pred_matched.sum())}, FP={int((~pred_matched).sum())}, "
            f"FN={int((~target_matched).sum())}"
        )
        axis.set_axis_off()
    for axis in flat_axes[count:]:
        axis.set_axis_off()
    figure.suptitle("Validation errors: blue=TP, red=FP, yellow=FN, green=matched GT")
    figure.tight_layout()
    _save_figure(figure, path)


def render_dataset_diagnostics(
    targets: Sequence[DetectionRecord],
    class_names: Sequence[str],
    path: Path,
) -> None:
    """Render class balance, object density, box area, and aspect ratio."""
    labels: list[int] = []
    objects_per_image: list[int] = []
    areas: list[float] = []
    aspects: list[float] = []
    for target in targets:
        boxes = _boxes(target.get("boxes"))
        target_labels = _labels(target)
        labels.extend(int(value) for value in target_labels)
        objects_per_image.append(len(target_labels))
        if boxes.numel():
            widths = boxes[:, 2].clamp_min(1e-8)
            heights = boxes[:, 3].clamp_min(1e-8)
            areas.extend((widths * heights).tolist())
            aspects.extend((widths / heights).tolist())

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    counts = np.bincount(labels, minlength=len(class_names))
    axes[0, 0].bar(np.arange(len(counts)), counts, color="tab:blue")
    axes[0, 0].set(
        title="Objects per class",
        xlabel="class id",
        ylabel="objects",
        xticks=np.arange(len(counts)),
    )
    axes[0, 0].tick_params(axis="x", labelrotation=90, labelsize=7)
    axes[0, 0].set_xticklabels(class_names)
    axes[0, 1].hist(objects_per_image, bins=_bins(objects_per_image), color="tab:green")
    axes[0, 1].set(title="Objects per image", xlabel="count", ylabel="images")
    axes[1, 0].hist(areas, bins=30, color="tab:orange")
    axes[1, 0].set(title="Normalized box area", xlabel="w × h", ylabel="objects")
    axes[1, 1].hist(
        aspects, bins=30, range=(0, min(max(aspects, default=1), 8)), color="tab:purple"
    )
    axes[1, 1].set(title="Box aspect ratio", xlabel="w / h", ylabel="objects")
    for axis in axes.flat:
        axis.grid(True, alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, path)


def render_confidence_iou_diagnostics(
    predictions: Sequence[DetectionRecord],
    targets: Sequence[DetectionRecord],
    path: Path,
    *,
    score_threshold: float,
) -> None:
    """Render confidence, best IoU, score/IoU relation, and error composition."""
    scores: list[float] = []
    ious: list[float] = []
    true_positives = false_positives = false_negatives = 0
    for prediction, target in zip(predictions, targets, strict=True):
        target_boxes = _target_boxes_absolute(target)
        target_labels = _labels(target)
        pred_boxes, pred_scores, pred_labels = _prediction_boxes_absolute(
            prediction, score_threshold
        )
        matched, target_matched, best = _greedy_matches(
            pred_boxes, pred_scores, pred_labels, target_boxes, target_labels
        )
        scores.extend(pred_scores.tolist())
        ious.extend(best.tolist())
        true_positives += int(matched.sum())
        false_positives += int((~matched).sum())
        false_negatives += int((~target_matched).sum())

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(scores, bins=25, range=(0, 1), color="tab:blue")
    axes[0, 0].set(title="Prediction confidence", xlabel="score", ylabel="detections")
    axes[0, 1].hist(ious, bins=25, range=(0, 1), color="tab:orange")
    axes[0, 1].set(title="Best class-aware IoU", xlabel="IoU", ylabel="detections")
    axes[1, 0].scatter(scores, ious, s=10, alpha=0.35, color="tab:purple")
    axes[1, 0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set(
        title="Confidence vs localization", xlabel="score", ylabel="best IoU"
    )
    axes[1, 1].bar(
        ["TP", "FP", "FN"],
        [true_positives, false_positives, false_negatives],
        color=["tab:green", "tab:red", "gold"],
    )
    axes[1, 1].set(title="Error composition at IoU 0.5", ylabel="objects")
    for axis in axes.flat:
        axis.grid(True, alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, path)


def render_confusion_matrix(
    predictions: Sequence[DetectionRecord],
    targets: Sequence[DetectionRecord],
    class_names: Sequence[str],
    path: Path,
    *,
    score_threshold: float,
) -> None:
    """Render a class confusion matrix with an explicit background row/column."""
    size = len(class_names) + 1
    background = size - 1
    matrix = np.zeros((size, size), dtype=np.int64)
    for prediction, target in zip(predictions, targets, strict=True):
        target_boxes = _target_boxes_absolute(target)
        target_labels = _labels(target)
        pred_boxes, pred_scores, pred_labels = _prediction_boxes_absolute(
            prediction, score_threshold
        )
        pred_matched, target_matched, _ = _greedy_matches(
            pred_boxes,
            pred_scores,
            pred_labels,
            target_boxes,
            target_labels,
            require_same_class=False,
        )
        iou = _box_iou(pred_boxes, target_boxes)
        used_targets: set[int] = set()
        for pred_index in pred_scores.argsort(descending=True).tolist():
            available = [
                index
                for index in range(len(target_boxes))
                if index not in used_targets and iou[pred_index, index] >= 0.5
            ]
            if not available:
                matrix[background, int(pred_labels[pred_index])] += 1
                continue
            target_index = max(
                available, key=lambda index: float(iou[pred_index, index])
            )
            used_targets.add(target_index)
            matrix[int(target_labels[target_index]), int(pred_labels[pred_index])] += 1
        for target_index in range(len(target_boxes)):
            if target_index not in used_targets:
                matrix[int(target_labels[target_index]), background] += 1
        del pred_matched, target_matched

    labels = [*class_names, "background"]
    figure, axis = plt.subplots(figsize=(max(9, size * 0.55), max(8, size * 0.5)))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        title="Detection confusion matrix",
        xlabel="predicted",
        ylabel="ground truth",
        xticks=np.arange(size),
        yticks=np.arange(size),
    )
    axis.set_xticklabels(labels, rotation=90, fontsize=7)
    axis.set_yticklabels(labels, fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    _save_figure(figure, path)


def render_precision_recall(
    predictions: Sequence[DetectionRecord],
    targets: Sequence[DetectionRecord],
    path: Path,
) -> None:
    """Render global precision, recall, and F1 across score thresholds."""
    thresholds = torch.linspace(0.0, 1.0, 51)
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    for threshold in thresholds:
        tp = fp = fn = 0
        for prediction, target in zip(predictions, targets, strict=True):
            target_boxes = _target_boxes_absolute(target)
            target_labels = _labels(target)
            pred_boxes, pred_scores, pred_labels = _prediction_boxes_absolute(
                prediction, float(threshold)
            )
            matched, target_matched, _ = _greedy_matches(
                pred_boxes, pred_scores, pred_labels, target_boxes, target_labels
            )
            tp += int(matched.sum())
            fp += int((~matched).sum())
            fn += int((~target_matched).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / max(p + r, 1e-12))

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(recall, precision, color="tab:blue")
    axes[0].set(
        title="Global PR curve @ IoU 0.5",
        xlabel="recall",
        ylabel="precision",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1].plot(thresholds.numpy(), f1, color="tab:green", label="F1")
    axes[1].plot(
        thresholds.numpy(), precision, color="tab:blue", alpha=0.6, label="precision"
    )
    axes[1].plot(
        thresholds.numpy(), recall, color="tab:orange", alpha=0.6, label="recall"
    )
    axes[1].set(
        title="Threshold sweep",
        xlabel="score threshold",
        ylabel="metric",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1].legend()
    for axis in axes:
        axis.grid(True, alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, path)


def render_metric_history(
    history: Mapping[str, Sequence[tuple[int, float]]], path: Path
) -> None:
    """Render loss, detection quality, learning-rate, and runtime histories."""
    groups = (
        ("loss", lambda name: "loss" in name.lower()),
        (
            "quality",
            lambda name: any(token in name.lower() for token in ("map", "mar", "f1")),
        ),
        ("learning rate", lambda name: "lr" in name.lower()),
        (
            "runtime",
            lambda name: any(
                token in name.lower()
                for token in ("memory", "second", "throughput", "grad_norm")
            ),
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    for axis, (title, selector) in zip(axes.flat, groups, strict=True):
        for name, points in sorted(history.items()):
            if not points or not selector(name):
                continue
            axis.plot(
                [step for step, _ in points],
                [value for _, value in points],
                label=name,
                linewidth=1.4,
            )
        axis.set(title=title, xlabel="global step")
        axis.grid(True, alpha=0.2)
        if axis.lines:
            axis.legend(fontsize=7)
    figure.tight_layout()
    _save_figure(figure, path)


def _prediction_boxes_absolute(
    prediction: DetectionRecord, score_threshold: float
) -> tuple[Tensor, Tensor, Tensor]:
    boxes = _boxes(prediction.get("boxes"))
    scores = torch.as_tensor(prediction.get("scores", []), dtype=torch.float32).cpu()
    labels = _labels(prediction)
    keep = scores >= score_threshold
    return boxes[keep], scores[keep], labels[keep]


def _prediction_boxes_xyxy(
    prediction: DetectionRecord,
    target: DetectionRecord,
    display_width: int,
    display_height: int,
    score_threshold: float,
) -> tuple[Tensor, Tensor, Tensor]:
    boxes, scores, labels = _prediction_boxes_absolute(prediction, score_threshold)
    original_height, original_width = _original_size(target)
    scale = torch.tensor(
        [
            display_width / original_width,
            display_height / original_height,
            display_width / original_width,
            display_height / original_height,
        ]
    )
    return boxes * scale, scores, labels


def _target_boxes_absolute(target: DetectionRecord) -> Tensor:
    height, width = _original_size(target)
    boxes = _cxcywh_to_xyxy(_boxes(target.get("boxes")))
    return boxes * torch.tensor([width, height, width, height])


def _target_boxes_xyxy(
    target: DetectionRecord, display_width: int, display_height: int
) -> Tensor:
    boxes = _cxcywh_to_xyxy(_boxes(target.get("boxes")))
    return boxes * torch.tensor(
        [display_width, display_height, display_width, display_height]
    )


def _greedy_matches(
    prediction_boxes: Tensor,
    prediction_scores: Tensor,
    prediction_labels: Tensor,
    target_boxes: Tensor,
    target_labels: Tensor,
    *,
    require_same_class: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    pred_matched = torch.zeros(len(prediction_boxes), dtype=torch.bool)
    target_matched = torch.zeros(len(target_boxes), dtype=torch.bool)
    best_ious = torch.zeros(len(prediction_boxes), dtype=torch.float32)
    if not len(prediction_boxes) or not len(target_boxes):
        return pred_matched, target_matched, best_ious
    ious = _box_iou(prediction_boxes, target_boxes)
    if require_same_class:
        ious = ious * (prediction_labels[:, None] == target_labels[None, :])
    best_ious = ious.max(dim=1).values
    for pred_index in prediction_scores.argsort(descending=True).tolist():
        candidates = ious[pred_index].clone()
        candidates[target_matched] = -1
        best_value, target_index = candidates.max(dim=0)
        if best_value >= 0.5:
            pred_matched[pred_index] = True
            target_matched[target_index] = True
    return pred_matched, target_matched, best_ious


def _box_iou(first: Tensor, second: Tensor) -> Tensor:
    if not len(first) or not len(second):
        return torch.zeros((len(first), len(second)), dtype=torch.float32)
    top_left = torch.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    first_area = (first[:, 2:] - first[:, :2]).clamp_min(0).prod(dim=-1)
    second_area = (second[:, 2:] - second[:, :2]).clamp_min(0).prod(dim=-1)
    return intersection / (
        first_area[:, None] + second_area[None, :] - intersection
    ).clamp_min(1e-8)


def _cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    if not boxes.numel():
        return boxes.clone()
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ),
        dim=-1,
    )


def _boxes(value: Any) -> Tensor:
    boxes = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("detection boxes must have shape [N,4]")
    return boxes


def _labels(record: DetectionRecord) -> Tensor:
    labels = torch.as_tensor(record.get("labels", []), dtype=torch.long).detach().cpu()
    if labels.ndim != 1:
        raise ValueError("detection labels must have shape [N]")
    return labels


def _original_size(target: DetectionRecord) -> tuple[int, int]:
    size = torch.as_tensor(target.get("orig_size", target.get("size"))).flatten()
    if size.numel() != 2:
        raise ValueError("target orig_size/size must contain [H,W]")
    return int(size[0]), int(size[1])


def _denormalize_image(image: Tensor) -> np.ndarray[Any, Any]:
    tensor = image.detach().float().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()


def _draw_box(axis: Any, box: Tensor, color: str, label: str) -> None:
    x1, y1, x2, y2 = (float(value) for value in box)
    axis.add_patch(
        Rectangle(
            (x1, y1),
            max(x2 - x1, 0),
            max(y2 - y1, 0),
            fill=False,
            edgecolor=color,
            linewidth=1.8,
        )
    )
    axis.text(
        x1,
        max(y1 - 2, 0),
        label,
        color=color,
        fontsize=7,
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 1},
    )


def _class_name(label: Tensor | int, class_names: Sequence[str]) -> str:
    index = int(label)
    return class_names[index] if 0 <= index < len(class_names) else f"class-{index}"


def _bins(values: Sequence[int]) -> int:
    return max(1, min(30, max(values, default=0) + 1))


def _save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
