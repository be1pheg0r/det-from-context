"""Bounded video, identity, association, and memory visualizations."""

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


def render_tracking_grid(
    images: Sequence[Tensor],
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    class_names: Sequence[str],
    path: Path,
    *,
    max_images: int,
) -> None:
    """Overlay predicted and ground-truth identities on sequential frames."""
    count = min(len(images), len(predictions), len(targets), max_images)
    if not count:
        raise ValueError("tracking grid requires at least one frame")
    columns = min(3, count)
    rows = ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(7 * columns, 5 * rows))
    flat_axes = np.asarray(axes, dtype=object).reshape(-1)
    for axis, image, prediction, target in zip(
        flat_axes, images[:count], predictions[:count], targets[:count], strict=False
    ):
        displayed = _denormalize_image(image)
        height, width = displayed.shape[:2]
        axis.imshow(displayed)
        _draw_records(axis, target, width, height, class_names, ground_truth=True)
        _draw_records(axis, prediction, width, height, class_names, ground_truth=False)
        axis.set_title(
            f"{prediction.get('sequence_id', 'sequence')} / "
            f"frame {prediction.get('frame_id', '?')}"
        )
        axis.set_axis_off()
    for axis in flat_axes[count:]:
        axis.set_axis_off()
    figure.suptitle("MeMOT identities: GT dashed, predictions solid")
    figure.tight_layout()
    _save(figure, path)


def render_association_heatmap(
    association_logits: Tensor,
    path: Path,
    *,
    max_proposals: int = 40,
    max_slots: int = 32,
) -> None:
    """Render proposal-to-track probabilities including the new-track class."""
    if association_logits.ndim != 3 or association_logits.shape[0] == 0:
        raise ValueError("association_logits must have shape [B, N, S+1]")
    probability = association_logits[0].detach().float().softmax(dim=-1).cpu()
    proposal_confidence = probability.max(dim=-1).values
    proposals = proposal_confidence.argsort(descending=True)[:max_proposals]
    slot_count = min(probability.shape[1] - 1, max_slots)
    selected = torch.cat(
        (probability[proposals, :slot_count], probability[proposals, -1:]), dim=-1
    )
    figure, axis = plt.subplots(
        figsize=(max(7, selected.shape[1] * 0.32), max(5, selected.shape[0] * 0.2))
    )
    image = axis.imshow(selected.numpy(), aspect="auto", vmin=0, vmax=1, cmap="magma")
    labels = [*(f"slot-{index}" for index in range(slot_count)), "new"]
    axis.set(
        title="MeMOT association probabilities",
        xlabel="track memory",
        ylabel="current proposal (ranked)",
        xticks=np.arange(len(labels)),
    )
    axis.set_xticklabels(labels, rotation=90, fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.03)
    figure.tight_layout()
    _save(figure, path)


def render_memory_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]], path: Path
) -> None:
    """Plot bounded memory occupancy, age, misses, write rate, and evictions."""
    scalar_names = ("active_slots", "mean_age", "mean_missed", "write_rate", "evicted")
    values: dict[str, list[float]] = {name: [] for name in scalar_names}
    for item in diagnostics:
        for name in scalar_names:
            value = item.get(name)
            if isinstance(value, Tensor) and value.numel() == 1:
                value = float(value.detach().cpu())
            if isinstance(value, int | float):
                values[name].append(float(value))
    if not any(values.values()):
        raise ValueError("memory diagnostics contain no scalar values")
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for name in ("active_slots", "mean_age", "mean_missed"):
        if values[name]:
            axes[0].plot(values[name], label=name)
    for name in ("write_rate", "evicted"):
        if values[name]:
            axes[1].plot(values[name], label=name)
    axes[0].set(title="Track memory occupancy and lifecycle", ylabel="slots / frames")
    axes[1].set(title="Memory writes and evictions", xlabel="validation batch")
    for axis in axes:
        axis.grid(True, alpha=0.2)
        if axis.lines:
            axis.legend()
    figure.tight_layout()
    _save(figure, path)


def _draw_records(
    axis: Any,
    record: Mapping[str, Any],
    width: int,
    height: int,
    class_names: Sequence[str],
    *,
    ground_truth: bool,
) -> None:
    boxes = torch.as_tensor(record.get("boxes", []), dtype=torch.float32).reshape(-1, 4)
    labels = torch.as_tensor(record.get("labels", []), dtype=torch.int64)
    track_ids = torch.as_tensor(record.get("track_ids", []), dtype=torch.int64)
    for box, label, track_id in zip(boxes, labels, track_ids, strict=True):
        center_x, center_y, box_width, box_height = box.tolist()
        x1 = (center_x - box_width / 2) * width
        y1 = (center_y - box_height / 2) * height
        class_id = int(label)
        class_name = (
            class_names[class_id]
            if 0 <= class_id < len(class_names)
            else f"class-{class_id}"
        )
        color = "lime" if ground_truth else f"C{int(track_id) % 10}"
        axis.add_patch(
            Rectangle(
                (x1, y1),
                box_width * width,
                box_height * height,
                fill=False,
                edgecolor=color,
                linewidth=1.8,
                linestyle="--" if ground_truth else "-",
            )
        )
        prefix = "GT" if ground_truth else "P"
        axis.text(
            x1,
            max(y1 - 2, 0),
            f"{prefix}#{int(track_id)} {class_name}",
            color=color,
            fontsize=7,
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 1},
        )


def _denormalize_image(image: Tensor) -> np.ndarray[Any, Any]:
    tensor = image.detach().float().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()


def _save(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
