"""Tests for Lightning logging and bounded RF-DETR diagnostic figures."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

from context_detection.monitoring.detection import (
    render_confidence_iou_diagnostics,
    render_confusion_matrix,
    render_dataset_diagnostics,
    render_metric_history,
    render_precision_recall,
    render_prediction_grid,
)
from context_detection.monitoring.lightning import (
    ExperimentLightningLogger,
    MeMOTMonitoringCallback,
    MetricHistory,
    RFDetrMonitoringCallback,
)
from context_detection.monitoring.tracking import (
    render_association_heatmap,
    render_memory_diagnostics,
    render_tracking_gif,
    render_tracking_grid,
)


class _Run:
    """Small in-memory stand-in for ExperimentRun side effects."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.config = SimpleNamespace(name="monitoring-test")
        self.metrics: list[tuple[str, int, dict[str, float]]] = []
        self.metadata: dict[str, Any] = {}
        self.messages: list[str] = []
        self.images: list[Path] = []
        self.media: list[Path] = []
        self.artifacts: list[Path] = []

    def log_metrics(self, metrics: dict[str, float], step: int, split: str) -> None:
        self.metrics.append((split, step, metrics))

    def record_metadata(self, name: str, value: Any) -> None:
        self.metadata[name] = value

    def log_message(self, message: str, level: int = 20) -> None:
        del level
        self.messages.append(message)

    def log_image(self, title: str, series: str, step: int, path: Path) -> None:
        del title, series, step
        assert path.is_file()
        self.images.append(path)

    def log_media(self, title: str, series: str, step: int, path: Path) -> None:
        del title, series, step
        assert path.is_file()
        self.media.append(path)

    def save_artifact(self, name: str, path: Path) -> Path:
        assert name == path.name and path.is_file()
        self.artifacts.append(path)
        return path


def _records() -> tuple[list[torch.Tensor], list[dict[str, Any]], list[dict[str, Any]]]:
    images = [torch.zeros(3, 16, 16), torch.ones(3, 16, 16) * 0.1]
    targets = [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
            "labels": torch.tensor([0]),
            "orig_size": torch.tensor([16, 16]),
            "size": torch.tensor([16, 16]),
        },
        {
            "boxes": torch.tensor([[0.25, 0.25, 0.2, 0.2]]),
            "labels": torch.tensor([1]),
            "orig_size": torch.tensor([16, 16]),
            "size": torch.tensor([16, 16]),
        },
    ]
    predictions = [
        {
            "boxes": torch.tensor([[4.0, 4.0, 12.0, 12.0], [0.0, 0.0, 2.0, 2.0]]),
            "scores": torch.tensor([0.9, 0.6]),
            "labels": torch.tensor([0, 1]),
        },
        {
            "boxes": torch.tensor([[2.4, 2.4, 5.6, 5.6]]),
            "scores": torch.tensor([0.8]),
            "labels": torch.tensor([1]),
        },
    ]
    return images, predictions, targets


def test_lightning_logger_groups_scalars_and_keeps_metric_history(
    tmp_path: Path,
) -> None:
    run = _Run(tmp_path)
    history = MetricHistory()
    logger = ExperimentLightningLogger(run, history)  # type: ignore[arg-type]

    logger.log_metrics(
        {
            "train/loss_step": torch.tensor(1.25),
            "val/mAP_50_95": 0.4,
            "epoch": 2,
            "ignored_vector": torch.ones(2),
        },
        step=7,
    )
    logger.log_runtime({"batch_seconds": 0.2}, step=7)

    assert run.metrics == [
        ("train", 7, {"loss": 1.25}),
        ("val", 7, {"mAP_50_95": 0.4}),
        ("runtime", 7, {"batch_seconds": 0.2}),
    ]
    assert history.snapshot() == {
        "train/loss": [(7, 1.25)],
        "val/mAP_50_95": [(7, 0.4)],
        "runtime/batch_seconds": [(7, 0.2)],
    }


def test_runtime_logger_skips_amp_overflow_without_stopping(tmp_path: Path) -> None:
    run = _Run(tmp_path)
    logger = ExperimentLightningLogger(run)  # type: ignore[arg-type]

    logger.log_runtime({"grad_norm": float("inf"), "batch_seconds": 0.5}, step=42)

    assert run.metrics == [("runtime", 42, {"batch_seconds": 0.5})]
    assert any("AMP overflow" in message for message in run.messages)


def test_detection_diagnostics_render_nonempty_png_artifacts(tmp_path: Path) -> None:
    images, predictions, targets = _records()
    class_names = ["car", "person"]
    renderers = {
        "predictions.png": lambda path: render_prediction_grid(
            images,
            predictions,
            targets,
            class_names,
            path,
            score_threshold=0.25,
            max_images=2,
        ),
        "dataset.png": lambda path: render_dataset_diagnostics(
            targets, class_names, path
        ),
        "confidence.png": lambda path: render_confidence_iou_diagnostics(
            predictions, targets, path, score_threshold=0.25
        ),
        "confusion.png": lambda path: render_confusion_matrix(
            predictions, targets, class_names, path, score_threshold=0.25
        ),
        "pr.png": lambda path: render_precision_recall(predictions, targets, path),
        "history.png": lambda path: render_metric_history(
            {
                "train/loss": [(1, 2.0), (2, 1.0)],
                "val/mAP_50_95": [(1, 0.2), (2, 0.4)],
                "runtime/grad_norm": [(1, 3.0), (2, 2.0)],
            },
            path,
        ),
    }

    for name, renderer in renderers.items():
        path = tmp_path / name
        renderer(path)
        assert path.stat().st_size > 1_000


def test_tracking_diagnostics_render_nonempty_png_artifacts(tmp_path: Path) -> None:
    images = [torch.zeros(3, 16, 16), torch.ones(3, 16, 16) * 0.1]
    targets = [
        {
            "sequence_id": "video",
            "frame_id": index,
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
            "labels": torch.tensor([0]),
            "track_ids": torch.tensor([3]),
        }
        for index in range(2)
    ]
    predictions = [
        {
            **target,
            "track_ids": torch.tensor([8]),
            "scores": torch.tensor([0.9]),
        }
        for target in targets
    ]
    renderers = {
        "tracks.png": lambda path: render_tracking_grid(
            images,
            predictions,
            targets,
            ["car"],
            path,
            max_images=2,
        ),
        "tracks.gif": lambda path: render_tracking_gif(
            images,
            predictions,
            ["car"],
            path,
            max_frames=2,
        ),
        "association.png": lambda path: render_association_heatmap(
            torch.randn(1, 8, 5), path
        ),
        "memory.png": lambda path: render_memory_diagnostics(
            [
                {
                    "active_slots": 3,
                    "mean_age": 2.0,
                    "mean_missed": 0.5,
                    "write_rate": 0.75,
                    "evicted": 1,
                }
            ],
            path,
        ),
    }

    for name, renderer in renderers.items():
        path = tmp_path / name
        renderer(path)
        assert path.stat().st_size > 1_000
    with Image.open(tmp_path / "tracks.gif") as animation:
        assert animation.n_frames == 2


def test_memot_callback_publishes_prediction_gif_every_epoch(tmp_path: Path) -> None:
    run = _Run(tmp_path)
    logger = ExperimentLightningLogger(run)  # type: ignore[arg-type]
    callback = MeMOTMonitoringCallback(
        run,  # type: ignore[arg-type]
        logger,
        class_names=["car"],
        every_n_steps=1,
        visualize_every_n_epochs=10,
        max_visual_images=2,
        max_diagnostic_images=10,
        score_threshold=0.25,
    )
    images = [torch.zeros(3, 16, 16), torch.ones(3, 16, 16) * 0.1]
    callback._tracking_images.extend(images)
    callback._tracking_predictions.extend(
        {
            "sequence_id": "video",
            "frame_id": index,
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
            "labels": torch.tensor([0]),
            "track_ids": torch.tensor([8]),
        }
        for index in range(2)
    )
    trainer = SimpleNamespace(sanity_checking=False, current_epoch=0, max_epochs=3)

    callback.on_validation_epoch_end(trainer, SimpleNamespace())  # type: ignore[arg-type]

    assert [path.suffix for path in run.media] == [".gif"]
    assert run.media == run.artifacts


def test_monitoring_callback_publishes_all_validation_diagnostics(
    tmp_path: Path,
) -> None:
    run = _Run(tmp_path)
    logger = ExperimentLightningLogger(run)  # type: ignore[arg-type]
    callback = RFDetrMonitoringCallback(
        run,  # type: ignore[arg-type]
        logger,
        class_names=["car", "person"],
        every_n_steps=1,
        visualize_every_n_epochs=1,
        max_visual_images=2,
        max_diagnostic_images=10,
        score_threshold=0.25,
    )
    images, predictions, targets = _records()
    nested = nested_tensor_from_tensor_list(images, block_size=16)
    trainer = SimpleNamespace(sanity_checking=False, current_epoch=0, max_epochs=1)

    callback.on_validation_epoch_start(trainer, SimpleNamespace())  # type: ignore[arg-type]
    callback.on_validation_batch_end(
        trainer,  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        {"results": predictions, "targets": targets},
        (nested, targets),
        0,
    )
    callback.on_validation_epoch_end(
        trainer,
        SimpleNamespace(),  # type: ignore[arg-type]
    )

    assert len(run.images) == 6
    assert len(run.artifacts) == 6
    assert any(
        "published diagnostic prediction-errors" in item for item in run.messages
    )
