"""Метрики детекции и упаковка результатов для ExperimentProtocol."""

from __future__ import annotations

import pytest
import torch

from context_detection.contracts import DetectorOutput
from context_detection.evaluation import (
    add_newly_appeared_flags,
    coco_ap,
    detector_output_to_predictions,
    failure_mode_grid,
    memory_diagnostics_summary,
    metric_deltas,
    newly_appeared_coco_ap,
    occluded_coco_ap,
)


def _target(*, image_id: int = 0) -> dict:
    return {
        "image_id": image_id,
        "image_size": torch.tensor([100, 200]),
        "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.2]]),
        "labels": torch.tensor([0]),
    }


def _prediction(*, image_id: int = 0, box=None) -> dict:
    return {
        "image_id": image_id,
        "boxes": torch.tensor(box or [[0.5, 0.5, 0.4, 0.2]]),
        "labels": torch.tensor([0]),
        "scores": torch.tensor([0.99]),
    }


def test_coco_ap_is_one_for_exact_prediction() -> None:
    pytest.importorskip("pycocotools")
    metrics = coco_ap([_prediction()], [_target()])
    assert metrics["map"] == pytest.approx(1.0)
    assert metrics["map_50"] == pytest.approx(1.0)
    assert metrics["map_75"] == pytest.approx(1.0)
    assert "map_small" in metrics


def test_coco_ap_omits_area_metrics_without_image_size() -> None:
    pytest.importorskip("pycocotools")
    target = _target()
    target.pop("image_size")
    metrics = coco_ap([_prediction()], [target])
    assert metrics["map"] == pytest.approx(1.0)
    assert "map_small" not in metrics
    assert "mar_small" not in metrics


def test_coco_ap_accepts_image_id_from_prediction_only() -> None:
    pytest.importorskip("pycocotools")
    target = _target()
    target.pop("image_id")
    metrics = coco_ap([_prediction(image_id=17)], [target])
    assert metrics["map"] == pytest.approx(1.0)


def test_coco_ap_handles_empty_predictions() -> None:
    pytest.importorskip("pycocotools")
    prediction = {
        "boxes": torch.empty(0, 4),
        "labels": torch.empty(0, dtype=torch.int64),
        "scores": torch.empty(0),
    }
    metrics = coco_ap([prediction], [_target()])
    assert metrics["map"] == 0.0


def test_occluded_coco_ap_ignores_other_ground_truth() -> None:
    pytest.importorskip("pycocotools")
    target = {
        "image_size": torch.tensor([100, 100]),
        "boxes": torch.tensor([[0.25, 0.5, 0.2, 0.2], [0.75, 0.5, 0.2, 0.2]]),
        "labels": torch.tensor([0, 0]),
        "occluded": torch.tensor([True, False]),
    }
    prediction = {
        "boxes": target["boxes"].clone(),
        "labels": target["labels"].clone(),
        "scores": torch.tensor([0.9, 0.8]),
    }
    metrics = occluded_coco_ap([prediction], [target])
    assert metrics["map"] == pytest.approx(1.0)


def test_newly_appeared_flags_follow_instances_across_frames() -> None:
    targets = [
        {
            **_target(image_id=10),
            "sequence_id": "video-a",
            "frame_id": 2,
            "instance_ids": ["car-1"],
        },
        {
            **_target(image_id=11),
            "sequence_id": "video-a",
            "frame_id": 1,
            "instance_ids": ["car-1"],
        },
        {
            **_target(image_id=12),
            "sequence_id": "video-b",
            "frame_id": 1,
            "instance_ids": ["car-1"],
        },
    ]
    prepared = add_newly_appeared_flags(targets)
    assert prepared[0]["newly_appeared"].tolist() == [False]
    assert prepared[1]["newly_appeared"].tolist() == [True]
    assert prepared[2]["newly_appeared"].tolist() == [True]


def test_newly_appeared_coco_ap_ignores_later_observations() -> None:
    pytest.importorskip("pycocotools")
    targets = [
        {
            **_target(image_id=10),
            "sequence_id": "video-a",
            "frame_id": 1,
            "instance_ids": ["car-1"],
        },
        {
            **_target(image_id=11),
            "sequence_id": "video-a",
            "frame_id": 2,
            "instance_ids": ["car-1"],
        },
    ]
    predictions = [_prediction(image_id=10), _prediction(image_id=11)]
    metrics = newly_appeared_coco_ap(predictions, targets)
    assert metrics["map"] == pytest.approx(1.0)


def test_detector_output_conversion_applies_threshold_and_top_k() -> None:
    output = DetectorOutput(
        logits=torch.tensor([[[10.0, -10.0], [-10.0, -10.0], [2.0, 3.0]]]),
        boxes=torch.tensor(
            [[[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.1, 0.1], [0.3, 0.3, 0.2, 0.2]]]
        ),
        queries=torch.zeros(1, 3, 4),
        reference_points=torch.full((1, 3, 4), 0.5),
    )
    predictions = detector_output_to_predictions(
        output,
        [42],
        score_threshold=0.8,
        max_detections=1,
    )
    assert predictions[0]["image_id"] == 42
    assert predictions[0]["boxes"].shape == (1, 4)
    assert predictions[0]["labels"].tolist() == [0]


def test_metric_deltas_and_flat_grid_match_experiment_mapping() -> None:
    baseline = {"map": 0.40, "map_50": 0.60, "map_75": 0.35}
    memot = {"map": 0.43, "map_50": 0.64, "map_75": 0.36}
    assert metric_deltas(memot, baseline, names=["map"]) == {
        "delta_map": pytest.approx(0.03)
    }

    grid = failure_mode_grid({"baseline": baseline, "memot": memot})
    assert grid["baseline.map"] == 0.40
    assert grid["memot.map"] == 0.43
    assert grid["memot.delta_map"] == pytest.approx(0.03)
    assert grid["memot.delta_map_50"] == pytest.approx(0.04)


def test_memory_diagnostics_summary_skips_attention_tensors() -> None:
    summary = memory_diagnostics_summary(
        [
            {
                "active_slots": 2,
                "mean_age": 1.0,
                "write_rate": 0.5,
                "evicted": 1,
                "read_weights": torch.ones(2, 3),
            },
            {
                "active_slots": torch.tensor(4),
                "mean_age": 3.0,
                "write_rate": 1.0,
                "evicted": 2,
            },
        ]
    )
    assert summary == {
        "memory_active_slots_mean": 3.0,
        "memory_mean_age": 2.0,
        "memory_write_rate_mean": 0.75,
        "memory_evicted_total": 3.0,
    }
