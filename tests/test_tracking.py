"""Tests for sequential MeMOT postprocessing and MOT metrics."""

from __future__ import annotations

import pytest
import torch

from context_detection.contracts import DetectionBatch, DetectorOutput
from context_detection.tracking import (
    tracking_metrics,
    tracking_output_to_predictions,
)


def _record(frame_id: int, track_id: int) -> dict[str, object]:
    return {
        "sequence_id": "video",
        "frame_id": frame_id,
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "labels": torch.tensor([0], dtype=torch.int64),
        "track_ids": torch.tensor([track_id], dtype=torch.int64),
    }


def test_perfect_tracks_score_one_for_detection_and_identity() -> None:
    targets = [_record(0, 4), _record(1, 4)]
    predictions = [_record(0, 17), _record(1, 17)]

    result = tracking_metrics(
        predictions,
        targets,
        annotation_mode="tracking",
    )

    assert result.available
    for name in ("hota", "deta", "assa", "idf1", "mota", "motp"):
        assert result.metrics[name] == pytest.approx(1.0)
    assert result.metrics["id_switches"] == 0


def test_identity_switch_reduces_association_metrics() -> None:
    targets = [_record(0, 4), _record(1, 4)]
    predictions = [_record(0, 17), _record(1, 18)]

    result = tracking_metrics(
        predictions,
        targets,
        annotation_mode="tracking",
        iou_thresholds=[0.5],
    )

    assert result.metrics["deta"] == pytest.approx(1.0)
    assert result.metrics["assa"] == pytest.approx(0.5)
    assert result.metrics["hota"] == pytest.approx(2**-0.5)
    assert result.metrics["idf1"] == pytest.approx(0.5)
    assert result.metrics["id_switches"] == 1
    assert result.metrics["mota"] == pytest.approx(0.5)


def test_reference_frame_annotations_explicitly_disable_tracking_metrics() -> None:
    result = tracking_metrics(
        [_record(0, 1)],
        [_record(0, -1)],
        annotation_mode="reference_frame",
    )

    assert not result.available
    assert result.metrics == {}
    assert "temporal identities" in str(result.reason)
    assert result.as_dict()["tracking_available"] == 0.0


def test_single_tracking_frame_is_not_enough_for_identity_metrics() -> None:
    result = tracking_metrics(
        [_record(0, 1)],
        [_record(0, 2)],
        annotation_mode="tracking",
    )

    assert not result.available
    assert "at least two frames" in str(result.reason)


def test_tracking_postprocess_filters_untracked_and_low_score_proposals() -> None:
    output = DetectorOutput(
        logits=torch.tensor([[[5.0, -5.0], [4.0, -4.0], [-5.0, -4.0]]]),
        boxes=torch.tensor(
            [[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]]
        ),
        queries=torch.zeros(1, 3, 4),
        reference_points=torch.full((1, 3, 4), 0.5),
        aux={"memot": {"track_ids": torch.tensor([[7, -1, 9]])}},
    )
    batch = DetectionBatch(
        images=torch.zeros(1, 3, 8, 8),
        targets=[
            {
                "boxes": torch.empty(0, 4),
                "labels": torch.empty(0, dtype=torch.int64),
            }
        ],
        sequence_id=["video"],
        frame_id=torch.tensor([3], dtype=torch.int64),
        timestamp=torch.tensor([0.3]),
        is_sequence_start=torch.tensor([False]),
    )

    records = tracking_output_to_predictions(output, batch, score_threshold=0.5)

    assert records[0]["sequence_id"] == "video"
    assert records[0]["frame_id"] == 3
    assert records[0]["track_ids"].tolist() == [7]
