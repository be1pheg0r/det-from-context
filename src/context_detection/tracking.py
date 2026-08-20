"""Sequential MOT postprocessing and identity-aware evaluation."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from .contracts import DetectionBatch, DetectorOutput

TrackingRecord = Mapping[str, Any]
TrackKey = tuple[str, int]


@dataclass(frozen=True)
class TrackingMetricsResult:
    """Metrics plus an explicit reason when temporal evaluation is impossible."""

    available: bool
    metrics: dict[str, float]
    reason: str | None = None

    def as_dict(self, prefix: str = "tracking") -> dict[str, float | str]:
        """Flatten for experiment logging and metadata."""
        result: dict[str, float | str] = {f"{prefix}_available": float(self.available)}
        result.update(
            {f"{prefix}_{name}": value for name, value in self.metrics.items()}
        )
        if self.reason is not None:
            result[f"{prefix}_reason"] = self.reason
        return result


def tracking_output_to_predictions(
    output: DetectorOutput,
    batch: DetectionBatch,
    *,
    score_threshold: float = 0.5,
    max_detections: int = 100,
) -> list[dict[str, Any]]:
    """Convert one sequential MeMOT step into identity-bearing records."""
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be in [0, 1]")
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")
    memot = output.aux.get("memot")
    if not isinstance(memot, Mapping) or not isinstance(memot.get("track_ids"), Tensor):
        raise TypeError("MeMOT output must contain tensor memot.track_ids")
    track_ids: Tensor = memot["track_ids"]
    if track_ids.shape != output.logits.shape[:2]:
        raise ValueError("MeMOT track_ids shape does not match proposals")

    scores, labels = output.logits.detach().sigmoid().max(dim=-1)
    predictions: list[dict[str, Any]] = []
    for batch_index in range(batch.batch_size):
        keep = (scores[batch_index] >= score_threshold) & (track_ids[batch_index] >= 0)
        selected = keep.nonzero(as_tuple=False).flatten()
        if selected.numel() > max_detections:
            order = scores[batch_index, selected].argsort(descending=True)
            selected = selected[order[:max_detections]]
        predictions.append(
            {
                "sequence_id": batch.sequence_id[batch_index],
                "frame_id": int(batch.frame_id[batch_index]),
                "boxes": output.boxes[batch_index, selected].detach().cpu(),
                "scores": scores[batch_index, selected].detach().cpu(),
                "labels": labels[batch_index, selected].detach().cpu(),
                "track_ids": track_ids[batch_index, selected].detach().cpu(),
            }
        )
    return predictions


def tracking_metrics(
    predictions: Sequence[TrackingRecord],
    targets: Sequence[TrackingRecord],
    *,
    annotation_mode: str,
    iou_thresholds: Sequence[float] = tuple(index / 20 for index in range(1, 20)),
) -> TrackingMetricsResult:
    """Compute HOTA/DetA/AssA and ID metrics from true temporal identities.

    HOTA components are averaged over IoU thresholds 0.05..0.95. IDF1, MOTA,
    MOTP, and identity switches use the conventional 0.5 matching threshold.
    """
    if annotation_mode != "tracking":
        return TrackingMetricsResult(
            available=False,
            metrics={},
            reason="reference-frame annotations do not provide temporal identities",
        )
    if len(predictions) != len(targets) or not targets:
        return TrackingMetricsResult(
            available=False,
            metrics={},
            reason="tracking evaluation needs paired non-empty frame records",
        )
    thresholds = tuple(float(value) for value in iou_thresholds)
    if not thresholds or any(value <= 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("iou_thresholds must be non-empty and inside (0, 1]")
    prepared = _prepare_frames(predictions, targets)
    sequence_frames: dict[str, int] = Counter(frame.sequence_id for frame in prepared)
    if max(sequence_frames.values(), default=0) < 2:
        return TrackingMetricsResult(
            available=False,
            metrics={},
            reason="tracking metrics require at least two frames in one sequence",
        )
    if any((frame.gt_track_ids < 0).any() for frame in prepared):
        return TrackingMetricsResult(
            available=False,
            metrics={},
            reason="ground-truth track IDs are missing",
        )

    per_threshold = [
        _threshold_metrics(prepared, threshold) for threshold in thresholds
    ]
    identity = _threshold_metrics(prepared, 0.5)
    gt_total = identity["gt_total"]
    pred_total = identity["pred_total"]
    idtp = _global_identity_true_positives(identity["contingency"])
    idfp = pred_total - idtp
    idfn = gt_total - idtp
    id_denominator = 2 * idtp + idfp + idfn
    idf1 = 2 * idtp / id_denominator if id_denominator else 1.0
    mota = (
        1.0 - (identity["fn"] + identity["fp"] + identity["id_switches"]) / gt_total
        if gt_total
        else 0.0
    )
    return TrackingMetricsResult(
        available=True,
        metrics={
            "hota": sum(item["hota"] for item in per_threshold) / len(per_threshold),
            "deta": sum(item["deta"] for item in per_threshold) / len(per_threshold),
            "assa": sum(item["assa"] for item in per_threshold) / len(per_threshold),
            "idf1": idf1,
            "mota": mota,
            "motp": identity["motp"],
            "id_switches": float(identity["id_switches"]),
            "true_positives": float(identity["tp"]),
            "false_positives": float(identity["fp"]),
            "false_negatives": float(identity["fn"]),
        },
    )


@dataclass(frozen=True)
class _Frame:
    sequence_id: str
    frame_id: int
    pred_boxes: Tensor
    pred_labels: Tensor
    pred_track_ids: Tensor
    gt_boxes: Tensor
    gt_labels: Tensor
    gt_track_ids: Tensor


def _prepare_frames(
    predictions: Sequence[TrackingRecord], targets: Sequence[TrackingRecord]
) -> list[_Frame]:
    frames: list[_Frame] = []
    seen: set[tuple[str, int]] = set()
    for index, (prediction, target) in enumerate(
        zip(predictions, targets, strict=True)
    ):
        prediction_key = _frame_key(prediction, index)
        target_key = _frame_key(target, index)
        if prediction_key != target_key:
            raise ValueError(
                f"prediction/target frame mismatch: {prediction_key} != {target_key}"
            )
        if prediction_key in seen:
            raise ValueError(f"duplicate tracking frame {prediction_key}")
        seen.add(prediction_key)
        pred_boxes = _boxes(prediction, "prediction")
        gt_boxes = _boxes(target, "target")
        frames.append(
            _Frame(
                sequence_id=prediction_key[0],
                frame_id=prediction_key[1],
                pred_boxes=pred_boxes,
                pred_labels=_vector(
                    prediction, "labels", pred_boxes.shape[0], torch.int64
                ),
                pred_track_ids=_vector(
                    prediction, "track_ids", pred_boxes.shape[0], torch.int64
                ),
                gt_boxes=gt_boxes,
                gt_labels=_vector(target, "labels", gt_boxes.shape[0], torch.int64),
                gt_track_ids=_vector(
                    target, "track_ids", gt_boxes.shape[0], torch.int64
                ),
            )
        )
    return sorted(frames, key=lambda frame: (frame.sequence_id, frame.frame_id))


def _threshold_metrics(frames: Sequence[_Frame], threshold: float) -> dict[str, Any]:
    gt_counts: Counter[TrackKey] = Counter()
    pred_counts: Counter[TrackKey] = Counter()
    contingency: Counter[tuple[TrackKey, TrackKey]] = Counter()
    last_match: dict[TrackKey, TrackKey] = {}
    tp = fp = fn = id_switches = 0
    matched_iou = 0.0
    for frame in frames:
        for track_id in frame.gt_track_ids.tolist():
            gt_counts[(frame.sequence_id, int(track_id))] += 1
        for track_id in frame.pred_track_ids.tolist():
            pred_counts[(frame.sequence_id, int(track_id))] += 1
        matches = _match_frame(frame, threshold)
        tp += len(matches)
        fp += frame.pred_boxes.shape[0] - len(matches)
        fn += frame.gt_boxes.shape[0] - len(matches)
        for gt_index, pred_index, iou in matches:
            gt_key = (frame.sequence_id, int(frame.gt_track_ids[gt_index]))
            pred_key = (frame.sequence_id, int(frame.pred_track_ids[pred_index]))
            contingency[(gt_key, pred_key)] += 1
            previous = last_match.get(gt_key)
            if previous is not None and previous != pred_key:
                id_switches += 1
            last_match[gt_key] = pred_key
            matched_iou += iou

    deta_denominator = tp + fp + fn
    deta = tp / deta_denominator if deta_denominator else 1.0
    association_sum = 0.0
    for (gt_key, pred_key), count in contingency.items():
        union = gt_counts[gt_key] + pred_counts[pred_key] - count
        association_sum += count * count / union
    assa = association_sum / tp if tp else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_total": sum(gt_counts.values()),
        "pred_total": sum(pred_counts.values()),
        "id_switches": id_switches,
        "contingency": contingency,
        "deta": deta,
        "assa": assa,
        "hota": math.sqrt(deta * assa),
        "motp": matched_iou / tp if tp else 0.0,
    }


def _match_frame(frame: _Frame, threshold: float) -> list[tuple[int, int, float]]:
    if not frame.gt_boxes.numel() or not frame.pred_boxes.numel():
        return []
    iou = _box_iou(frame.gt_boxes, frame.pred_boxes)
    same_class = frame.gt_labels[:, None] == frame.pred_labels[None, :]
    allowed = same_class & (iou >= threshold)
    cost = (1.0 - iou).masked_fill(~allowed, 1e6).cpu().numpy()
    gt_indices, pred_indices = linear_sum_assignment(cost)
    return [
        (int(gt_index), int(pred_index), float(iou[gt_index, pred_index]))
        for gt_index, pred_index in zip(gt_indices, pred_indices, strict=True)
        if bool(allowed[gt_index, pred_index])
    ]


def _global_identity_true_positives(
    contingency: Counter[tuple[TrackKey, TrackKey]],
) -> int:
    by_sequence: dict[str, Counter[tuple[int, int]]] = defaultdict(Counter)
    for (gt_key, pred_key), count in contingency.items():
        by_sequence[gt_key[0]][(gt_key[1], pred_key[1])] += count
    total = 0
    for pairs in by_sequence.values():
        gt_ids = sorted({pair[0] for pair in pairs})
        pred_ids = sorted({pair[1] for pair in pairs})
        if not gt_ids or not pred_ids:
            continue
        matrix = torch.zeros(len(gt_ids), len(pred_ids), dtype=torch.int64)
        gt_lookup = {value: index for index, value in enumerate(gt_ids)}
        pred_lookup = {value: index for index, value in enumerate(pred_ids)}
        for (gt_id, pred_id), count in pairs.items():
            matrix[gt_lookup[gt_id], pred_lookup[pred_id]] = count
        rows, columns = linear_sum_assignment(-matrix.numpy())
        total += sum(
            int(matrix[row, column]) for row, column in zip(rows, columns, strict=True)
        )
    return total


def _box_iou(first: Tensor, second: Tensor) -> Tensor:
    def xyxy(boxes: Tensor) -> Tensor:
        half = boxes[:, 2:] / 2
        return torch.cat((boxes[:, :2] - half, boxes[:, :2] + half), dim=-1)

    first_xyxy = xyxy(first)
    second_xyxy = xyxy(second)
    top_left = torch.maximum(first_xyxy[:, None, :2], second_xyxy[None, :, :2])
    bottom_right = torch.minimum(first_xyxy[:, None, 2:], second_xyxy[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    first_area = (first_xyxy[:, 2:] - first_xyxy[:, :2]).clamp_min(0).prod(-1)
    second_area = (second_xyxy[:, 2:] - second_xyxy[:, :2]).clamp_min(0).prod(-1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return intersection / union.clamp_min(torch.finfo(union.dtype).eps)


def _frame_key(record: TrackingRecord, fallback: int) -> tuple[str, int]:
    return str(record.get("sequence_id", "sequence")), int(
        record.get("frame_id", fallback)
    )


def _boxes(record: TrackingRecord, kind: str) -> Tensor:
    value = record.get("boxes")
    if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[1] != 4:
        raise ValueError(f"{kind}.boxes must be a [N, 4] tensor")
    if not torch.isfinite(value).all() or value.lt(0).any() or value.gt(1).any():
        raise ValueError(f"{kind}.boxes must be finite normalized cxcywh")
    return value.detach().to(device="cpu", dtype=torch.float32)


def _vector(
    record: TrackingRecord,
    name: str,
    length: int,
    dtype: torch.dtype,
) -> Tensor:
    value = record.get(name)
    if not isinstance(value, Tensor) or value.shape != (length,):
        raise ValueError(f"{name} must be a tensor with length {length}")
    return value.detach().to(device="cpu", dtype=dtype)
