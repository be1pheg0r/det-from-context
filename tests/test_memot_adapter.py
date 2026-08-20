"""Contract and integration tests for the external RF-DETR + MeMOT path."""

from __future__ import annotations

import torch
from torch import Tensor

from context_detection.build import build_model
from context_detection.config import ExperimentConfig
from context_detection.contracts import ContextBatch, DetectionBatch, DetectorOutput
from context_detection.models.detector import DummyDetector
from context_detection.models.memory import MeMOTMemory
from context_detection.models.memot import (
    MeMOTMemoryDecoder,
    MeMOTMemoryEncoder,
    MeMOTTracker,
)


class _SpyDetector(DummyDetector):
    """Record whether temporal queries were injected into the detector."""

    def __init__(self) -> None:
        super().__init__(
            num_queries=4,
            dim=8,
            num_classes=3,
            num_decoder_layers=1,
            num_heads=2,
        )
        self.query_overrides: list[Tensor | None] = []
        self.last_output: DetectorOutput | None = None

    def forward(
        self, batch: DetectionBatch, query_init: Tensor | None = None
    ) -> DetectorOutput:
        self.query_overrides.append(query_init)
        self.last_output = super().forward(batch, query_init)
        return self.last_output


def _memory() -> MeMOTMemory:
    return MeMOTMemory(
        dim=8,
        num_heads=2,
        num_slots=3,
        memory_length=4,
        short_memory_length=2,
        write_threshold=0.0,
        max_missed=2,
        association_iou_threshold=0.1,
        association_cosine_threshold=0.5,
        association_appearance_weight=0.25,
        motion_momentum=0.8,
    )


def _tracker(*, detach_state: bool = True) -> tuple[MeMOTTracker, _SpyDetector]:
    detector = _SpyDetector()
    tracker = MeMOTTracker(
        detector,
        MeMOTMemoryEncoder(_memory()),
        MeMOTMemoryDecoder(
            dim=8,
            num_heads=2,
            num_classes=3,
            num_slots=3,
            num_layers=1,
        ),
        detach_state=detach_state,
    )
    return tracker, detector


def _batch(timestamp: float, *, sequence_start: bool) -> DetectionBatch:
    return DetectionBatch(
        images=torch.rand(1, 3, 16, 16),
        targets=[
            {
                "boxes": torch.empty(0, 4),
                "labels": torch.empty(0, dtype=torch.int64),
            }
        ],
        sequence_id=["sequence"],
        frame_id=torch.tensor([round(timestamp * 10)], dtype=torch.int64),
        timestamp=torch.tensor([timestamp]),
        is_sequence_start=torch.tensor([sequence_start]),
    )


def _context() -> ContextBatch:
    return ContextBatch(
        valid_mask=torch.zeros(1, 0, dtype=torch.bool),
        time_offsets=torch.zeros(1, 0),
    )


def test_memot_runs_after_unmodified_detector_forward() -> None:
    torch.manual_seed(3)
    tracker, detector = _tracker()
    first_output, state = tracker(_batch(0.0, sequence_start=True), _context())
    first_hypotheses = detector.last_output

    assert detector.query_overrides == [None]
    assert first_hypotheses is not None
    torch.testing.assert_close(first_output.logits, first_hypotheses.logits)
    torch.testing.assert_close(first_output.boxes, first_hypotheses.boxes)
    assert state.valid.sum().item() == 3
    assert sorted(state.track_id[state.valid].tolist()) == [0, 1, 2]
    assert first_output.aux["memot"]["association_logits"].shape == (1, 4, 4)
    assert first_output.aux["memot"]["track_ids"].shape == (1, 4)


def test_memot_falls_back_when_detector_has_extra_class_logits() -> None:
    tracker, _ = _tracker()
    tracker.memory_decoder.num_classes = 2
    tracker.memory_decoder.class_delta = torch.nn.Linear(8, 2)

    output, _ = tracker(_batch(0.0, sequence_start=True), _context())

    assert output.logits.shape[-1] == 3


def test_memot_reads_empty_context_batch_from_recurrent_state() -> None:
    torch.manual_seed(5)
    tracker, detector = _tracker()
    _, state = tracker(_batch(0.0, sequence_start=True), _context())
    output, next_state = tracker(_batch(1.0, sequence_start=False), _context(), state)

    association = output.aux["memot"]["association_logits"]
    assert detector.query_overrides == [None, None]
    assert torch.isfinite(association[..., :-1]).any()
    assert output.aux["memot"]["diagnostics"]["active_slots"] == 3
    assert set(output.aux["memot"]["track_ids"].flatten().tolist()) >= {-1, 0, 1, 2}
    assert next_state.next_track_id.item() == 3


def test_memot_association_path_is_differentiable() -> None:
    torch.manual_seed(7)
    tracker, _ = _tracker(detach_state=True)
    _, state = tracker(_batch(0.0, sequence_start=True), _context())
    output, _ = tracker(_batch(1.0, sequence_start=False), _context(), state)
    association = output.aux["memot"]["association_logits"]
    finite_existing = association[..., :-1][torch.isfinite(association[..., :-1])]

    loss = finite_existing.mean() + association[..., -1].mean()
    loss.backward()

    attention = tracker.memory_encoder.memory.query_attention
    assert attention.in_proj_weight.grad is not None
    assert torch.isfinite(attention.in_proj_weight.grad).all()


def test_sequence_reset_restarts_runtime_track_ids() -> None:
    tracker, _ = _tracker()
    _, state = tracker(_batch(0.0, sequence_start=True), _context())
    _, state = tracker(_batch(1.0, sequence_start=False), _context(), state)
    _, reset_state = tracker(_batch(2.0, sequence_start=True), _context(), state)

    assert sorted(reset_state.track_id[reset_state.valid].tolist()) == [0, 1, 2]
    assert reset_state.next_track_id.item() == 3


def test_model_registry_builds_external_memot_tracker() -> None:
    config = ExperimentConfig.model_validate(
        {
            "data": {"clip_len": 2},
            "detector": {
                "name": "dummy",
                "num_queries": 4,
                "dim": 8,
                "num_classes": 3,
                "num_heads": 2,
                "num_decoder_layers": 1,
            },
            "context": {
                "name": "memot",
                "num_slots": 3,
                "memory_length": 4,
                "short_memory_length": 2,
                "write_threshold": 0.0,
                "memory_decoder_layers": 1,
            },
        }
    )

    model = build_model(config)

    assert isinstance(model, MeMOTTracker)
    assert model.detector.__class__ is DummyDetector
