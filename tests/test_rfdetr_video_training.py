"""Tests for clip-aware RF-DETR + MeMOT Lightning semantics."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn

from context_detection.config import ExperimentConfig
from context_detection.contracts import (
    ContextBatch,
    DetectionBatch,
    DetectionClipBatch,
    DetectorOutput,
)
from context_detection.models.detector import DetectorAdapter
from context_detection.models.memory import MeMOTMemory
from context_detection.models.memot import (
    MeMOTMemoryDecoder,
    MeMOTMemoryEncoder,
    MeMOTTracker,
)
from context_detection.models.rfdetr_training import (
    ComponentRFDetrMeMOTModule,
    _clip_to_device,
)


class _Detector(DetectorAdapter):
    """Small differentiable hypothesis generator with upstream-like outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(2, 8))
        self.class_head = nn.Linear(8, 2)
        self.box_head = nn.Linear(8, 4)

    @property
    def dim(self) -> int:
        return 8

    def initial_queries(self, batch: DetectionBatch) -> Tensor:
        return self.query.unsqueeze(0).expand(batch.batch_size, -1, -1)

    def forward(
        self, batch: DetectionBatch, query_init: Tensor | None = None
    ) -> DetectorOutput:
        queries = self.initial_queries(batch) if query_init is None else query_init
        logits = self.class_head(queries)
        boxes = self.box_head(queries).sigmoid()
        upstream = {"pred_logits": logits, "pred_boxes": boxes}
        return DetectorOutput(
            logits=logits,
            boxes=boxes,
            queries=queries,
            reference_points=boxes,
            features=[queries.transpose(1, 2).unsqueeze(-1)],
            aux={"upstream_outputs": upstream},
        )

    def freeze(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


class _Criterion:
    """Stand-in exposing the same matcher/weight surface as RF-DETR."""

    weight_dict = {"loss_ce": 2.0}

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]
    ) -> dict[str, Tensor]:
        del targets
        self.calls += 1
        return {"loss_ce": outputs["pred_logits"].square().mean()}

    @staticmethod
    def matcher(
        outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]
    ) -> list[tuple[Tensor, Tensor]]:
        result: list[tuple[Tensor, Tensor]] = []
        for target in targets:
            count = min(outputs["pred_logits"].shape[1], target["boxes"].shape[0])
            indices = torch.arange(count, device=outputs["pred_logits"].device)
            result.append((indices, indices))
        return result


class _Harness(ComponentRFDetrMeMOTModule):
    """Exercise clip logic without constructing upstream RF-DETR weights."""

    def __init__(self, tracker: MeMOTTracker, criterion: _Criterion) -> None:
        LightningModule.__init__(self)
        self.__dict__["_tracker"] = tracker
        self.criterion = criterion
        self.config = ExperimentConfig.model_validate(
            {
                "data": {"clip_len": 2},
                "detector": {
                    "name": "dummy",
                    "dim": 8,
                    "num_classes": 2,
                    "num_heads": 2,
                },
                "context": {
                    "name": "memot",
                    "num_slots": 2,
                    "memory_length": 3,
                    "short_memory_length": 2,
                },
            }
        )
        self.train_config = SimpleNamespace(
            compute_train_metrics=False,
            train_log_on_step=False,
            train_log_sync_dist=False,
        )
        self.postprocess = lambda outputs, sizes: [
            {"boxes": outputs["pred_boxes"][index], "orig_size": size}
            for index, size in enumerate(sizes)
        ]
        self.logged: dict[str, Tensor] = {}

    def log_dict(
        self, dictionary: dict[str, Tensor], *args: Any, **kwargs: Any
    ) -> None:
        del args, kwargs
        self.logged.update(dictionary)

    def log(self, name: str, value: Tensor, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.logged[name] = value


def _tracker() -> MeMOTTracker:
    memory = MeMOTMemory(
        dim=8,
        num_heads=2,
        num_slots=2,
        memory_length=3,
        short_memory_length=2,
        write_threshold=0.0,
        max_missed=2,
        association_iou_threshold=0.0,
        association_cosine_threshold=-1.0,
        association_appearance_weight=0.25,
        motion_momentum=0.8,
    )
    return MeMOTTracker(
        _Detector(),
        MeMOTMemoryEncoder(memory),
        MeMOTMemoryDecoder(8, 2, 2, 2, num_layers=1),
        detach_state=True,
    )


def _step(timestamp: float, *, start: bool) -> tuple[DetectionBatch, ContextBatch]:
    target = {
        "boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
        "labels": torch.tensor([0], dtype=torch.int64),
        "track_ids": torch.tensor([0], dtype=torch.int64),
        "orig_size": torch.tensor([16, 16], dtype=torch.int64),
        "size": torch.tensor([16, 16], dtype=torch.int64),
    }
    detection = DetectionBatch(
        images=torch.rand(1, 3, 16, 16),
        targets=[target],
        sequence_id=["sequence"],
        frame_id=torch.tensor([round(timestamp * 10)], dtype=torch.int64),
        timestamp=torch.tensor([timestamp]),
        is_sequence_start=torch.tensor([start]),
    )
    context = ContextBatch(
        valid_mask=torch.zeros(1, 0, dtype=torch.bool),
        time_offsets=torch.zeros(1, 0),
    )
    return detection, context


def _clip(mode: str) -> DetectionClipBatch:
    supervision = (
        torch.tensor([[True], [True]])
        if mode == "tracking"
        else torch.tensor([[False], [True]])
    )
    return DetectionClipBatch(
        steps=[_step(0.0, start=True), _step(1.0, start=False)],
        supervision_mask=supervision,
        mode=mode,
    )


def test_tracking_clip_combines_native_and_memot_losses() -> None:
    torch.manual_seed(11)
    criterion = _Criterion()
    module = _Harness(_tracker(), criterion)

    loss = module.training_step(_clip("tracking"), 0)
    assert isinstance(loss, Tensor)
    loss.backward()

    assert criterion.calls == 2
    assert "train/loss_detection" in module.logged
    assert "train/loss_association" in module.logged
    assert "train/loss_uniqueness" in module.logged
    decoder = module.tracker.memory_decoder
    assert decoder.new_track_logit.weight.grad is not None


def test_reference_frame_clip_only_supervises_final_detection() -> None:
    criterion = _Criterion()
    module = _Harness(_tracker(), criterion)

    result = module.validation_step(_clip("reference_frame"), 0)

    assert criterion.calls == 1
    assert "val/loss_detection" in module.logged
    assert "val/loss_association" not in module.logged
    assert len(result["results"]) == 1


def test_clip_contract_moves_targets_and_supervision_together() -> None:
    clip = _clip("tracking")

    moved = _clip_to_device(clip, torch.device("cpu"))

    assert moved.supervision_mask.device.type == "cpu"
    for detection, context in moved.steps:
        assert detection.images.device.type == "cpu"
        assert detection.targets[0]["track_ids"].device.type == "cpu"
        assert context.valid_mask.device.type == "cpu"
