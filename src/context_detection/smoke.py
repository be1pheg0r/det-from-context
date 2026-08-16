"""CPU smoke-suite для MeMOT memory encoder."""

from __future__ import annotations

from pathlib import Path

import torch

from .build import build_model, make_dummy_batch
from .config import ExperimentConfig, load_config
from .contracts import ContextBatch, DetectorOutput
from .models.memory import MeMOTMemory, MeMOTState

DIM = 8
NUM_HEADS = 2


def _module(
    *,
    num_slots: int = 4,
    memory_length: int = 5,
    write_threshold: float = 0.5,
    max_missed: int = 2,
) -> MeMOTMemory:
    """Построить малый детерминированный memory encoder для smoke."""
    torch.manual_seed(0)
    return MeMOTMemory(
        dim=DIM,
        num_heads=NUM_HEADS,
        num_slots=num_slots,
        memory_length=memory_length,
        short_memory_length=min(2, memory_length),
        write_threshold=write_threshold,
        max_missed=max_missed,
        association_iou_threshold=0.1,
        association_cosine_threshold=0.5,
        association_appearance_weight=0.25,
        motion_momentum=0.8,
    )


def _context(
    batch_size: int, *, valid: bool = True, offset: float = 1.0
) -> ContextBatch:
    """Создать рекуррентный контекст с заданным временным шагом."""
    valid_mask = torch.full((batch_size, 1), valid, dtype=torch.bool)
    return ContextBatch(
        valid_mask=valid_mask,
        time_offsets=valid_mask.to(torch.float32) * offset,
    )


def _output(
    features: torch.Tensor,
    boxes: torch.Tensor,
    *,
    logit: float = 10.0,
) -> DetectorOutput:
    """Собрать синтетический выход детектора из явных features и boxes."""
    batch_size, num_queries = features.shape[:2]
    logits = torch.full((batch_size, num_queries, 3), logit)
    return DetectorOutput(
        logits=logits,
        boxes=boxes,
        queries=features,
        reference_points=boxes.clone(),
    )


def _assert_first_read_and_masking() -> None:
    """Проверить baseline первого кадра и безопасное пустое attention."""
    module = _module()
    queries = torch.randn(2, 4, DIM)
    result = module.read(queries, None, _context(2, valid=False))
    assert result.memory_state is None
    assert torch.equal(result.query_delta, torch.zeros_like(queries))

    features = torch.randn(2, 4, DIM)
    boxes = torch.full((2, 4, 4), 0.5)
    state = module.write(None, _output(features, boxes), _context(2))
    state.history_valid[0] = False
    state.valid[0] = False
    result = module.read(
        queries,
        state,
        _context(2),
        current_timestamp=torch.ones(2),
    )
    assert torch.isfinite(result.query_delta).all()
    assert result.query_delta[0].count_nonzero() == 0


def _assert_query_permutation_keeps_track_slots() -> None:
    """Проверить независимость track slots от перестановки DETR queries."""
    module = _module(num_slots=2)
    first_features = torch.zeros(1, 2, DIM)
    first_features[0, 0, 0] = 1
    first_features[0, 1, 1] = 1
    first_boxes = torch.tensor([[[0.2, 0.5, 0.1, 0.2], [0.8, 0.5, 0.1, 0.2]]])
    state = module.write(
        None,
        _output(first_features, first_boxes),
        _context(1),
        current_timestamp=torch.tensor([0.0]),
    )

    second_features = first_features.flip(1).clone()
    second_features[0, 0, 2] = 0.1
    second_features[0, 1, 3] = 0.1
    second_boxes = first_boxes.flip(1)
    state = module.write(
        state,
        _output(second_features, second_boxes),
        _context(1),
        current_timestamp=torch.tensor([1.0]),
    )
    assert torch.equal(state.box[0, 0], second_boxes[0, 1])
    assert torch.equal(state.box[0, 1], second_boxes[0, 0])
    assert torch.equal(state.feature[0, 0], second_features[0, 1])
    assert torch.equal(state.feature[0, 1], second_features[0, 0])


def _assert_time_and_motion_are_real() -> None:
    """Проверить реальные timestamps, motion и временное кодирование."""
    module = _module(num_slots=1)
    features = torch.zeros(1, 1, DIM)
    features[0, 0, 0] = 1
    first_box = torch.tensor([[[0.2, 0.5, 0.1, 0.2]]])
    state = module.write(
        None,
        _output(features, first_box),
        _context(1),
        current_timestamp=torch.tensor([10.0]),
    )
    second_box = torch.tensor([[[0.3, 0.5, 0.1, 0.2]]])
    state = module.write(
        state,
        _output(features, second_box),
        _context(1),
        current_timestamp=torch.tensor([12.0]),
    )
    assert state.timestamp.item() == 12.0
    assert state.motion is not None and state.motion.abs().sum() > 0

    queries = torch.randn(1, 2, DIM)
    near = module.read(
        queries, state, _context(1), current_timestamp=torch.tensor([13.0])
    )
    far = module.read(
        queries, state, _context(1), current_timestamp=torch.tensor([30.0])
    )
    assert not torch.allclose(near.query_delta, far.query_delta)
    wider_context = module.read(
        queries,
        state,
        _context(1, offset=10.0),
        current_timestamp=torch.tensor([13.0]),
    )
    assert not torch.allclose(near.query_delta, wider_context.query_delta)
    try:
        module.read(
            queries,
            state,
            _context(1),
            current_timestamp=torch.tensor([11.0]),
        )
    except ValueError as error:
        assert "идти назад" in str(error)
    else:
        raise AssertionError("немонотонный timestamp должен быть отклонён")


def _assert_expiry_clears_identity_state() -> None:
    """Проверить expiry и очистку identity-specific DMAT/history."""
    module = _module(num_slots=1, max_missed=1)
    features = torch.randn(1, 1, DIM)
    boxes = torch.full((1, 1, 4), 0.5)
    state = module.write(
        None,
        _output(features, boxes),
        _context(1),
        current_timestamp=torch.tensor([0.0]),
    )
    read = module.read(
        torch.randn(1, 1, DIM),
        state,
        _context(1),
        current_timestamp=torch.tensor([0.5]),
    )
    assert isinstance(read.memory_state, MeMOTState)
    assert read.memory_state.dmat.count_nonzero() > 0

    missing = _output(torch.randn(1, 1, DIM), boxes, logit=-10.0)
    state = module.write(
        read.memory_state,
        missing,
        _context(1),
        current_timestamp=torch.tensor([1.0]),
    )
    assert state.valid.item() and state.missed.item() == 1
    state = module.write(
        state,
        missing,
        _context(1),
        current_timestamp=torch.tensor([2.0]),
    )
    assert not state.valid.item()
    assert state.dmat.count_nonzero() == 0
    assert state.history_valid.count_nonzero() == 0
    assert state.evicted.item() == 1

    new_state = module.write(
        state,
        _output(torch.randn(1, 1, DIM), boxes),
        _context(1),
        current_timestamp=torch.tensor([3.0]),
    )
    assert new_state.valid.item() and new_state.age.item() == 1
    assert new_state.dmat.count_nonzero() == 0


def _assert_backward_reaches_memory_parameters() -> None:
    """Проверить градиенты всех обучаемых частей memory encoder."""
    module = _module(write_threshold=0.0)
    previous = torch.randn(2, 4, DIM, requires_grad=True)
    boxes = torch.full((2, 4, 4), 0.5)
    state = module.write(
        None,
        _output(previous, boxes),
        _context(2),
        current_timestamp=torch.zeros(2),
    )
    current = torch.randn(2, 4, DIM, requires_grad=True)
    result = module.read(
        current,
        state,
        _context(2),
        current_timestamp=torch.ones(2),
    )
    result.query_delta.square().mean().backward()
    missing = [
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not missing, f"градиент не дошёл до: {missing}"
    assert previous.grad is not None and current.grad is not None


def _assert_wrapper_two_steps() -> None:
    """Проверить два шага общего ContextDetector с точными timestamps."""
    config = ExperimentConfig.model_validate(
        {
            "data": {"clip_len": 2},
            "detector": {
                "dim": DIM,
                "num_heads": NUM_HEADS,
                "num_queries": 4,
                "num_classes": 3,
                "num_decoder_layers": 1,
            },
            "context": {
                "name": "memot",
                "fusion": "gated_residual",
                "num_slots": 4,
                "memory_length": 5,
                "short_memory_length": 2,
                "write_threshold": 0.0,
            },
        }
    )
    model = build_model(config)
    batch, context = make_dummy_batch(batch_size=2, context_k=2, image_size=16)
    _, state = model(batch, context)
    assert isinstance(state, MeMOTState)
    assert not state.history_feature.requires_grad

    next_batch = batch.model_copy(
        update={
            "is_sequence_start": torch.zeros(2, dtype=torch.bool),
            "frame_id": batch.frame_id + 1,
            "timestamp": batch.timestamp + 0.04,
        }
    )
    output, next_state = model(next_batch, context, state)
    assert isinstance(next_state, MeMOTState)
    assert torch.isfinite(output.logits).all()
    output.logits.square().mean().backward()
    assert model.context_module.initial_dmat.grad is not None


def run(config_path: str | Path | None = None) -> None:
    """Запустить полный CPU smoke-suite.

    Args:
        config_path: Необязательный MeMOT-конфиг для проверки загрузки.
    """
    if config_path is not None:
        config = load_config(
            config_path,
            [
                "data.name=dummy",
                "data.root=null",
                "data.clip_len=2",
                "detector.name=dummy",
                "detector.variant=null",
                "detector.freeze_backbone=false",
            ],
        )
        assert config.context.name == "memot"

    checks = (
        _assert_first_read_and_masking,
        _assert_query_permutation_keeps_track_slots,
        _assert_time_and_motion_are_real,
        _assert_expiry_clears_identity_state,
        _assert_backward_reaches_memory_parameters,
        _assert_wrapper_two_steps,
    )
    for check in checks:
        check()
    print(f"MeMOT smoke: {len(checks)} checks passed")
