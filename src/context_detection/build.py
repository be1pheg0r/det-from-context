"""Фабрики из конфига + dummy-батч. Человек 1."""

from __future__ import annotations

import torch

from .config import ExperimentConfig, load_config
from .contracts import ContextBatch, DetectionBatch
from .models.detector import DetectorAdapter, DummyDetector
from .models.memory import (
    BridgeADMemory,
    ContextCrossAttention,
    ContextModule,
    EMASlot,
    NoContext,
    StreamQueue,
)
from .models.wrapper import ContextDetector

__all__ = [
    "ContextDetector",
    "ExperimentConfig",
    "build_context_module",
    "build_detector",
    "build_model",
    "load_config",
    "make_dummy_batch",
]

_CONTEXT_MODULES: dict[str, type[ContextModule]] = {
    "none": NoContext,
    "ema_slot": EMASlot,
    "cross_attn": ContextCrossAttention,
    "stream_queue": StreamQueue,
    "bridge_ad": BridgeADMemory,
}


def build_detector(cfg: ExperimentConfig) -> DetectorAdapter:
    d = cfg.detector
    if d.name == "dummy":
        detector = DummyDetector(
            num_queries=d.num_queries,
            dim=d.dim,
            num_classes=d.num_classes,
            num_decoder_layers=d.num_decoder_layers,
            num_heads=d.num_heads,
        )
    else:  # rfdetr
        from .models.rfdetr import RFDetrAdapter

        detector = RFDetrAdapter(variant=d.variant, weights=d.weights)

    if d.freeze_backbone or d.freeze_decoder:
        detector.freeze(backbone=d.freeze_backbone, decoder=d.freeze_decoder)
    return detector


def build_context_module(cfg: ExperimentConfig, dim: int) -> ContextModule:
    name = cfg.context.name
    cls = _CONTEXT_MODULES[name]  # имя уже провалидировано Literal'ом в конфиге
    if name == "none":
        return cls()
    # TODO(чел.4): у веток появятся собственные аргументы (num_slots, write_gate,
    # motion, horizon) — прокидывать отсюда, конфиг их уже несёт и валидирует.
    return cls()


def build_model(cfg: ExperimentConfig) -> ContextDetector:
    detector = build_detector(cfg)
    return ContextDetector(
        detector=detector,
        context_module=build_context_module(cfg, detector.dim),
        fusion=cfg.context.fusion,
        detach_state=cfg.train.detach_state,
    )


def build_dataset(cfg: ExperimentConfig, split: str):
    """TODO(чел.2): imagenet_vid | ovis. Для "dummy" датасета нет — там
    make_dummy_batch."""
    raise NotImplementedError("Человек 2")


def make_dummy_batch(
    batch_size: int = 2,
    context_k: int = 4,
    image_size: int = 64,
    num_boxes: int = 3,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[DetectionBatch, ContextBatch]:
    """Случайный батч правильной формы — без датасета и без RF-DETR.

    valid_mask заполняется частично (первый элемент батча имитирует начало
    последовательности), иначе маскирование у Человека 4 никогда не проверится
    на dummy-прогоне и всплывёт только на реальных данных.
    """

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, device=device, generator=generator)

    boxes = rand(batch_size, num_boxes, 4) * 0.5 + 0.25  # заведомо внутри [0, 1]
    targets = [
        {
            "boxes": boxes[i],
            "labels": torch.randint(
                0, 31, (num_boxes,), device=device, generator=generator
            ),
        }
        for i in range(batch_size)
    ]

    valid = torch.ones(batch_size, context_k, dtype=torch.bool, device=device)
    if batch_size and context_k:
        valid[0, : max(1, context_k // 2)] = False  # начало последовательности

    offsets = (
        torch.arange(context_k, 0, -1, device=device, dtype=torch.float32).expand(
            batch_size, context_k
        )
        * 0.04
    )  # ~25 fps
    offsets = offsets * valid

    batch = DetectionBatch(
        images=rand(batch_size, 3, image_size, image_size),
        targets=targets,
        sequence_id=[f"seq{i}" for i in range(batch_size)],
        frame_id=torch.full((batch_size,), context_k, dtype=torch.int64, device=device),
        timestamp=torch.full((batch_size,), context_k * 0.04, device=device),
        is_sequence_start=torch.zeros(batch_size, dtype=torch.bool, device=device),
    )
    context = ContextBatch(
        images=rand(batch_size, context_k, 3, image_size, image_size)
        if context_k
        else None,
        valid_mask=valid,
        time_offsets=offsets,
    )
    return batch, context
