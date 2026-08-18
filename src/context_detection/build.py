"""Фабрики из конфига + dummy-батч. Человек 1."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig, load_config
from .contracts import ContextBatch, DetectionBatch
from .data.protocols import build_dataloader
from .models.detector import DetectorAdapter
from .models.memory import ContextModule
from .models.protocols import _CONTEXT_MODULES as _MODEL_CONTEXT_MODULES
from .models.protocols import (
    build_context_module as build_registered_context_module,
)
from .models.protocols import (
    build_registered_detector,
    build_registered_model,
)
from .models.wrapper import ContextDetector

_CONTEXT_MODULES = _MODEL_CONTEXT_MODULES

__all__ = [
    "ContextDetector",
    "ExperimentConfig",
    "build_context_module",
    "build_dataset",
    "build_detector",
    "build_model",
    "load_config",
    "make_dummy_batch",
]


def build_detector(cfg: ExperimentConfig) -> DetectorAdapter:
    """Совместимый detector-only API поверх model protocol registry."""
    return build_registered_detector(cfg)


def build_context_module(cfg: ExperimentConfig) -> ContextModule:
    """Совместимый API сборки context module."""
    return build_registered_context_module(cfg)


def build_model(cfg: ExperimentConfig) -> nn.Module:
    """Построить публичный nn.Module endpoint через model protocol."""
    return build_registered_model(cfg)


def build_dataset(cfg: ExperimentConfig, split: str) -> DataLoader[Any]:
    """Построить публичный DataLoader endpoint через dataset protocol."""
    return build_dataloader(cfg, split)


def make_dummy_batch(
    batch_size: int = 2,
    context_k: int = 4,
    image_size: int = 64,
    num_boxes: int = 3,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[DetectionBatch, ContextBatch]:
    """Случайный батч правильной формы — без датасета и без RF-DETR.

    Батч намеренно неоднородный, иначе на dummy-прогоне не проверится ничего
    из того, ради чего он существует, и всплывёт только на реальных данных:

      * элемент 0 — первый кадр сцены: истории нет (все слоты невалидны),
        frame_id=0, is_sequence_start=True. Единственное, что заставляет
        ContextDetector зайти в ветку reset — без него утечка памяти между
        сценами не ловится ничем;
      * элемент 1 — частичная история: половина слотов заглушки. Проверяет
        маскирование у Человека 4;
      * остальные — полная история.
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
    start = torch.zeros(batch_size, dtype=torch.bool, device=device)
    if batch_size and context_k:
        valid[0] = False  # первый кадр сцены: истории нет...
        start[0] = True  # ...значит память для него надо сбросить
        if batch_size > 1:
            valid[1, : max(1, context_k // 2)] = False  # частичная история

    offsets = (
        torch.arange(context_k, 0, -1, device=device, dtype=torch.float32).expand(
            batch_size, context_k
        )
        * 0.04
    )  # ~25 fps
    offsets = offsets * valid

    # Сколько кадров сцены уже прошло = сколько слотов истории заполнено.
    # Иначе frame_id противоречит valid_mask, и первый же тест Человека 2 на
    # согласованность истории с индексом упрётся в кривую фикстуру.
    frame_id = valid.sum(1)

    batch = DetectionBatch(
        images=rand(batch_size, 3, image_size, image_size),
        targets=targets,
        sequence_id=[f"seq{i}" for i in range(batch_size)],
        frame_id=frame_id,
        timestamp=frame_id.to(torch.float32) * 0.04,
        is_sequence_start=start,
    )
    context = ContextBatch(
        images=rand(batch_size, context_k, 3, image_size, image_size)
        if context_k
        else None,
        valid_mask=valid,
        time_offsets=offsets,
    )
    return batch, context
