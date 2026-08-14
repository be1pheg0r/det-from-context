"""Интерфейс детектора + заглушка для dummy pipeline. Человек 1 (протокол) / 3.

Протокол согласуется Человеком 1 с Человеком 3 ДО начала работы над RF-DETR.
"""

from __future__ import annotations

from typing import Protocol

from ..contracts import ContextBatch, ContextOutput, DetectionBatch, DetectorOutput


class DetectorAdapter(Protocol):
    """Детектор, из которого можно вытащить точки подключения контекста."""

    def forward(
        self,
        batch: DetectionBatch,
        context: ContextOutput | None = None,
    ) -> DetectorOutput:
        """context=None должно быть строго эквивалентно оригинальному детектору.
        Это проверяется regression-тестом Человека 3."""
        ...

    def encode_context_frames(self, context: ContextBatch) -> list:
        """Прогнать контекстные кадры через backbone/projector.

        Отдельный метод, потому что для рекуррентных веток он не вызывается
        вовсе (там контекст берётся из MemoryState, а не из пикселей), а для
        cross-attention ветки вызывается с no_grad или с общим backbone —
        это решение конфига, а не жёстко зашитое.
        """
        ...

    def freeze(self, backbone: bool = True, decoder: bool = False) -> None:
        """Режимы: замороженный backbone / только context-блок / полный FT."""
        ...


class DummyDetector:
    """Случайные тензоры правильной формы. Человек 1.

    Нужен, чтобы Люди 4 и 5 могли работать до готовности RF-DETR.
    TODO(чел.1): заполнять ВСЕ поля DetectorOutput, включая decoder_layers —
    иначе memory-ветки на dummy не проверить.
    """

    def __init__(self, num_queries: int = 300, dim: int = 256, num_classes: int = 31):
        raise NotImplementedError("Человек 1")
