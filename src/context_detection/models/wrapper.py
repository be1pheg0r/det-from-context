"""Склейка детектора и context-модуля. Человек 1.

Единственное место, где они встречаются. Ни детектор, ни память друг о друге
не знают.
"""

from __future__ import annotations

from ..contracts import ContextBatch, DetectionBatch, DetectorOutput, MemoryState


class ContextDetector:
    """TODO(чел.1). Порядок шагов на одном кадре:

      1. reset памяти по batch.is_sequence_start
      2. encode_context_frames  — только если ветка этого требует
      3. context_module.read    — получить query_delta
      4. detector.forward(batch, context)
      5. context_module.write   — обновить состояние
      6. state.detach()         — обрыв BPTT между шагами клипа

    Шаг 6 обязателен: без него граф растёт на всю последовательность и
    обучение на клипах длиной больше 3-4 кадров не влезет в память.

    Тонкость: read вызывается ДО forward, но queries для read берутся из
    инициализации детектора, а не из его выхода. Либо детектор отдаёт
    начальные queries отдельным методом, либо read вызывается внутри
    forward через callback. Решить Человеку 1 вместе с Человеком 3 —
    это влияет на сигнатуру DetectorAdapter.
    """

    def __init__(self, detector, context_module, fusion: str = "residual") -> None:
        raise NotImplementedError("Человек 1")

    def forward(
        self,
        batch: DetectionBatch,
        context: ContextBatch,
        state: MemoryState | None = None,
    ) -> tuple[DetectorOutput, MemoryState | None]:
        raise NotImplementedError("Человек 1")
