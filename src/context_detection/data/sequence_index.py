"""Sequence-aware индекс кадров. Человек 2.

Чистый stdlib, без torch — это позволяет тестировать логику истории отдельно
от пайплайна. Границы сцен = смена sequence_id, отдельного механизма не нужно.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FrameRef:
    """Один кадр. `payload` — всё, что нужно датасету (путь, id аннотаций)."""

    sequence_id: str
    frame_id: int
    timestamp: float
    payload: Any = None


class SequenceIndex:
    """Плоский список кадров + группировка по последовательностям.

    TODO(чел.2):
      * сортировка по (sequence_id, frame_id) — позиция в индексе != позиция
        во входном списке;
      * запретить дубликаты (sequence_id, frame_id) — они ломают однозначность
        истории;
      * хранить границы последовательностей, а не список позиций на каждую.
    """

    def __init__(self, frames: list[FrameRef]) -> None:
        raise NotImplementedError("Человек 2")

    def __len__(self) -> int:
        raise NotImplementedError("Человек 2")

    def __getitem__(self, pos: int) -> FrameRef:
        raise NotImplementedError("Человек 2")

    def history(self, pos: int) -> list[int]:
        """Позиции кадров той же последовательности строго раньше `pos`,
        от старых к новым. Пустой список = начало последовательности.

        ⚠️ Это единственное место, где определяется отсутствие утечки будущего.
        Всё, что видит контекст, проходит отсюда. Тест на это — обязателен.
        """
        raise NotImplementedError("Человек 2")

    def is_sequence_start(self, pos: int) -> bool:
        """Нужен для `DetectionBatch.is_sequence_start` → сброса памяти."""
        raise NotImplementedError("Человек 2")
