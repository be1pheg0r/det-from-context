"""Выбор контекстных кадров из истории. Человек 2.

Тоже без torch: на выходе — позиции и флаги, тензоры собирает collate.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .sequence_index import SequenceIndex


@dataclass
class ContextSelection:
    """Ровно K слотов. Недостающие добиты заглушками с valid=False."""

    positions: list[int]  # len K; для невалидных слотов значение игнорируется
    valid: list[bool]  # len K
    time_offsets: list[float]  # len K, секунды в прошлое, > 0; 0 для невалидных


def sample_context(
    index: SequenceIndex,
    pos: int,
    k: int,
    strategy: str,
    rng: Random,
) -> ContextSelection:
    """Выбрать K контекстных кадров для кадра `pos`.

    TODO(чел.2) стратегии:
      * "prev_k"   — последние K кадров истории
      * "uniform"  — K равномерно по всей истории
      * "random"   — K случайных из истории без повторов
      * "empty"    — контроль, все слоты невалидные
      * "shuffled" — кадры "prev_k" переставлены, а time_offsets оставлены в
                     исходном порядке. Просто перемешать порядок недостаточно:
                     модель получает time_offsets явно, поэтому контролем
                     является разрыв соответствия «кадр ↔ его время»

    Требования ко всем стратегиям:
      * длина результата ровно K, паддинг valid=False;
      * порядок валидных слотов — от старых к новым (кроме "shuffled");
      * ничего из будущего: источник кандидатов — только `index.history(pos)`.
    """
    raise NotImplementedError("Человек 2")
