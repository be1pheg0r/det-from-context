"""Синхронные трансформации. Человек 2.

Ключевое требование: один сэмпл параметров аугментации применяется ко ВСЕЙ
группе (target + K контекстных кадров + все их боксы). Независимая аугментация
кадров рвёт геометрическое соответствие между ними, и пропагация бокса из
памяти становится бессмысленной.
"""

from __future__ import annotations

from typing import Any


class GroupTransform:
    """TODO(чел.2): протокол трансформации над группой кадров.

    sample_params(rng) -> params;  apply(frame, boxes, params) -> (frame, boxes)
    Разделение на sample/apply — и есть механизм синхронности.
    """

    def __call__(self, group: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raise NotImplementedError("Человек 2")


# TODO(чел.2): RandomHorizontalFlip, RandomResize, Normalize, ToTensor —
# все как GroupTransform. Не тащить torchvision.transforms напрямую:
# они не умеют общий параметр на группу.
