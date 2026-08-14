"""collate_fn. Человек 2."""

from __future__ import annotations

from typing import Any

from ..contracts import ContextBatch, DetectionBatch


def collate_fn(samples: list[dict[str, Any]]) -> tuple[DetectionBatch, ContextBatch]:
    """Собрать сэмплы в (DetectionBatch, ContextBatch).

    TODO(чел.2):
      * контекст паддится до одинакового K внутри батча, лишнее закрывается
        valid_mask — переменное K не должно доходить до модели;
      * изображения разного размера: NestedTensor / padding + pixel mask,
        как в DETR. Решить один раз и записать в документ Человека 1;
      * targets остаются списком словарей переменной длины, не тензором;
      * is_sequence_start проставляется здесь из SequenceIndex — от него
        зависит сброс памяти.
    """
    raise NotImplementedError("Человек 2")
