"""Датасеты. Человек 2.

ImageNet VID — основной (длинные последовательности, плавное движение).
OVIS — стресс-тест на тяжёлые окклюзии, там же instance id и разметка
видимости, нужная Человеку 5 для метрики «AP под окклюзией».
"""

from __future__ import annotations

from typing import Any

from .sequence_index import SequenceIndex


class SequenceDetectionDataset:
    """Базовый класс: индекс кадров + выбор контекста + загрузка изображений.

    TODO(чел.2):
      * __getitem__ возвращает НЕ батч, а один сэмпл (target + context кадры
        + их аннотации + ContextSelection); сборку делает collate_fn;
      * аугментации применяются одним сэмплом на всю группу кадров, иначе
        перенос бокса между кадрами теряет смысл;
      * боксы конвертируются в cxcywh нормализованные прямо здесь — дальше по
        пайплайну COCO-формат не появляется.
    """

    def __init__(self, index: SequenceIndex, context_k: int, strategy: str) -> None:
        raise NotImplementedError("Человек 2")

    def __len__(self) -> int:
        raise NotImplementedError("Человек 2")

    def __getitem__(self, pos: int) -> dict[str, Any]:
        raise NotImplementedError("Человек 2")


def build_imagenet_vid(root: str, split: str) -> SequenceIndex:
    """TODO(чел.2): распарсить аннотации VID в SequenceIndex.

    timestamp: у VID нет реального времени — использовать frame_id / fps.
    Важно, чтобы time_offsets были в одних единицах с OVIS.
    """
    raise NotImplementedError("Человек 2")


def build_ovis(root: str, split: str) -> SequenceIndex:
    """TODO(чел.2): OVIS. Сохранить в payload instance id и флаг видимости —
    без них не посчитать AP под окклюзией и AP на новых объектах."""
    raise NotImplementedError("Человек 2")


class SequentialClipSampler:
    """Отдаёт кадры по порядку внутри последовательности. Человек 2.

    Рекуррентным веткам нужен последовательный проход, а не обычный shuffle:
    память переносится между соседними шагами. Перемешивать можно только
    порядок последовательностей, но не кадров внутри.
    """

    def __init__(self, index: SequenceIndex, clip_len: int) -> None:
        raise NotImplementedError("Человек 2")
