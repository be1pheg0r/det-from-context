"""Визуализация. Человек 5.

Отладка памяти по итоговому AP невозможна — нужно видеть, что в слотах.
"""

from __future__ import annotations


def show_batch(batch, context, path: str) -> None:
    """TODO(чел.5). Target + K контекстных кадров в ряд, боксы после
    трансформаций. Первый инструмент, который понадобится Человеку 2 —
    сделать раньше остального."""
    raise NotImplementedError("Человек 5")


def show_predictions(batch, output, path: str) -> None:
    """TODO(чел.5). GT vs predictions на целевом кадре."""
    raise NotImplementedError("Человек 5")


def show_memory(state, output, diagnostics, path: str) -> None:
    """TODO(чел.5). Соответствие слотов памяти объектам: бокс слота, его
    возраст, уверенность, наблюдаемость. Плюс read weights как heatmap
    queries × slots и веса cross-attention.

    Именно здесь видно граблю №3 (слот дрейфует и уезжает от объекта) и
    граблю №1 (новый объект появился, но все queries прилипли к памяти)."""
    raise NotImplementedError("Человек 5")
