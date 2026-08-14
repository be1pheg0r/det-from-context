"""Обучение, валидация, checkpoint. Человек 5."""

from __future__ import annotations

from typing import Any


def train_one_epoch(model, loader, optimizer, cfg) -> dict[str, float]:
    """TODO(чел.5).

    Отличие от обычного детекционного цикла: батч — это КЛИП, а не набор
    независимых кадров. Внутри клипа состояние памяти переносится с шага на
    шаг, backward — один на клип (collective average loss, MOTR).

    Обязательно: gradient accumulation, AMP (autocast только вокруг forward,
    память в float32), state.detach() между шагами.

    ⚠️ Грабля №5: клип длиной 2-3 кадра дёшев, но именно он создаёт
    train–inference gap — на инференсе последовательность в сотни кадров.
    Длина клипа должна быть параметром конфига и попасть в ablation.
    """
    raise NotImplementedError("Человек 5")


def evaluate(model, loader, cfg) -> dict[str, Any]:
    """TODO(чел.5). Инференс идёт строго последовательно по кадрам внутри
    последовательности, память не сбрасывается до смены sequence_id."""
    raise NotImplementedError("Человек 5")


def save_checkpoint(path, model, optimizer, epoch, cfg) -> None:
    """TODO(чел.5). MemoryState в checkpoint НЕ сохраняем — это runtime-состояние."""
    raise NotImplementedError("Человек 5")


def load_checkpoint(path, model, optimizer=None):
    """TODO(чел.5)."""
    raise NotImplementedError("Человек 5")
