"""Фабрики из конфига. Человек 1.

Конфиги — JSON (stdlib, без pyyaml). Наследование через ключ "_base_".
Ветки регистрируются здесь, чтобы Человек 5 запускал их одинаково, а Человек 4
добавлял новую ветку одной строкой.
"""

from __future__ import annotations

from typing import Any

# Единственный источник правды по именам. Импортируется без torch, поэтому
# тесты могут сверять с ним конфиги. Добавил ветку — впиши сюда, иначе
# конфиг с опечаткой упадёт только на запуске обучения.
DATASETS = frozenset({"imagenet_vid", "ovis", "dummy"})
DETECTORS = frozenset({"rfdetr", "dummy"})
CONTEXT_MODULES = frozenset(
    {"none", "ema_slot", "cross_attn", "stream_queue", "bridge_ad"}
)
FUSION_MODES = frozenset({"residual", "gated_residual", "concat_proj"})
CONTEXT_STRATEGIES = frozenset({"prev_k", "uniform", "random", "empty", "shuffled"})


def load_config(path: str, overrides: list[str] | None = None) -> dict[str, Any]:
    """TODO(чел.1). Рекурсивно раскрыть "_base_", применить overrides вида
    "model.context.k=8" — иначе Человек 5 не сможет прогнать ablation по
    длине истории одной командой."""
    raise NotImplementedError("Человек 1")


def build_dataset(cfg: dict[str, Any], split: str):
    """TODO(чел.1). imagenet_vid | ovis | dummy"""
    raise NotImplementedError("Человек 1")


def build_detector(cfg: dict[str, Any]):
    """TODO(чел.1). rfdetr | dummy"""
    raise NotImplementedError("Человек 1")


def build_context_module(cfg: dict[str, Any]):
    """TODO(чел.1). none | ema_slot | cross_attn | stream_queue | bridge_ad"""
    raise NotImplementedError("Человек 1")


def build_model(cfg: dict[str, Any]):
    """TODO(чел.1). Собрать ContextDetector. DoD Человека 1:

        model = build_model(cfg)
        output = model(*make_dummy_batch())

    работает без датасета и без RF-DETR.
    """
    raise NotImplementedError("Человек 1")


def make_dummy_batch(batch_size: int = 2, k: int = 4):
    """TODO(чел.1). Случайные тензоры правильных форм — DetectionBatch +
    ContextBatch. Заполнить valid_mask частично (не все True), иначе
    маскирование у Человека 4 никогда не проверится."""
    raise NotImplementedError("Человек 1")
