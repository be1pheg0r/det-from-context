"""Имена веток. Единственный источник правды. Человек 1.

Отдельный модуль, потому что он не импортирует ни torch, ни contracts:
валидация конфигов и тесты на неё остаются лёгкими. Добавил ветку — впиши
сюда, иначе конфиг с опечаткой упадёт только на запуске обучения.
"""

from __future__ import annotations

from typing import Literal

DatasetName = Literal["imagenet_vid", "ovis", "dummy"]
DetectorName = Literal["rfdetr", "dummy"]
ContextName = Literal["none", "ema_slot", "cross_attn", "stream_queue", "bridge_ad"]
FusionMode = Literal["residual", "gated_residual", "concat_proj"]
ContextStrategy = Literal["prev_k", "uniform", "random", "empty", "shuffled"]
AttachPoint = Literal[
    "after_backbone", "after_projector", "before_decoder", "decoder_layer"
]


def _values(alias) -> frozenset[str]:
    return frozenset(alias.__args__)


DATASETS = _values(DatasetName)
DETECTORS = _values(DetectorName)
CONTEXT_MODULES = _values(ContextName)
FUSION_MODES = _values(FusionMode)
CONTEXT_STRATEGIES = _values(ContextStrategy)
ATTACH_POINTS = _values(AttachPoint)
