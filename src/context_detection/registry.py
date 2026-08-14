"""Имена веток. Единственный источник правды. Человек 1.

Отдельный модуль, потому что он не импортирует ни torch, ни contracts:
валидация конфигов и тесты на неё остаются лёгкими. Добавил ветку — впиши
сюда, иначе конфиг с опечаткой упадёт только на запуске обучения.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

DatasetName = Literal["imagenet_vid", "ovis", "dummy"]
DetectorName = Literal["rfdetr", "dummy"]
ContextName = Literal["none", "ema_slot", "cross_attn", "stream_queue", "bridge_ad"]
FusionMode = Literal["residual", "gated_residual", "concat_proj"]
ContextStrategy = Literal["prev_k", "uniform", "random", "empty", "shuffled"]
AttachPoint = Literal[
    "after_backbone", "after_projector", "before_decoder", "decoder_layer"
]


def _values(alias: Any) -> frozenset[str]:
    return frozenset(get_args(alias))


DATASETS = _values(DatasetName)
DETECTORS = _values(DetectorName)
CONTEXT_MODULES = _values(ContextName)
FUSION_MODES = _values(FusionMode)
CONTEXT_STRATEGIES = _values(ContextStrategy)
ATTACH_POINTS = _values(AttachPoint)

#: Ветки, которым нужны пиксели контекстных кадров, а не MemoryState. Живёт
#: здесь, а не в config.py, потому что то же знание нужно и модулям (там оно
#: становится атрибутом класса ContextModule.needs_context_frames), а registry —
#: единственный модуль, который импортируют обе стороны и который не тянет torch.
#: Согласованность с классами проверяет test_needs_context_frames_matches_registry.
NEEDS_CONTEXT_FRAMES = frozenset({"cross_attn"})
