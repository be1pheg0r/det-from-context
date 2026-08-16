"""Конфиги на Pydantic. Человек 1.

Валидация имён веток — через Literal из registry, поэтому опечатка в конфиге
падает при загрузке, а не через полчаса обучения. Модуль не импортирует torch:
конфиг можно проверить без окружения для обучения.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .registry import (
    NEEDS_CONTEXT_FRAMES,
    ContextName,
    ContextStrategy,
    DatasetName,
    DetectorName,
    FusionMode,
)


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Section):
    name: DatasetName = "dummy"
    root: str | None = None
    context_k: int = Field(4, ge=0)
    context_strategy: ContextStrategy = "prev_k"
    clip_len: int = Field(4, ge=1)
    image_size: int = Field(224, gt=0)

    @model_validator(mode="after")
    def _check(self) -> DataConfig:
        if self.name != "dummy" and not self.root:
            raise ValueError(f"датасету {self.name!r} нужен root")
        if self.context_strategy == "empty" and self.context_k:
            raise ValueError(
                "strategy='empty' с context_k > 0: слоты будут всегда невалидны. "
                "Для baseline без контекста ставь context_k=0"
            )
        return self


class DetectorConfig(_Section):
    name: DetectorName = "dummy"
    variant: str | None = None
    weights: str | None = None
    freeze_backbone: bool = False
    freeze_decoder: bool = False
    num_queries: int = Field(100, gt=0)
    dim: int = Field(64, gt=0)
    num_classes: int = Field(31, gt=0)
    num_heads: int = Field(4, gt=0)
    num_decoder_layers: int = Field(2, gt=0)

    @model_validator(mode="after")
    def _check(self) -> DetectorConfig:
        if self.dim % self.num_heads:
            raise ValueError(f"dim={self.dim} не делится на num_heads={self.num_heads}")
        if self.name == "rfdetr" and not self.variant:
            raise ValueError("для rfdetr нужен variant")
        return self


class ContextConfig(_Section):
    name: ContextName = "none"
    fusion: FusionMode = "residual"
    num_slots: int = Field(64, ge=0)
    memory_length: int = Field(24, gt=0)
    short_memory_length: int = Field(3, gt=0)
    write_threshold: float = Field(0.5, ge=0.0, le=1.0)
    max_missed: int = Field(24, ge=0)
    association_iou_threshold: float = Field(0.1, ge=0.0, le=1.0)
    association_cosine_threshold: float = Field(0.5, ge=-1.0, le=1.0)
    association_appearance_weight: float = Field(0.25, ge=0.0)
    motion_momentum: float = Field(0.8, ge=0.0, lt=1.0)
    write_gate: bool = False
    motion: str | None = None
    horizon: int = Field(0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> ContextConfig:
        if self.name == "none":
            return self
        if not self.num_slots and self.name != "cross_attn":
            raise ValueError(f"ветке {self.name!r} нужен num_slots > 0")
        if self.name == "memot" and self.short_memory_length > self.memory_length:
            raise ValueError("memot требует short_memory_length <= memory_length")
        if self.name == "bridge_ad" and not self.horizon:
            raise ValueError("bridge_ad адресуется по горизонту — нужен horizon > 0")
        return self


class TrainConfig(_Section):
    epochs: int = Field(12, gt=0)
    lr: float = Field(1e-4, gt=0)
    batch_size: int = Field(2, gt=0)
    grad_accum: int = Field(1, ge=1)
    amp: bool = True
    denoising: bool = True
    #: False = полный BPTT по клипу. Не запрещаем — иногда нужно для проверки
    #: гипотезы, — но по умолчанию это прямой путь в OOM.
    detach_state: bool = True


class ExperimentConfig(_Section):
    name: str = "unnamed"
    data: DataConfig = Field(default_factory=DataConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)

    @model_validator(mode="after")
    def _check(self) -> ExperimentConfig:
        name = self.context.name
        if name in NEEDS_CONTEXT_FRAMES and not self.data.context_k:
            raise ValueError(f"{name} читает пиксели контекстных кадров, а context_k=0")
        # clip_len нужен только тем, кто переносит состояние с кадра на кадр.
        # Feature-level ветки читают K кадров из ContextBatch и на одиночном
        # кадре осмысленны — запрещать им clip_len=1 значит блокировать
        # легитимный ablation-контроль сообщением, которое к ним не относится.
        recurrent = name != "none" and name not in NEEDS_CONTEXT_FRAMES
        if recurrent and self.data.clip_len < 2:
            raise ValueError(
                f"ветка {name!r} переносит состояние между кадрами, "
                "а clip_len=1: память никогда не будет прочитана"
            )
        return self


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def load_config(
    path: str | Path, overrides: list[str] | None = None
) -> ExperimentConfig:
    """Загрузить конфиг с раскрытием "_base_" и override'ами "a.b=value".

    Override'ы нужны Человеку 5: сетка по длине истории — это один конфиг и
    десяток запусков с `data.context_k=...`, а не десять почти одинаковых
    файлов, которые разъезжаются на второй неделе.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("_comment", None)

    seen = {path.resolve()}
    while "_base_" in raw:
        base_path = (path.parent / raw.pop("_base_")).resolve()
        if base_path in seen:
            raise ValueError(f"цикл в _base_: {base_path}")
        seen.add(base_path)
        base = json.loads(base_path.read_text(encoding="utf-8"))
        base.pop("_comment", None)
        raw = _merge(base, raw)
        path = base_path

    for item in overrides or []:
        key, _, value = item.partition("=")
        if not _:
            raise ValueError(f"override должен быть вида a.b=value, получено {item!r}")
        node = raw
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = _coerce(value)

    return ExperimentConfig.model_validate(raw)
