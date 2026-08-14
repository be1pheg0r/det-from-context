"""Общие контракты между компонентами. Человек 1.

Меняется только по согласованию — остальные модули импортируют ТОЛЬКО отсюда
и никогда не импортируют реализации друг друга.

Контракты — Pydantic-модели с валидаторами, а не голые dataclass'ы: при работе
впятером несогласованная форма тензора обнаруживается через полчаса обучения,
а валидатор роняет её на границе модуля с внятным сообщением.

Соглашения:
  * B — батч, K — контекстные слоты, N — object queries, S — слоты памяти.
  * Боксы внутри модели: cxcywh, нормализованные в [0, 1]. COCO-формат живёт
    только на границах (загрузка датасета / выгрузка метрик).
  * time_offset — секунды в прошлое, строго > 0 для валидных слотов, 0 для
    заглушек. Текущий кадр не является собственным контекстом.
  * Все тензоры одного контракта — на одном device. Плавающие — float32;
    AMP включается только внутри autocast в engine, состояние памяти в fp32.
"""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

# ponytail: валидация выполняется на КАЖДОМ построении контракта, то есть на
# каждом кадре. Замеры показывают единицы микросекунд против миллисекунд
# forward'а, поэтому оставлено включённым. Если профиль покажет обратное —
# точка апгрейда одна: `Model.model_construct(...)` в engine на горячем пути,
# валидация останется в тестах и на границах датасета.


def _shape(name: str, t: Tensor, *dims: int | None) -> None:
    """dims: ожидаемая форма, None — «любое значение»."""
    if t.dim() != len(dims):
        raise ValueError(
            f"{name}: ожидалось {len(dims)} измерений, получено {t.dim()} "
            f"(форма {tuple(t.shape)})"
        )
    for i, want in enumerate(dims):
        if want is not None and t.shape[i] != want:
            raise ValueError(
                f"{name}: измерение {i} = {t.shape[i]}, ожидалось {want} "
                f"(форма {tuple(t.shape)})"
            )


def _same_device(**tensors: Tensor | None) -> None:
    seen = {n: t.device for n, t in tensors.items() if isinstance(t, Tensor)}
    if len(set(seen.values())) > 1:
        raise ValueError(f"тензоры на разных device: {seen}")


def _boxes_normalized(name: str, boxes: Tensor) -> None:
    """cxcywh в [0, 1]. Ловит два самых частых бага: забыли поделить на
    размер изображения и подсунули xyxy вместо cxcywh (тогда w/h вылезают
    за единицу или центр оказывается меньше половины ширины)."""
    if boxes.numel() == 0:
        return
    if not torch.isfinite(boxes).all():
        raise ValueError(f"{name}: NaN/Inf в боксах")
    # detach: боксы приходят из графа, а .item() на requires_grad-тензоре
    # предупреждает и без нужды тянет граф в питон.
    lo, hi = float(boxes.detach().min()), float(boxes.detach().max())
    if lo < -1e-4 or hi > 1.0 + 1e-4:
        raise ValueError(
            f"{name}: боксы вне [0, 1] (min={lo:.4f}, max={hi:.4f}). "
            "Ожидается cxcywh, нормализованный по размеру изображения"
        )


class _Contract(BaseModel):
    """Общая конфигурация: тензоры — произвольный тип для pydantic."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        # validate_assignment намеренно выключен: after-валидаторы проставляют
        # производные поля присваиванием, а с ним это уходит в рекурсию.
        # Контракты строятся, а не мутируются, — терять тут нечего.
        validate_assignment=False,
        extra="forbid",
    )


class DetectionBatch(_Contract):
    """Целевые кадры — то, что детектор должен разметить."""

    images: Tensor  # [B, C, H, W]
    #: len B; ключи: "boxes" [n, 4] cxcywh, "labels" [n] int64
    targets: list[dict[str, Tensor]]
    sequence_id: list[str]  # len B; граница сцены — смена значения
    frame_id: Tensor  # [B] int64
    timestamp: Tensor  # [B] float32, секунды
    #: [B] bool. True → память для этого элемента батча должна быть сброшена.
    #: Если не передан, вычисляется как «все False»: считаем, что клип
    #: продолжается. Это единственный безопасный дефолт — лишний сброс теряет
    #: контекст молча, а пропущенный ловится тестом на утечку между сценами.
    is_sequence_start: Tensor | None = None

    @property
    def batch_size(self) -> int:
        return self.images.shape[0]

    @model_validator(mode="after")
    def _check(self) -> DetectionBatch:
        _shape("images", self.images, None, None, None, None)
        b = self.images.shape[0]
        if not self.images.is_floating_point():
            raise ValueError("images должен быть плавающим типом")
        pairs = (("targets", self.targets), ("sequence_id", self.sequence_id))
        for name, value in pairs:
            if len(value) != b:
                raise ValueError(f"{name}: длина {len(value)}, ожидалось B={b}")
        _shape("frame_id", self.frame_id, b)
        _shape("timestamp", self.timestamp, b)
        if self.frame_id.dtype != torch.int64:
            raise ValueError("frame_id должен быть int64")

        for i, t in enumerate(self.targets):
            if "boxes" not in t or "labels" not in t:
                raise ValueError(f"targets[{i}]: нужны ключи 'boxes' и 'labels'")
            _shape(f"targets[{i}]['boxes']", t["boxes"], None, 4)
            _boxes_normalized(f"targets[{i}]['boxes']", t["boxes"])
            if t["labels"].shape[0] != t["boxes"].shape[0]:
                raise ValueError(
                    f"targets[{i}]: labels ({t['labels'].shape[0]}) != "
                    f"boxes ({t['boxes'].shape[0]})"
                )

        if self.is_sequence_start is None:
            self.is_sequence_start = torch.zeros(
                b, dtype=torch.bool, device=self.images.device
            )
        else:
            _shape("is_sequence_start", self.is_sequence_start, b)
            if self.is_sequence_start.dtype != torch.bool:
                raise ValueError("is_sequence_start должен быть bool")

        _same_device(
            images=self.images, frame_id=self.frame_id, timestamp=self.timestamp
        )
        return self


class ContextBatch(_Contract):
    """Контекстные кадры.

    images=None — легальное состояние: чисто рекуррентные ветки берут контекст
    из MemoryState, а не из пикселей, и гонять для них backbone по K кадрам
    незачем. valid_mask и time_offsets обязательны всегда — они описывают
    структуру истории независимо от того, загружены ли кадры.
    """

    images: Tensor | None = None  # [B, K, C, H, W]
    valid_mask: Tensor  # [B, K] bool; False = слот-заглушка
    time_offsets: Tensor  # [B, K] float, секунды в прошлое
    targets: list[list[dict[str, Tensor]]] | None = None
    extras: dict[str, Any] = Field(default_factory=dict)

    @property
    def num_slots(self) -> int:
        return self.valid_mask.shape[1]

    @model_validator(mode="after")
    def _check(self) -> ContextBatch:
        _shape("valid_mask", self.valid_mask, None, None)
        b, k = self.valid_mask.shape
        if self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask должен быть bool")
        _shape("time_offsets", self.time_offsets, b, k)
        if self.images is not None:
            _shape("images", self.images, b, k, None, None, None)

        # Заглушки не должны нести время: иначе модуль памяти незаметно
        # выучит смещение по «пустым» слотам.
        if self.time_offsets[~self.valid_mask].abs().gt(1e-6).any():
            raise ValueError("time_offsets != 0 в невалидных слотах")
        valid_offsets = self.time_offsets[self.valid_mask]
        if valid_offsets.numel() and valid_offsets.le(0).any():
            raise ValueError(
                "time_offsets должны быть > 0 в валидных слотах: контекст — это "
                "прошлое. Значение <= 0 означает утечку будущего или текущего кадра"
            )
        if self.targets is not None and len(self.targets) != b:
            raise ValueError(f"targets: длина {len(self.targets)}, ожидалось B={b}")

        _same_device(
            images=self.images,
            valid_mask=self.valid_mask,
            time_offsets=self.time_offsets,
        )
        return self


class DetectorOutput(_Contract):
    """Выход детектора в общем виде.

    Поля после boxes — точки подключения контекста. Детектор обязан заполнять
    их даже когда контекст выключен: на них завязаны все memory-ветки, и
    ветка, обнаружившая на них None, узнает об этом слишком поздно.
    """

    logits: Tensor  # [B, N, num_classes]
    boxes: Tensor  # [B, N, 4] cxcywh в [0, 1]
    queries: Tensor  # [B, N, D]
    reference_points: Tensor  # [B, N, 4] anchor-боксы (DAB-DETR)
    features: list[Tensor] = Field(default_factory=list)  # multi-scale, [B, C_i, H, W]
    #: Слои decoder'а для aux-loss и подключения памяти между слоями.
    decoder_layers: list[dict[str, Tensor]] = Field(default_factory=list)
    aux: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> DetectorOutput:
        _shape("logits", self.logits, None, None, None)
        b, n = self.logits.shape[:2]
        _shape("boxes", self.boxes, b, n, 4)
        _shape("queries", self.queries, b, n, None)
        _shape("reference_points", self.reference_points, b, n, 4)
        _boxes_normalized("boxes", self.boxes)
        _boxes_normalized("reference_points", self.reference_points)
        for i, layer in enumerate(self.decoder_layers):
            if "queries" not in layer:
                raise ValueError(f"decoder_layers[{i}]: нужен ключ 'queries'")
        _same_device(logits=self.logits, boxes=self.boxes, queries=self.queries)
        return self


class MemoryState(_Contract):
    """Состояние памяти. Один слот = одна объектная гипотеза.

    Состав слота задан литобзором (StreamPETR / TSA / DAB-DETR): семантика +
    геометрический anchor + время + качество + возраст + наблюдаемость. Без
    timestamp и motion перенос бокса во времени невозможен, и пропагация
    получается «слепой» (StreamPETR MLN, HAT).
    """

    feature: Tensor  # [B, S, D]
    box: Tensor  # [B, S, 4] cxcywh, последнее известное положение
    timestamp: Tensor  # [B, S] float, время последнего наблюдения
    confidence: Tensor  # [B, S] float в [0, 1]
    age: Tensor  # [B, S] int64, кадров с момента создания
    valid: Tensor  # [B, S] bool, слот занят
    #: Наблюдаемость на последнем кадре. Отличается от valid: слот жив, но
    #: объект не виден → писать в него нельзя (TSA: state drift).
    observed: Tensor | None = None  # [B, S] bool
    #: Параметры движения для выравнивания бокса на dt. Форма зависит от модели
    #: движения: constant velocity [B, S, 4], библиотека гипотез HAT [B, S, H, 4].
    motion: Tensor | None = None

    @model_validator(mode="after")
    def _check(self) -> MemoryState:
        _shape("feature", self.feature, None, None, None)
        b, s = self.feature.shape[:2]
        _shape("box", self.box, b, s, 4)
        for name in ("timestamp", "confidence", "age", "valid"):
            _shape(name, getattr(self, name), b, s)
        if self.valid.dtype != torch.bool:
            raise ValueError("valid должен быть bool")
        if self.age.dtype != torch.int64:
            raise ValueError("age должен быть int64")
        if self.age.lt(0).any():
            raise ValueError("age не может быть отрицательным")
        # Боксы проверяем только в занятых слотах: свободные содержат мусор.
        _boxes_normalized("box", self.box[self.valid])
        conf = self.confidence[self.valid]
        if conf.numel() and (conf.lt(-1e-4).any() or conf.gt(1.0 + 1e-4).any()):
            raise ValueError("confidence вне [0, 1] — ожидается вероятность, не логит")
        for name in ("observed", "motion"):
            t = getattr(self, name)
            if t is not None and t.shape[:2] != (b, s):
                raise ValueError(f"{name}: первые два измерения != (B={b}, S={s})")
        return self

    @property
    def batch_size(self) -> int:
        return self.feature.shape[0]

    @property
    def num_slots(self) -> int:
        return self.feature.shape[1]

    @classmethod
    def empty(
        cls,
        batch_size: int,
        num_slots: int,
        dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> MemoryState:
        """Пустая память. Боксы — вырожденный центр кадра, но valid=False,
        поэтому наружу они не попадают."""
        z = torch.zeros(batch_size, num_slots, device=device, dtype=dtype)
        return cls(
            feature=torch.zeros(batch_size, num_slots, dim, device=device, dtype=dtype),
            box=torch.full((batch_size, num_slots, 4), 0.5, device=device, dtype=dtype),
            timestamp=z.clone(),
            confidence=z.clone(),
            age=torch.zeros(batch_size, num_slots, device=device, dtype=torch.int64),
            valid=torch.zeros(batch_size, num_slots, device=device, dtype=torch.bool),
            observed=torch.zeros(
                batch_size, num_slots, device=device, dtype=torch.bool
            ),
        )

    def detach(self) -> MemoryState:
        """Обрыв графа между кадрами.

        Без этого BPTT растёт на всю последовательность и клип длиннее
        трёх-четырёх кадров не влезает в память. Вызывается в ContextDetector
        после каждого шага.
        """

        def d(t: Tensor | None) -> Tensor | None:
            return None if t is None else t.detach()

        return MemoryState.model_construct(
            feature=self.feature.detach(),
            box=self.box.detach(),
            timestamp=self.timestamp.detach(),
            confidence=self.confidence.detach(),
            age=self.age,
            valid=self.valid,
            observed=d(self.observed),
            motion=d(self.motion),
        )

    def reset(self, mask: Tensor) -> MemoryState:
        """Обнулить элементы батча, где mask=True (начало последовательности).

        mask: [B] bool. Возвращает новое состояние — на месте не мутирует,
        иначе прошлый шаг графа получит изменённый тензор.
        """
        _shape("reset mask", mask, self.batch_size)
        if mask.dtype != torch.bool:
            raise ValueError("mask должен быть bool")
        if not bool(mask.any()):
            return self
        fresh = MemoryState.empty(
            self.batch_size,
            self.num_slots,
            self.feature.shape[-1],
            device=self.feature.device,
            dtype=self.feature.dtype,
        )

        def clear(old: Tensor | None) -> Tensor | None:
            """Обнулить строки батча по маске. Маска расширяется под любое
            число хвостовых измерений — motion бывает и [B,S,4], и [B,S,H,4]."""
            if old is None:
                return None
            m = mask.view(-1, *([1] * (old.dim() - 1)))
            return torch.where(m, torch.zeros_like(old), old)

        m2 = mask.view(-1, 1, 1)
        return MemoryState.model_construct(
            feature=clear(self.feature),
            box=torch.where(m2, fresh.box, self.box),
            timestamp=clear(self.timestamp),
            confidence=clear(self.confidence),
            age=clear(self.age),
            valid=clear(self.valid),
            observed=clear(self.observed),
            motion=clear(self.motion),
        )


class ContextOutput(_Contract):
    """Что context/memory-модуль отдаёт детектору."""

    #: [B, N, D]; None = контекста нет (NoContext). Складывается с queries
    #: детектора выбранным fusion-режимом.
    query_delta: Tensor | None = None
    memory_state: MemoryState | None = None
    #: Диагностика для логов и визуализации: active_slots, mean_age,
    #: read_weights [B, N, S], write_rate, evicted. Скаляры уходят в лог,
    #: тензоры — в визуализацию Человека 5. Без неё отладка памяти
    #: превращается в гадание по итоговому AP.
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> ContextOutput:
        if self.query_delta is not None:
            _shape("query_delta", self.query_delta, None, None, None)
            if not torch.isfinite(self.query_delta).all():
                raise ValueError("query_delta содержит NaN/Inf")
        return self
