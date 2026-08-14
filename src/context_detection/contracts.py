"""Общие контракты между компонентами.

Владелец: Человек 1. Меняется только по согласованию — остальные модули
импортируют ТОЛЬКО отсюда, никогда не импортируют реализации друг друга.

Импортируется без torch (аннотации отложенные), чтобы контракты можно было
читать и тестировать в окружении без установленного torch.

Соглашения:
  * B — батч, K — число контекстных слотов, N — число object queries,
    S — число слотов памяти, C/H/W — канал/высота/ширина.
  * Боксы внутри модели: cxcywh, нормализованные в [0, 1] относительно
    входного изображения. В COCO-формат конвертируем только на границе
    (dataset in / evaluation out).
  * time_offset — секунды в прошлое, положительные. 0 = текущий кадр.
  * Все тензоры одного батча лежат на одном device; dtype плавающих — float32,
    AMP включается только внутри autocast-региона в engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class DetectionBatch:
    """Целевые кадры — то, что детектор должен разметить."""

    images: Tensor  # [B, C, H, W]
    # len B; ключи: "boxes" [n, 4] cxcywh, "labels" [n]
    targets: list[dict[str, Tensor]]
    sequence_id: list[str]  # len B; граница сцены — смена значения
    frame_id: Tensor  # [B] int64, порядковый номер кадра внутри последовательности
    timestamp: Tensor  # [B] float32, секунды
    #: True, если для этого элемента батча память должна быть сброшена
    #: (начало последовательности или разрыв). Читает MemoryModule.reset().
    is_sequence_start: Tensor | None = None  # [B] bool


@dataclass
class ContextBatch:
    """Контекстные кадры. Для чисто рекуррентных веток images может быть None
    — там контекст берётся из MemoryState, а не из пикселей."""

    images: Tensor | None  # [B, K, C, H, W]
    valid_mask: Tensor  # [B, K] bool; False = слот-заглушка (начало последовательности)
    time_offsets: Tensor  # [B, K] float32, секунды в прошлое, > 0
    #: GT контекстных кадров, если нужен (teacher-forcing памяти, denoising).
    targets: list[list[dict[str, Tensor]]] | None = None
    #: Всё, что специфично для датасета/эксперимента (ego-pose, id сцены и т.п.).
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectorOutput:
    """Выход детектора, приведённый к общему виду.

    Поля после `boxes` — точки подключения контекста. Детектор обязан их
    заполнять, даже когда контекст выключен: на них завязаны memory-ветки.
    """

    logits: Tensor  # [B, N, num_classes]
    boxes: Tensor  # [B, N, 4] cxcywh нормализованные
    queries: Tensor  # [B, N, D] content-эмбеддинги object queries последнего слоя
    reference_points: Tensor  # [B, N, 4] anchor-боксы queries (DAB-DETR формат)
    #: Multi-scale признаки после projector'а, от крупного к мелкому.
    features: list[Tensor] = field(default_factory=list)  # каждый [B, C_i, H_i, W_i]
    #: Промежуточные слои decoder'а для aux-loss и для подключения памяти
    #: между слоями. Каждый элемент — {"queries", "boxes", "logits"}.
    decoder_layers: list[dict[str, Tensor]] = field(default_factory=list)
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextOutput:
    """Что context/memory-модуль отдаёт детектору.

    `query_delta` складывается с queries детектора выбранным fusion-режимом
    (residual / gated / concat+proj — параметр модуля, не отдельный класс).
    """

    query_delta: Tensor | None  # [B, N, D]; None = контекста нет (NoContext)
    # обновлённое состояние для следующего кадра
    memory_state: MemoryState | None = None
    #: Диагностика для логов и визуализации: число активных слотов, средний
    #: возраст, read weights [B, N, S], write rate, число вытесненных слотов.
    #: Скаляры пишутся в лог, тензоры — в визуализацию.
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryState:
    """Явное состояние памяти. Один слот = одна объектная гипотеза.

    Состав слота задан литобзором (StreamPETR / TSA / DAB-DETR): семантика +
    геометрический anchor + время + оценка качества + возраст + наблюдаемость.
    Без timestamp и motion перенос бокса во времени невозможен — пропагация
    получается «слепой», см. StreamPETR MLN и HAT.
    """

    feature: Tensor  # [B, S, D] content-эмбеддинг слота
    box: Tensor  # [B, S, 4] cxcywh, последнее известное положение
    timestamp: Tensor  # [B, S] float32, время последнего наблюдения
    confidence: Tensor  # [B, S] float32, качество/уверенность слота
    age: Tensor  # [B, S] int64, кадров с момента создания
    valid: Tensor  # [B, S] bool, слот занят
    #: Наблюдаемость на последнем кадре. Отличается от valid: слот жив, но
    #: объект не виден → писать в него нельзя (TSA: state drift).
    observed: Tensor | None = None  # [B, S] bool
    #: Параметры движения для выравнивания бокса на dt. Форма зависит от
    #: модели движения (CV: [B, S, 4]; библиотека гипотез HAT: [B, S, H, 4]).
    motion: Tensor | None = None

    def detach(self) -> MemoryState:
        """Обрыв графа между кадрами — иначе BPTT растёт на всю последовательность."""
        raise NotImplementedError("Человек 4")

    def reset(self, mask: Tensor) -> MemoryState:
        """Обнулить элементы батча, где mask=True (начало последовательности)."""
        raise NotImplementedError("Человек 4")
