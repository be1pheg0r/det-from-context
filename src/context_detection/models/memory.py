"""Context и memory-модули. Интерфейс — Человек 1, реализации — Человек 4.

Все ветки за одним интерфейсом, иначе Человек 5 не сможет их сравнить.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn

from ..contracts import ContextBatch, ContextOutput, DetectorOutput, MemoryState


class ContextModule(nn.Module, ABC):
    """Единый интерфейс для всех веток контекста и памяти.

    Разделение read/write — не косметика. Гейт записи (TSA) существует только
    если у write есть собственный шаг и он может решить НЕ писать. Слить их в
    один forward — значит закрыть себе ветку lifecycle, то есть ровно то, что
    литобзор называет незакрытым участком.

    `forward` намеренно не определён: у модуля два разных вызова на разных
    этапах шага, и один forward здесь только запутывает.
    """

    #: Нужны ли модулю пиксели контекстных кадров. False → ContextDetector не
    #: гоняет backbone по K кадрам (экономия, ради которой память и затевалась).
    needs_context_frames: bool = False

    @abstractmethod
    def read(
        self,
        queries: Tensor,
        state: MemoryState | None,
        context: ContextBatch,
        encoded_context: list[Tensor] | None = None,
    ) -> ContextOutput:
        """Прочитать контекст ключом `queries` [B, N, D].

        Обязан учитывать context.valid_mask и context.time_offsets.
        Компенсация движения слота на dt — здесь (грабля №4: без неё
        пропагация слепая).
        """

    @abstractmethod
    def write(
        self,
        state: MemoryState | None,
        output: DetectorOutput,
        context: ContextBatch,
    ) -> MemoryState | None:
        """Обновить память по результатам текущего кадра.

        Грабля №3 (TSA): безусловная запись каждый кадр даёт state drift.
        Все detection-работы пишут безусловно — это и есть незанятая ниша.
        """

    def reset(self, state: MemoryState | None, mask: Tensor) -> MemoryState | None:
        """Сброс по маске [B] начала последовательности.

        Реализация одна на всех и живёт в MemoryState.reset — переопределять
        нужно только если у ветки есть состояние вне MemoryState.
        """
        return None if state is None else state.reset(mask)

    def init_state(
        self, batch_size: int, dim: int, device, dtype
    ) -> MemoryState | None:
        """Начальное состояние. None — ветка без явной памяти."""
        return None


class NoContext(ContextModule):
    """Baseline: контекста нет. Человек 1.

    Не «пустышка на будущее», а точка отсчёта: все ΔAP считаются относительно
    неё, и она обязана быть бит в бит равна детектору без обёртки. Поэтому
    read возвращает query_delta=None, а не нулевой тензор — сложение с нулём
    формально безобидно, но добавляет операцию в граф и ломает побитовое
    совпадение с оригиналом.
    """

    def read(
        self,
        queries: Tensor,
        state: MemoryState | None,
        context: ContextBatch,
        encoded_context: list[Tensor] | None = None,
    ) -> ContextOutput:
        return ContextOutput(query_delta=None, memory_state=None)

    def write(
        self,
        state: MemoryState | None,
        output: DetectorOutput,
        context: ContextBatch,
    ) -> MemoryState | None:
        return None


class EMASlot(ContextModule):
    """MeMOTR-стиль: экспоненциально сглаженное состояние объекта + attention.

    TODO(чел.4). Первая реальная ветка — самая простая рабочая память,
    с неё начинать. needs_context_frames = False: читает из MemoryState.
    """


class ContextCrossAttention(ContextModule):
    """Queries читают K закодированных контекстных кадров.

    TODO(чел.4). Формально feature-level ветка (TransVOD/FAQ), стоимость
    растёт с K. Нужна как ablation-контроль «query-level vs feature-level»,
    а не как основная линия. needs_context_frames = True.
    """

    needs_context_frames = True


class StreamQueue(ContextModule):
    """StreamPETR: FIFO-очередь top-K queries + motion-aware выравнивание.

    TODO(чел.4). Референс-архитектура. Слот хранит feature + бокс + timestamp
    + confidence; чтение — cross-attention текущих queries к очереди.
    Компенсация движения: начать с constant velocity, библиотека
    кинематических гипотез (HAT) — второй заход.
    """


class BridgeADMemory(ContextModule):
    """Память хранит прошлые предсказания О БУДУЩЕМ, retrieval по целевому
    моменту времени, а не по свежести.

    TODO(чел.4). На кадре t поднимается то, что кадр t−3 предсказывал именно
    для момента t. Принципиально другая схема адресации, чем у StreamQueue.
    """


# TODO(чел.4): fusion уже реализован в models/wrapper.py как параметр —
# отдельную иерархию классов под три режима не заводить.
