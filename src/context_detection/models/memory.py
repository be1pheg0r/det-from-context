"""Context и memory-модули. Человек 4 (протокол согласует Человек 1).

Все ветки — за одним интерфейсом, иначе Человек 5 не сможет их сравнить.
"""

from __future__ import annotations

from typing import Protocol

from ..contracts import ContextBatch, ContextOutput, DetectorOutput, MemoryState


class ContextModule(Protocol):
    """Единый интерфейс для всех веток контекста и памяти.

    Разделение read/write — не косметика: гейт записи (TSA) существует только
    если write имеет собственный шаг и может решить НЕ писать. Слить их в один
    forward — значит закрыть себе ветку lifecycle.
    """

    def read(
        self,
        queries,  # [B, N, D] object queries текущего кадра
        state: MemoryState | None,
        context: ContextBatch,
        encoded_context: list | None = None,
    ) -> ContextOutput:
        """Прочитать контекст. Обязан учитывать context.valid_mask и
        context.time_offsets. Компенсация движения слота на dt — здесь."""
        ...

    def write(
        self,
        state: MemoryState | None,
        output: DetectorOutput,
        context: ContextBatch,
    ) -> MemoryState:
        """Обновить память по результатам текущего кадра.

        ⚠️ Грабля №3 (TSA): безусловная запись каждый кадр даёт state drift.
        Все detection-работы пишут безусловно — это и есть незанятая ниша.
        """
        ...

    def reset(self, state: MemoryState | None, mask) -> MemoryState | None:
        """Сброс по маске [B] начала последовательности. Без этого память
        течёт между сценами — обязательный тест Человека 4."""
        ...


class NoContext:
    """Baseline: контекста нет. Человек 4.

    TODO: read возвращает ContextOutput(query_delta=None), write — no-op.
    Должен воспроизводить regression baseline Человека 3 бит в бит.
    """


class EMASlot:
    """MeMOTR-стиль: экспоненциально сглаженное состояние объекта + attention.

    TODO(чел.4). Первая реальная ветка — самая простая рабочая память.
    Начинать с неё, а не с cross-attention или очереди.
    """


class ContextCrossAttention:
    """Queries читают K закодированных контекстных кадров.

    TODO(чел.4). Формально feature-level ветка (TransVOD/FAQ), стоимость
    растёт с K. Нужна как ablation-контроль «query-level vs feature-level»,
    а не как основная линия.
    """


class StreamQueue:
    """StreamPETR: FIFO-очередь top-K queries прошлых кадров + motion-aware
    выравнивание перед чтением.

    TODO(чел.4). Референс-архитектура памяти. Слот хранит feature + бокс +
    timestamp + confidence; чтение — cross-attention текущих queries к очереди.

    ⚠️ Грабля №4: без переноса бокса на dt пропагация «слепая». Начать с
    constant velocity, библиотека кинематических гипотез (HAT) — второй заход.
    """


class BridgeADMemory:
    """В памяти лежат прошлые предсказания О БУДУЩЕМ, retrieval по целевому
    моменту времени, а не по свежести.

    TODO(чел.4). На кадре t поднимается то, что кадр t−3 предсказывал именно
    для момента t. Принципиально другая схема адресации, чем у StreamQueue.
    Самое буквальное прочтение задачи «переиспользовать прошлые предсказания».
    """


# TODO(чел.4): fusion — параметр модуля, НЕ отдельная иерархия классов:
#   residual | gated_residual | concat_proj
# Три строки кода на режим, отдельный пакет под это не нужен.

# TODO(чел.4): диагностика в ContextOutput.diagnostics —
#   active_slots, mean_age, read_weights [B, N, S], write_rate, evicted.
# Человек 5 строит на этом визуализацию; без диагностики отладка памяти
# превращается в гадание по итоговому AP.
