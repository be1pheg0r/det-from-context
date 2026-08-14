"""Склейка детектора и context-модуля. Человек 1.

Единственное место, где они встречаются. Ни детектор, ни память друг о друге
не знают — обе стороны видят только контракты.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..contracts import ContextBatch, DetectionBatch, DetectorOutput, MemoryState
from ..registry import FUSION_MODES, FusionMode
from .detector import DetectorAdapter
from .memory import ContextModule


class Fusion(nn.Module):
    """Как контекст вливается в queries. Параметр, а не иерархия классов:
    на режим приходится по одной строке.

    residual        — q + d. Ничего не learnable, самый честный baseline
    gated_residual  — q + sigmoid(W[q, d]) * d. Модель сама решает, доверять
                      ли памяти на этом кадре; ближе всего к propagation gate
                      из QueryProp и к активации слота из TSA
    concat_proj     — W[q, d]. Самый выразительный и самый склонный забыть q
    """

    def __init__(self, mode: FusionMode, dim: int) -> None:
        super().__init__()
        if mode not in FUSION_MODES:
            raise ValueError(
                f"неизвестный fusion {mode!r}, есть: {sorted(FUSION_MODES)}"
            )
        self.mode = mode
        if mode == "gated_residual":
            self.gate = nn.Linear(2 * dim, dim)
            # Стартуем закрытыми: sigmoid(-2) ≈ 0.12. Модель начинает почти как
            # baseline и открывает гейт, только если память реально помогает.
            # Иначе первые эпохи память подмешивает шум в ещё не обученные
            # queries и обучение уходит в минус — обычная причина «память не
            # работает», которая на самом деле про инициализацию.
            nn.init.constant_(self.gate.bias, -2.0)
        elif mode == "concat_proj":
            self.proj = nn.Linear(2 * dim, dim)

    def forward(self, queries: Tensor, delta: Tensor | None) -> Tensor:
        if delta is None:  # NoContext — граф не трогаем вовсе
            return queries
        if self.mode == "residual":
            return queries + delta
        joint = torch.cat([queries, delta], dim=-1)
        if self.mode == "gated_residual":
            return queries + torch.sigmoid(self.gate(joint)) * delta
        return self.proj(joint)


class ContextDetector(nn.Module):
    """Детектор + модуль контекста. Один шаг по времени.

    Порядок на кадре:
      1. reset памяти по batch.is_sequence_start
      2. encode_context_frames — только если ветка этого требует
      3. context.read  — ключ чтения: начальные queries детектора
      4. fusion
      5. detector.forward с уже слитыми queries
      6. context.write — обновление состояния
      7. state.detach() — обрыв BPTT между шагами клипа

    Шаг 7 обязателен: без него граф растёт на всю последовательность и клип
    длиннее трёх-четырёх кадров не влезает в память. Отключается только
    осознанно, через detach_state=False.
    """

    def __init__(
        self,
        detector: DetectorAdapter,
        context_module: ContextModule,
        fusion: FusionMode = "residual",
        detach_state: bool = True,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.context_module = context_module
        self.fusion = Fusion(fusion, detector.dim)
        self.detach_state = detach_state

    def forward(
        self,
        batch: DetectionBatch,
        context: ContextBatch,
        state: MemoryState | None = None,
    ) -> tuple[DetectorOutput, MemoryState | None]:
        if batch.batch_size != context.valid_mask.shape[0]:
            raise ValueError(
                f"B у батча ({batch.batch_size}) и контекста "
                f"({context.valid_mask.shape[0]}) не совпадают"
            )

        if state is not None and batch.is_sequence_start is not None:
            state = self.context_module.reset(state, batch.is_sequence_start)

        encoded = None
        if self.context_module.needs_context_frames:
            encoded = self.detector.encode_context_frames(context)

        queries = self.detector.initial_queries(batch)
        ctx = self.context_module.read(queries, state, context, encoded)
        output = self.detector(batch, query_init=self.fusion(queries, ctx.query_delta))

        state = self.context_module.write(
            ctx.memory_state if ctx.memory_state is not None else state,
            output,
            context,
        )
        if state is not None and self.detach_state:
            state = state.detach()
        return output, state

    @torch.no_grad()
    def rollout(
        self,
        steps: list[tuple[DetectionBatch, ContextBatch]],
        state: MemoryState | None = None,
    ) -> list[DetectorOutput]:
        """Последовательный прогон клипа без градиентов — для eval и профиля.

        Обучающий проход по клипу живёт в engine (Человек 5): там нужен один
        backward на весь клип, а не на кадр.
        """
        outputs = []
        for batch, context in steps:
            output, state = self(batch, context, state)
            outputs.append(output)
        return outputs
