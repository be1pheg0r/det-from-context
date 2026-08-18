"""Context и memory-модули. Интерфейс — Человек 1, реализации — Человек 4.

Все ветки за одним интерфейсом, иначе Человек 5 не сможет их сравнить.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import torch
from pydantic import model_validator
from torch import Tensor, nn
from torch.nn import functional as F

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
        query_reference_points: Tensor | None = None,
        current_timestamp: Tensor | None = None,
    ) -> ContextOutput:
        """Прочитать контекст ключом `queries` [B, N, D].

        Обязан учитывать context.valid_mask и context.time_offsets.
        Компенсация движения слота на dt — здесь (грабля №4: без неё
        пропагация слепая).

        Args:
            query_reference_points: Encoder/decoder anchors ``[B, N, 4]``.
            current_timestamp: Время текущего DetectionBatch в секундах.
        """

    @abstractmethod
    def write(
        self,
        state: MemoryState | None,
        output: DetectorOutput,
        context: ContextBatch,
        current_timestamp: Tensor | None = None,
    ) -> MemoryState | None:
        """Обновить память по результатам текущего кадра.

        Грабля №3 (TSA): безусловная запись каждый кадр даёт state drift.
        Все detection-работы пишут безусловно — это и есть незанятая ниша.

        Args:
            current_timestamp: Время текущего DetectionBatch в секундах.
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
        query_reference_points: Tensor | None = None,
        current_timestamp: Tensor | None = None,
    ) -> ContextOutput:
        del query_reference_points
        return ContextOutput(query_delta=None, memory_state=None)

    def write(
        self,
        state: MemoryState | None,
        output: DetectorOutput,
        context: ContextBatch,
        current_timestamp: Tensor | None = None,
    ) -> MemoryState | None:
        return None


class MeMOTState(MemoryState):
    """Расширение общего состояния временным буфером MeMOT.

    Базовые поля сохраняют последний наблюдённый state каждого object slot,
    поэтому состояние остаётся совместимым с общим протоколом. Дополнительные
    поля реализуют измерение времени ``T`` из памяти ``[B, S, T, D]`` статьи.
    """

    history_feature: Tensor  # [B, S, T, D]
    history_valid: Tensor  # [B, S, T] bool
    history_timestamp: Tensor  # [B, S, T] float32
    dmat: Tensor  # [B, S, D]
    missed: Tensor  # [B, S] int64, последовательных пропущенных кадров
    clock: Tensor  # [B] float32, время последнего обработанного кадра
    evicted: Tensor  # [B] int64, вытеснений на последней записи

    @model_validator(mode="after")
    def _check_memot(self) -> Self:
        """Проверить временные поля MeMOT относительно базовых слотов."""
        batch_size: int = self.batch_size
        num_slots: int = self.num_slots
        dim: int = self.feature.shape[-1]
        expected_history: tuple[int, int] = (batch_size, num_slots)
        if self.history_feature.dim() != 4:
            raise ValueError("history_feature должен иметь форму [B, S, T, D]")
        if self.history_feature.shape[:2] != expected_history:
            raise ValueError("history_feature: первые измерения не совпадают с B, S")
        if self.history_feature.shape[-1] != dim:
            raise ValueError("history_feature: D не совпадает с feature")
        if self.history_valid.shape != self.history_feature.shape[:3]:
            raise ValueError("history_valid должен иметь форму [B, S, T]")
        if self.history_valid.dtype != torch.bool:
            raise ValueError("history_valid должен быть bool")
        if self.history_timestamp.shape != self.history_feature.shape[:3]:
            raise ValueError("history_timestamp должен иметь форму [B, S, T]")
        if self.dmat.shape != (batch_size, num_slots, dim):
            raise ValueError("dmat должен иметь форму [B, S, D]")
        if self.missed.shape != (batch_size, num_slots):
            raise ValueError("missed должен иметь форму [B, S]")
        if self.missed.dtype != torch.int64 or self.missed.lt(0).any():
            raise ValueError("missed должен быть неотрицательным int64")
        if self.clock.shape != (batch_size,):
            raise ValueError("clock должен иметь форму [B]")
        if self.evicted.shape != (batch_size,):
            raise ValueError("evicted должен иметь форму [B]")
        if self.evicted.dtype != torch.int64 or self.evicted.lt(0).any():
            raise ValueError("evicted должен быть неотрицательным int64")
        devices: set[torch.device] = {
            self.feature.device,
            self.history_feature.device,
            self.history_valid.device,
            self.history_timestamp.device,
            self.dmat.device,
            self.missed.device,
            self.clock.device,
            self.evicted.device,
        }
        if len(devices) != 1:
            raise ValueError("поля MeMOTState находятся на разных device")
        return self

    @classmethod
    def create(
        cls,
        batch_size: int,
        num_slots: int,
        dim: int,
        memory_length: int,
        device: torch.device | str,
    ) -> MeMOTState:
        """Создать пустой буфер MeMOT в обязательном для памяти float32."""
        base: MemoryState = MemoryState.empty(
            batch_size=batch_size,
            num_slots=num_slots,
            dim=dim,
            device=device,
            dtype=torch.float32,
        )
        return cls(
            **base.model_dump(),
            history_feature=torch.zeros(
                batch_size,
                num_slots,
                memory_length,
                dim,
                device=device,
                dtype=torch.float32,
            ),
            history_valid=torch.zeros(
                batch_size,
                num_slots,
                memory_length,
                device=device,
                dtype=torch.bool,
            ),
            history_timestamp=torch.zeros(
                batch_size,
                num_slots,
                memory_length,
                device=device,
                dtype=torch.float32,
            ),
            dmat=torch.zeros(
                batch_size,
                num_slots,
                dim,
                device=device,
                dtype=torch.float32,
            ),
            missed=torch.zeros(batch_size, num_slots, device=device, dtype=torch.int64),
            clock=torch.zeros(batch_size, device=device, dtype=torch.float32),
            evicted=torch.zeros(batch_size, device=device, dtype=torch.int64),
        )

    def detach(self) -> MeMOTState:
        """Оборвать BPTT и для общих полей, и для временной памяти."""
        base: MemoryState = super().detach()
        return MeMOTState.model_construct(
            **base.__dict__,
            history_feature=self.history_feature.detach(),
            history_valid=self.history_valid,
            history_timestamp=self.history_timestamp.detach(),
            dmat=self.dmat.detach(),
            missed=self.missed,
            clock=self.clock.detach(),
            evicted=self.evicted,
        )

    def reset(self, mask: Tensor) -> MeMOTState:
        """Сбросить выбранные последовательности вместе с историей и DMAT."""
        base: MemoryState = super().reset(mask)
        if base is self:
            return self
        expanded_history_mask: Tensor = mask.view(-1, 1, 1, 1)
        expanded_valid_mask: Tensor = mask.view(-1, 1, 1)
        expanded_slot_mask: Tensor = mask.view(-1, 1)
        return MeMOTState.model_construct(
            **base.__dict__,
            history_feature=torch.where(
                expanded_history_mask,
                torch.zeros_like(self.history_feature),
                self.history_feature,
            ),
            history_valid=torch.where(
                expanded_valid_mask,
                torch.zeros_like(self.history_valid),
                self.history_valid,
            ),
            history_timestamp=torch.where(
                expanded_valid_mask,
                torch.zeros_like(self.history_timestamp),
                self.history_timestamp,
            ),
            dmat=torch.where(
                expanded_valid_mask,
                torch.zeros_like(self.dmat),
                self.dmat,
            ),
            missed=torch.where(
                expanded_slot_mask,
                torch.zeros_like(self.missed),
                self.missed,
            ),
            clock=torch.where(mask, torch.zeros_like(self.clock), self.clock),
            evicted=torch.where(mask, torch.zeros_like(self.evicted), self.evicted),
        )


class MeMOTMemory(ContextModule):
    """MeMOT-inspired memory encoder with explicit internal track slots.

    A slot is no longer tied to a DETR query index. Predictions are associated
    one-to-one with existing slots using motion-compensated boxes and identity
    embeddings; unmatched confident predictions create or replace slots.
    This implements the memory-encoding half of MeMOT. A faithful Memory
    Decoder still requires separate proposal/track queries in DetectorAdapter.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_slots: int,
        memory_length: int,
        short_memory_length: int,
        write_threshold: float,
        max_missed: int,
        association_iou_threshold: float,
        association_cosine_threshold: float,
        association_appearance_weight: float,
        motion_momentum: float,
    ) -> None:
        super().__init__()
        if short_memory_length > memory_length:
            raise ValueError("short_memory_length не может превышать memory_length")
        self.dim: int = dim
        self.num_slots: int = num_slots
        self.memory_length: int = memory_length
        self.short_memory_length: int = short_memory_length
        self.write_threshold: float = write_threshold
        self.max_missed: int = max_missed
        self.association_iou_threshold: float = association_iou_threshold
        self.association_cosine_threshold: float = association_cosine_threshold
        self.association_appearance_weight: float = association_appearance_weight
        self.motion_momentum: float = motion_momentum
        self.initial_dmat = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.initial_dmat, std=dim**-0.5)

        self.short_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.long_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.fusion_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.query_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.short_norm = nn.LayerNorm(dim)
        self.long_norm = nn.LayerNorm(dim)
        self.fusion_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, dim)
        self.time_projection = nn.Sequential(nn.Linear(2, dim), nn.Tanh())

    @staticmethod
    def _checked_state(state: MemoryState) -> MeMOTState:
        """Сделать ошибку смешения state разных веток явной."""
        if not isinstance(state, MeMOTState):
            raise TypeError("MeMOTMemory ожидает MeMOTState")
        return state

    @staticmethod
    def _context_horizon(context: ContextBatch) -> Tensor:
        """Временной масштаб доступного контекста, не считая padding."""
        offsets: Tensor = context.time_offsets.masked_fill(~context.valid_mask, 0.0)
        horizon: Tensor = offsets.amax(dim=-1)
        return torch.where(horizon > 0, horizon, torch.ones_like(horizon))

    @staticmethod
    def _resolve_timestamp(
        state: MeMOTState | None,
        context: ContextBatch,
        current_timestamp: Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Получить время кадра; fallback использует реальные time_offsets."""
        if current_timestamp is not None:
            if current_timestamp.shape != (batch_size,):
                raise ValueError("current_timestamp должен иметь форму [B]")
            if current_timestamp.device != device:
                raise ValueError(
                    "current_timestamp и состояние находятся на разных device"
                )
            resolved: Tensor = current_timestamp.to(dtype=torch.float32)
            if state is not None and (resolved < state.clock).any():
                raise ValueError("current_timestamp не может идти назад")
            return resolved
        if state is None:
            return torch.zeros(batch_size, device=device, dtype=torch.float32)

        positive: Tensor = context.time_offsets.masked_fill(
            ~context.valid_mask, torch.inf
        )
        step: Tensor = positive.amin(dim=-1)
        step = torch.where(torch.isfinite(step), step, torch.ones_like(step))
        return state.clock + step.to(device=device, dtype=torch.float32)

    @staticmethod
    def _predict_boxes(state: MeMOTState, current_timestamp: Tensor) -> Tensor:
        """Constant-velocity перенос последнего бокса к текущему времени."""
        if state.motion is None:
            return state.box
        dt: Tensor = (current_timestamp[:, None] - state.timestamp).clamp_min(0.0)
        return (state.box + state.motion * dt.unsqueeze(-1)).clamp(0.0, 1.0)

    @staticmethod
    def _box_iou(first: Tensor, second: Tensor) -> Tensor:
        """Попарный IoU для cxcywh-боксов: [N,4] x [S,4] -> [N,S]."""

        def xyxy(boxes: Tensor) -> Tensor:
            half: Tensor = boxes[:, 2:] / 2
            return torch.cat((boxes[:, :2] - half, boxes[:, :2] + half), dim=-1)

        a: Tensor = xyxy(first)
        b: Tensor = xyxy(second)
        top_left: Tensor = torch.maximum(a[:, None, :2], b[None, :, :2])
        bottom_right: Tensor = torch.minimum(a[:, None, 2:], b[None, :, 2:])
        intersection: Tensor = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
        area_a: Tensor = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(dim=-1)
        area_b: Tensor = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(dim=-1)
        union: Tensor = area_a[:, None] + area_b[None, :] - intersection
        return intersection / union.clamp_min(torch.finfo(union.dtype).eps)

    def _match_existing(
        self,
        state: MeMOTState,
        batch_index: int,
        detection_indices: Tensor,
        boxes: Tensor,
        features: Tensor,
        predicted_boxes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Жадное one-to-one сопоставление confident detections и track slots."""
        assignments: Tensor = torch.full(
            (detection_indices.numel(),),
            -1,
            dtype=torch.int64,
            device=boxes.device,
        )
        matched_slots: Tensor = torch.zeros(
            self.num_slots, dtype=torch.bool, device=boxes.device
        )
        eligible: Tensor = state.valid[batch_index] & (
            state.missed[batch_index] <= self.max_missed
        )
        slot_indices: Tensor = eligible.nonzero(as_tuple=False).flatten()
        if not detection_indices.numel() or not slot_indices.numel():
            return assignments, matched_slots

        candidate_boxes: Tensor = boxes[detection_indices]
        candidate_features: Tensor = features[detection_indices]
        iou: Tensor = self._box_iou(
            candidate_boxes.detach(),
            predicted_boxes[batch_index, slot_indices].detach(),
        )
        cosine: Tensor = F.cosine_similarity(
            candidate_features.detach()[:, None, :],
            state.feature[batch_index, slot_indices].detach()[None, :, :],
            dim=-1,
        )
        allowed: Tensor = (iou >= self.association_iou_threshold) | (
            cosine >= self.association_cosine_threshold
        )
        score: Tensor = iou + self.association_appearance_weight * (cosine + 1.0) / 2.0
        score = score.masked_fill(~allowed, -torch.inf)

        for flat_index in score.flatten().argsort(descending=True):
            flat: int = int(flat_index)
            det_local: int = flat // slot_indices.numel()
            slot_local: int = flat % slot_indices.numel()
            if not bool(torch.isfinite(score[det_local, slot_local])):
                break
            slot: int = int(slot_indices[slot_local])
            if assignments[det_local] >= 0 or matched_slots[slot]:
                continue
            assignments[det_local] = slot
            matched_slots[slot] = True
        return assignments, matched_slots

    @staticmethod
    def _attention(
        module: nn.MultiheadAttention,
        query: Tensor,
        memory: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Cross-attention без NaN для строк с полностью пустой памятью."""
        has_memory: Tensor = valid.any(dim=-1)
        safe_valid: Tensor = valid.clone()
        safe_valid[~has_memory, 0] = True
        value: Tensor
        weights: Tensor
        value, weights = module(
            query,
            memory,
            memory,
            key_padding_mask=~safe_valid,
            need_weights=True,
        )
        value = value.masked_fill(~has_memory[:, None, None], 0.0)
        weights = weights.masked_fill(~has_memory[:, None, None], 0.0)
        return value, weights

    @staticmethod
    def _latest(history: Tensor, valid: Tensor) -> Tensor:
        """Взять последнее валидное наблюдение каждого object slot."""
        history_length: int = history.shape[-2]
        reverse_index: Tensor = valid.flip(-1).to(torch.int64).argmax(dim=-1)
        latest_index: Tensor = history_length - reverse_index - 1
        gather_index: Tensor = latest_index[..., None, None].expand(
            -1, -1, 1, history.shape[-1]
        )
        latest: Tensor = history.gather(dim=2, index=gather_index).squeeze(2)
        return latest.masked_fill(~valid.any(dim=-1, keepdim=True), 0.0)

    def read(
        self,
        queries: Tensor,
        state: MemoryState | None,
        context: ContextBatch,
        encoded_context: list[Tensor] | None = None,
        query_reference_points: Tensor | None = None,
        current_timestamp: Tensor | None = None,
    ) -> ContextOutput:
        """Агрегировать short/long history и прочитать track tokens queries."""
        batch_size: int = queries.shape[0]
        if query_reference_points is not None:
            expected_shape: tuple[int, int, int] = (*queries.shape[:2], 4)
            if query_reference_points.shape != expected_shape:
                raise ValueError(
                    "query_reference_points имеет форму "
                    f"{tuple(query_reference_points.shape)}, "
                    f"ожидалось {expected_shape}"
                )
            if query_reference_points.device != queries.device:
                raise ValueError(
                    "query_reference_points и queries находятся на разных device"
                )
        if state is None:
            empty_weights: Tensor = queries.new_zeros(
                batch_size, queries.shape[1], self.num_slots
            )
            return ContextOutput(
                query_delta=torch.zeros_like(queries),
                diagnostics={
                    "active_slots": 0,
                    "mean_age": 0.0,
                    "read_weights": empty_weights,
                    "write_rate": 0.0,
                    "evicted": 0,
                },
            )

        memot_state: MeMOTState = self._checked_state(state)
        if memot_state.batch_size != batch_size:
            raise ValueError("B у queries и MeMOTState не совпадают")
        if memot_state.num_slots != self.num_slots:
            raise ValueError("S у MeMOTState не совпадает с конфигурацией")
        if memot_state.feature.shape[-1] != self.dim:
            raise ValueError("D у MeMOTState не совпадает с конфигурацией")
        if context.valid_mask.shape[0] != batch_size:
            raise ValueError("B у ContextBatch и queries не совпадают")

        current_time: Tensor = self._resolve_timestamp(
            memot_state,
            context,
            current_timestamp,
            batch_size,
            queries.device,
        )
        horizon: Tensor = self._context_horizon(context).to(device=queries.device)
        history_dt: Tensor = (
            current_time[:, None, None] - memot_state.history_timestamp
        ).clamp_min(0.0)
        temporal_input: Tensor = torch.stack(
            (
                torch.log1p(history_dt),
                history_dt
                / horizon[:, None, None].clamp_min(torch.finfo(horizon.dtype).eps),
            ),
            dim=-1,
        ).to(dtype=queries.dtype)
        history: Tensor = memot_state.history_feature.to(dtype=queries.dtype)
        history = history + self.time_projection(temporal_input)
        history_valid: Tensor = memot_state.history_valid
        context_available: Tensor = context.valid_mask.any(dim=-1, keepdim=True)
        active: Tensor = (
            memot_state.valid & history_valid.any(dim=-1) & context_available
        )
        latest: Tensor = self._latest(history, history_valid)
        flat_latest: Tensor = latest.flatten(0, 1).unsqueeze(1)

        short_history: Tensor = history[:, :, -self.short_memory_length :]
        short_valid: Tensor = history_valid[:, :, -self.short_memory_length :]
        short_delta: Tensor
        short_weights: Tensor
        short_delta, short_weights = self._attention(
            self.short_attention,
            flat_latest,
            short_history.flatten(0, 1),
            short_valid.flatten(0, 1),
        )
        short_token: Tensor = self.short_norm(flat_latest + short_delta)

        initial_dmat: Tensor = self.initial_dmat.expand(batch_size, self.num_slots, -1)
        dmat: Tensor = torch.where(
            (memot_state.age <= 1).unsqueeze(-1),
            initial_dmat,
            memot_state.dmat,
        ).to(dtype=queries.dtype)
        flat_dmat: Tensor = dmat.flatten(0, 1).unsqueeze(1)
        long_delta: Tensor
        long_weights: Tensor
        long_delta, long_weights = self._attention(
            self.long_attention,
            flat_dmat,
            history.flatten(0, 1),
            history_valid.flatten(0, 1),
        )
        long_token: Tensor = self.long_norm(flat_dmat + long_delta)

        fusion_input: Tensor = torch.cat((short_token, long_token), dim=1)
        fusion_delta: Tensor
        fusion_delta, _ = self.fusion_attention(
            fusion_input, fusion_input, fusion_input, need_weights=False
        )
        fused: Tensor = self.fusion_norm(fusion_input + fusion_delta)
        track_token: Tensor = fused[:, 0].unflatten(0, (batch_size, self.num_slots))
        next_dmat: Tensor = fused[:, 1].unflatten(0, (batch_size, self.num_slots))
        track_token = track_token.masked_fill(~active.unsqueeze(-1), 0.0)
        next_dmat = torch.where(active.unsqueeze(-1), next_dmat, dmat)

        query_delta: Tensor
        read_weights: Tensor
        query_delta, read_weights = self._attention(
            self.query_attention,
            queries,
            track_token,
            active,
        )
        query_delta = self.output_projection(query_delta)
        query_delta = query_delta.masked_fill(~active.any(dim=-1)[:, None, None], 0.0)

        updated_state: MeMOTState = MeMOTState.model_construct(
            **{
                **memot_state.__dict__,
                "dmat": next_dmat.to(dtype=torch.float32),
            }
        )
        valid_age: Tensor = memot_state.age[active]
        mean_age: float = (
            float(valid_age.to(torch.float32).mean()) if valid_age.numel() else 0.0
        )
        observed: Tensor = (
            memot_state.observed
            if memot_state.observed is not None
            else torch.zeros_like(active)
        )
        return ContextOutput(
            query_delta=query_delta,
            memory_state=updated_state,
            diagnostics={
                "active_slots": int(active.sum()),
                "mean_age": mean_age,
                "read_weights": read_weights,
                "short_read_weights": short_weights.unflatten(
                    0, (batch_size, self.num_slots)
                ).squeeze(2),
                "long_read_weights": long_weights.unflatten(
                    0, (batch_size, self.num_slots)
                ).squeeze(2),
                "write_rate": float(observed.to(torch.float32).mean()),
                "evicted": int(memot_state.evicted.sum()),
                "mean_missed": float(
                    memot_state.missed[active].to(torch.float32).mean()
                )
                if active.any()
                else 0.0,
            },
        )

    def write(
        self,
        state: MemoryState | None,
        output: DetectorOutput,
        context: ContextBatch,
        current_timestamp: Tensor | None = None,
    ) -> MemoryState:
        """Сопоставить detections со слотами и обновить FIFO-history."""
        batch_size: int = output.queries.shape[0]
        was_empty: bool = state is None
        if state is None:
            memot_state: MeMOTState = MeMOTState.create(
                batch_size=batch_size,
                num_slots=self.num_slots,
                dim=self.dim,
                memory_length=self.memory_length,
                device=output.queries.device,
            )
        else:
            memot_state = self._checked_state(state)
        if memot_state.batch_size != batch_size:
            raise ValueError("B у DetectorOutput и MeMOTState не совпадают")
        if memot_state.num_slots != self.num_slots:
            raise ValueError("S у MeMOTState не совпадает с конфигурацией")
        if output.queries.shape[-1] != self.dim:
            raise ValueError("D у DetectorOutput не совпадает с конфигурацией")

        current_time: Tensor = self._resolve_timestamp(
            None if was_empty else memot_state,
            context,
            current_timestamp,
            batch_size,
            output.queries.device,
        )
        features: Tensor = output.queries.to(dtype=torch.float32)
        boxes: Tensor = output.boxes.to(dtype=torch.float32)
        confidence: Tensor = output.logits.sigmoid().amax(dim=-1).to(torch.float32)
        predicted_boxes: Tensor = self._predict_boxes(memot_state, current_time)

        feature: Tensor = memot_state.feature.clone()
        box: Tensor = memot_state.box.clone()
        timestamp: Tensor = memot_state.timestamp.clone()
        stored_confidence: Tensor = memot_state.confidence.clone()
        age: Tensor = torch.where(
            memot_state.valid,
            memot_state.age + 1,
            torch.zeros_like(memot_state.age),
        )
        valid: Tensor = memot_state.valid.clone()
        observed: Tensor = torch.zeros_like(valid)
        missed: Tensor = torch.where(
            valid, memot_state.missed + 1, torch.zeros_like(memot_state.missed)
        )
        motion: Tensor = (
            torch.zeros_like(memot_state.box)
            if memot_state.motion is None
            else memot_state.motion.clone()
        )
        dmat: Tensor = memot_state.dmat.clone()
        evicted: Tensor = torch.zeros_like(memot_state.evicted)
        history_feature: Tensor = torch.cat(
            (
                memot_state.history_feature[:, :, 1:],
                torch.zeros_like(memot_state.history_feature[:, :, :1]),
            ),
            dim=2,
        )
        history_valid: Tensor = torch.cat(
            (
                memot_state.history_valid[:, :, 1:],
                torch.zeros_like(memot_state.history_valid[:, :, :1]),
            ),
            dim=2,
        )
        history_timestamp: Tensor = torch.cat(
            (
                memot_state.history_timestamp[:, :, 1:],
                torch.zeros_like(memot_state.history_timestamp[:, :, :1]),
            ),
            dim=2,
        )

        for batch_index in range(batch_size):
            detection_indices: Tensor = (
                (confidence[batch_index] >= self.write_threshold)
                .nonzero(as_tuple=False)
                .flatten()
            )
            assignments, matched_slots = self._match_existing(
                memot_state,
                batch_index,
                detection_indices,
                boxes[batch_index],
                features[batch_index],
                predicted_boxes,
            )

            expired: Tensor = (
                memot_state.valid[batch_index]
                & ~matched_slots
                & (missed[batch_index] > self.max_missed)
            )
            reset_slots: Tensor = expired.clone()
            unmatched_local: Tensor = (
                (assignments < 0).nonzero(as_tuple=False).flatten()
            )
            free_slots: Tensor = (
                (~memot_state.valid[batch_index] | expired)
                .nonzero(as_tuple=False)
                .flatten()
            )

            take_free: int = min(unmatched_local.numel(), free_slots.numel())
            if take_free:
                selected_detections: Tensor = unmatched_local[:take_free]
                selected_slots: Tensor = free_slots[:take_free]
                assignments[selected_detections] = selected_slots
                reset_slots[selected_slots] = True

            remaining: Tensor = unmatched_local[take_free:]
            if remaining.numel():
                replaceable: Tensor = (
                    (memot_state.valid[batch_index] & ~matched_slots & ~expired)
                    .nonzero(as_tuple=False)
                    .flatten()
                )
                confidence_order: Tensor = memot_state.confidence[
                    batch_index, replaceable
                ].argsort(descending=False, stable=True)
                replaceable = replaceable[confidence_order]
                age_order: Tensor = memot_state.age[batch_index, replaceable].argsort(
                    descending=True, stable=True
                )
                replaceable = replaceable[age_order]
                missed_order: Tensor = memot_state.missed[
                    batch_index, replaceable
                ].argsort(descending=True, stable=True)
                replaceable = replaceable[missed_order]
                take_replacements: int = min(remaining.numel(), replaceable.numel())
                selected_detections = remaining[:take_replacements]
                selected_slots = replaceable[:take_replacements]
                assignments[selected_detections] = selected_slots
                reset_slots[selected_slots] = True

            evicted[batch_index] = (reset_slots & memot_state.valid[batch_index]).sum()
            if reset_slots.any():
                feature[batch_index, reset_slots] = 0
                box[batch_index, reset_slots] = 0.5
                timestamp[batch_index, reset_slots] = 0
                stored_confidence[batch_index, reset_slots] = 0
                age[batch_index, reset_slots] = 0
                valid[batch_index, reset_slots] = False
                missed[batch_index, reset_slots] = 0
                motion[batch_index, reset_slots] = 0
                dmat[batch_index, reset_slots] = 0
                history_feature[batch_index, reset_slots] = 0
                history_valid[batch_index, reset_slots] = False
                history_timestamp[batch_index, reset_slots] = 0

            for local_index, detection_index in enumerate(detection_indices):
                slot: int = int(assignments[local_index])
                if slot < 0:
                    continue
                detection: int = int(detection_index)
                is_new: bool = bool(reset_slots[slot]) or not bool(
                    memot_state.valid[batch_index, slot]
                )
                if not is_new:
                    dt: Tensor = (
                        current_time[batch_index]
                        - memot_state.timestamp[batch_index, slot]
                    ).clamp_min(torch.finfo(current_time.dtype).eps)
                    instant_motion: Tensor = (
                        boxes[batch_index, detection]
                        - memot_state.box[batch_index, slot]
                    ) / dt
                    motion[batch_index, slot] = (
                        self.motion_momentum * motion[batch_index, slot]
                        + (1.0 - self.motion_momentum) * instant_motion
                    )
                else:
                    age[batch_index, slot] = 1

                feature[batch_index, slot] = features[batch_index, detection]
                box[batch_index, slot] = boxes[batch_index, detection]
                timestamp[batch_index, slot] = current_time[batch_index]
                stored_confidence[batch_index, slot] = confidence[
                    batch_index, detection
                ]
                valid[batch_index, slot] = True
                observed[batch_index, slot] = True
                missed[batch_index, slot] = 0
                history_feature[batch_index, slot, -1] = features[
                    batch_index, detection
                ]
                history_valid[batch_index, slot, -1] = True
                history_timestamp[batch_index, slot, -1] = current_time[batch_index]

        return MeMOTState(
            feature=feature,
            box=box,
            timestamp=timestamp,
            confidence=stored_confidence,
            age=age,
            valid=valid,
            observed=observed,
            motion=motion,
            history_feature=history_feature,
            history_valid=history_valid,
            history_timestamp=history_timestamp,
            dmat=dmat,
            missed=missed,
            clock=current_time,
            evicted=evicted,
        )


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
