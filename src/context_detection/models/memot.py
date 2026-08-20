"""External MeMOT adapter operating strictly after current-frame detection."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..contracts import (
    ContextBatch,
    ContextOutput,
    DetectionBatch,
    DetectorOutput,
    MemoryState,
)
from .detector import DetectorAdapter
from .memory import MeMOTMemory, MeMOTState


class MeMOTMemoryEncoder(nn.Module):
    """Expose MeMOT history encoding independently from the detector."""

    def __init__(self, memory: MeMOTMemory) -> None:
        super().__init__()
        self.memory = memory

    def forward(
        self,
        hypotheses: DetectorOutput,
        state: MeMOTState | None,
        context: ContextBatch,
        timestamp: Tensor,
    ) -> ContextOutput:
        """Encode short/long histories and retrieve them for current proposals."""
        return self.memory.read(
            hypotheses.queries,
            state,
            context,
            current_timestamp=timestamp,
        )

    def update(
        self,
        state: MeMOTState | None,
        output: DetectorOutput,
        context: ContextBatch,
        timestamp: Tensor,
    ) -> MeMOTState:
        """Associate decoded proposals and append observations to track memory."""
        return self.memory.write(
            state,
            output,
            context,
            current_timestamp=timestamp,
        )

    def reset(self, state: MeMOTState, mask: Tensor) -> MeMOTState:
        """Reset sequence rows together with their stable track-ID counters."""
        reset = self.memory.reset(state, mask)
        if not isinstance(reset, MeMOTState):
            raise TypeError("MeMOT reset must preserve MeMOTState")
        return reset


class MeMOTMemoryDecoder(nn.Module):
    """Jointly refine RF-DETR hypotheses and predict track association.

    RF-DETR queries and feature maps are treated as immutable current-frame
    hypotheses. Track context is fused only here, after the official detector
    has completed its forward pass.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_classes: int,
        num_slots: int,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.num_slots = num_slots
        self.image_projection = nn.Linear(3, dim)
        self.memory_gate = nn.Linear(3 * dim, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=4 * dim,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.proposal_decoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(dim)
        self.class_delta = nn.Linear(dim, num_classes)
        self.box_delta = nn.Linear(dim, 4)
        self.new_track_logit = nn.Linear(dim, 1)

        nn.init.constant_(self.memory_gate.bias, -2.0)
        nn.init.zeros_(self.class_delta.weight)
        nn.init.zeros_(self.class_delta.bias)
        nn.init.zeros_(self.box_delta.weight)
        nn.init.zeros_(self.box_delta.bias)
        nn.init.zeros_(self.new_track_logit.weight)
        nn.init.zeros_(self.new_track_logit.bias)

    def _image_token(self, hypotheses: DetectorOutput) -> Tensor:
        """Pool the final RF-DETR feature map without re-running its backbone."""
        if not hypotheses.features:
            return hypotheses.queries.new_zeros(
                hypotheses.queries.shape[0], 1, self.dim
            )
        feature = hypotheses.features[-1]
        if feature.ndim != 4:
            raise ValueError(
                "MeMOT Memory Decoder expects RF-DETR features [B, C, H, W], "
                f"got {tuple(feature.shape)}"
            )
        flattened = feature.flatten(1).to(dtype=hypotheses.queries.dtype)
        statistics = torch.stack(
            (
                flattened.mean(dim=-1),
                flattened.std(dim=-1, unbiased=False),
                flattened.amax(dim=-1),
            ),
            dim=-1,
        )
        return self.image_projection(statistics).unsqueeze(1)

    def _association_logits(
        self,
        decoded: Tensor,
        context_output: ContextOutput,
        state: MeMOTState | None,
    ) -> Tensor:
        """Convert differentiable memory attention into slot/new-track logits."""
        weights = context_output.diagnostics.get("read_weights")
        if not isinstance(weights, Tensor):
            weights = decoded.new_zeros(
                decoded.shape[0], decoded.shape[1], self.num_slots
            )
        if weights.shape != (*decoded.shape[:2], self.num_slots):
            raise ValueError(
                "read_weights must have shape "
                f"{(*decoded.shape[:2], self.num_slots)}, got {tuple(weights.shape)}"
            )
        eps = torch.finfo(decoded.dtype).eps
        existing = weights.to(dtype=decoded.dtype).clamp_min(eps).log()
        if state is None:
            active = torch.zeros(
                decoded.shape[0],
                self.num_slots,
                dtype=torch.bool,
                device=decoded.device,
            )
        else:
            active = state.valid
        existing = existing.masked_fill(~active[:, None, :], -torch.inf)
        return torch.cat((existing, self.new_track_logit(decoded)), dim=-1)

    def forward(
        self,
        hypotheses: DetectorOutput,
        context_output: ContextOutput,
        state: MeMOTState | None,
    ) -> DetectorOutput:
        """Decode memory externally and preserve upstream outputs for RF-DETR loss."""
        if hypotheses.queries.shape[-1] != self.dim:
            raise ValueError("RF-DETR query dimension does not match MeMOT")
        if hypotheses.logits.shape[-1] != self.num_classes:
            raise ValueError("RF-DETR class count does not match MeMOT")
        memory_delta = context_output.query_delta
        if memory_delta is None:
            memory_delta = torch.zeros_like(hypotheses.queries)
        image_token = self._image_token(hypotheses).expand_as(hypotheses.queries)
        joint = torch.cat((hypotheses.queries, memory_delta, image_token), dim=-1)
        gated = hypotheses.queries + torch.sigmoid(self.memory_gate(joint)) * (
            memory_delta + image_token
        )
        decoded = self.output_norm(self.proposal_decoder(gated))
        logits = hypotheses.logits + self.class_delta(decoded)
        box_logits = torch.logit(hypotheses.boxes.clamp(1e-6, 1.0 - 1e-6))
        boxes = (box_logits + self.box_delta(decoded)).sigmoid()
        association_logits = self._association_logits(decoded, context_output, state)
        memot_aux = {
            "association_logits": association_logits,
            "read_weights": context_output.diagnostics.get("read_weights"),
            "active_slots": None if state is None else state.valid,
            "hypothesis_logits": hypotheses.logits,
            "hypothesis_boxes": hypotheses.boxes,
        }
        return DetectorOutput(
            logits=logits,
            boxes=boxes,
            queries=decoded,
            reference_points=hypotheses.reference_points,
            features=hypotheses.features,
            decoder_layers=hypotheses.decoder_layers,
            aux={**hypotheses.aux, "memot": memot_aux},
        )


class MeMOTTracker(nn.Module):
    """RF-DETR Hypothesis Generator followed by external MeMOT modules."""

    def __init__(
        self,
        detector: DetectorAdapter,
        memory_encoder: MeMOTMemoryEncoder,
        memory_decoder: MeMOTMemoryDecoder,
        *,
        detach_state: bool = True,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.memory_encoder = memory_encoder
        self.memory_decoder = memory_decoder
        self.detach_state = detach_state

    def forward(
        self,
        batch: DetectionBatch,
        context: ContextBatch,
        state: MemoryState | None = None,
    ) -> tuple[DetectorOutput, MeMOTState]:
        """Run current-frame detection, memory decoding, then track update."""
        if state is not None and not isinstance(state, MeMOTState):
            raise TypeError("MeMOTTracker expects MeMOTState")
        memot_state = state
        if memot_state is not None and batch.is_sequence_start is not None:
            memot_state = self.memory_encoder.reset(
                memot_state, batch.is_sequence_start
            )

        hypotheses = self.detector(batch)
        encoded = self.memory_encoder(
            hypotheses,
            memot_state,
            context,
            batch.timestamp,
        )
        output = self.memory_decoder(hypotheses, encoded, memot_state)
        encoder_state = (
            encoded.memory_state
            if isinstance(encoded.memory_state, MeMOTState)
            else memot_state
        )
        next_state = self.memory_encoder.update(
            encoder_state,
            output,
            context,
            batch.timestamp,
        )
        memot_aux = output.aux["memot"]
        memot_aux["diagnostics"] = encoded.diagnostics
        if self.detach_state:
            next_state = next_state.detach()
        return output, next_state

    @torch.no_grad()
    def rollout(
        self,
        steps: list[tuple[DetectionBatch, ContextBatch]],
        state: MeMOTState | None = None,
    ) -> list[DetectorOutput]:
        """Run a clip sequentially while carrying track memory between frames."""
        outputs: list[DetectorOutput] = []
        for batch, context in steps:
            output, state = self(batch, context, state)
            outputs.append(output)
        return outputs
