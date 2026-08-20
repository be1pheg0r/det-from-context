"""MeMOT on top of the existing RF-DETR + MeMOTMemory components.

This module deliberately reuses the project's existing DetectorAdapter/RFDetrAdapter,
MeMOTMemory/MeMOTState and DetectionBatch contracts.  It adds the missing MeMOT
Memory-Decoding part: proposal/track query split, joint RF-DETR decoding, objectness /
uniqueness heads, inference-time track identities, and the paper-style losses.

Expected target format per frame:
    {
        "boxes": Tensor[N,4]        # normalized cxcywh
        "labels": Tensor[N]         # optional for tracking loss; used by aux DETR loss
        "instance_ids": Tensor[N]   # REQUIRED for MOT training
    }
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..contracts import ContextBatch, DetectionBatch, DetectorOutput
from .detector import DetectorAdapter
from .memory import MeMOTMemory, MeMOTState
from .losses import generalized_box_iou, box_cxcywh_to_xyxy, SetCriterion


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _focal_binary(logits: Tensor, target: Tensor, alpha: float = 0.25, gamma: float = 2.0) -> Tensor:
    target = target.to(dtype=logits.dtype)
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1.0 - p) * (1.0 - target)
    at = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (at * (1.0 - pt).pow(gamma) * ce).mean()


def _giou_diag(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    if boxes1.numel() == 0:
        return boxes1.new_zeros(())
    return generalized_box_iou(
        box_cxcywh_to_xyxy(boxes1), box_cxcywh_to_xyxy(boxes2)
    ).diag()


def _pairwise_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    a = box_cxcywh_to_xyxy(boxes1)
    b = box_cxcywh_to_xyxy(boxes2)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp_min(1e-6)


def _hungarian(cost: Tensor) -> tuple[Tensor, Tensor]:
    """CPU scipy Hungarian, kept outside autograd exactly like DETR matching."""
    from scipy.optimize import linear_sum_assignment

    row, col = linear_sum_assignment(cost.detach().float().cpu().numpy())
    return (
        torch.as_tensor(row, dtype=torch.long, device=cost.device),
        torch.as_tensor(col, dtype=torch.long, device=cost.device),
    )


# ---------------------------------------------------------------------------
# Runtime state: detector/memory state + stable external track IDs.
# ---------------------------------------------------------------------------

@dataclass
class MeMOTRuntimeState:
    memory: MeMOTState
    track_ids: Tensor       # [B,S], int64; -1 means slot has no public identity yet
    next_track_id: Tensor   # [B], int64

    def detach(self) -> "MeMOTRuntimeState":
        return MeMOTRuntimeState(
            memory=self.memory.detach(),
            track_ids=self.track_ids,
            next_track_id=self.next_track_id,
        )

    def reset(self, mask: Tensor) -> "MeMOTRuntimeState":
        if not bool(mask.any()):
            return self
        b, s = self.track_ids.shape
        m = mask[:, None]
        ids = torch.where(m, torch.full_like(self.track_ids, -1), self.track_ids)
        nxt = torch.where(mask, torch.zeros_like(self.next_track_id), self.next_track_id)
        return MeMOTRuntimeState(self.memory.reset(mask), ids, nxt)


@dataclass
class MeMOTOutput:
    detector: DetectorOutput
    proposal: DetectorOutput
    proposal_boxes: Tensor
    proposal_objectness: Tensor
    proposal_uniqueness: Tensor
    track_boxes: Tensor
    track_objectness: Tensor
    track_uniqueness: Tensor
    track_ids: Tensor
    track_slot_indices: Tensor
    proposal_queries: Tensor
    track_queries: Tensor

    @property
    def proposal_confidence(self) -> Tensor:
        return self.proposal_objectness.sigmoid() * self.proposal_uniqueness.sigmoid()

    @property
    def track_confidence(self) -> Tensor:
        return self.track_objectness.sigmoid() * self.track_uniqueness.sigmoid()


# ---------------------------------------------------------------------------
# MeMOT model
# ---------------------------------------------------------------------------

class RFDETRMeMOT(nn.Module):
    """Full MeMOT-style tracker using the project's RF-DETR adapter.

    Flow per frame:
      1. RF-DETR hypothesis generation -> proposal embeddings.
      2. Existing MeMOTMemory encodes each active slot -> track embeddings.
      3. Proposal + track embeddings are concatenated and sent through RF-DETR
         again, which supplies the Transformer image cross-attention and query
         interaction.
      4. Two new scalar heads predict objectness and uniqueness.
      5. Existing MeMOTMemory.write updates the FIFO history.

    The second RF-DETR call is intentional: the existing adapter exposes a safe
    query-injection API but not a public "encode once / decode twice" API. This
    keeps the upstream RF-DETR implementation untouched and makes the model
    immediately compatible with the supplied codebase.
    """

    def __init__(
        self,
        detector: DetectorAdapter,
        memory: MeMOTMemory,
        *,
        objectness_hidden: int | None = None,
        track_query_norm: bool = True,
        proposal_aux: bool = True,
        proposal_count: int | None = None,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.memory = memory
        self.dim = detector.dim
        self.proposal_count = proposal_count
        self.proposal_aux = proposal_aux

        hidden = objectness_hidden or self.dim
        self.query_norm = nn.LayerNorm(self.dim) if track_query_norm else nn.Identity()
        self.objectness_head = nn.Sequential(
            nn.Linear(self.dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1)
        )
        self.uniqueness_head = nn.Sequential(
            nn.Linear(self.dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1)
        )

        # The supplied RF-DETR already has a trained detector head. We reuse
        # that exact proposal logits/boxes for the paper's auxiliary DETR loss
        # instead of adding a second classifier.

        self.register_buffer("_dummy", torch.empty(0), persistent=False)

    # ------------------------ state helpers ------------------------
    def init_state(self, batch_size: int, device: torch.device | str) -> MeMOTRuntimeState:
        memory = MeMOTState.create(
            batch_size=batch_size,
            num_slots=self.memory.num_slots,
            dim=self.dim,
            memory_length=self.memory.memory_length,
            device=device,
        )
        return MeMOTRuntimeState(
            memory=memory,
            track_ids=torch.full(
                (batch_size, self.memory.num_slots), -1, dtype=torch.long, device=device
            ),
            next_track_id=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    def _active_track_queries(
        self, state: MeMOTRuntimeState | None
    ) -> tuple[Tensor, Tensor]:
        if state is None:
            return (
                self._dummy.new_zeros((0, 0, self.dim)),
                self._dummy.new_zeros((0, 0), dtype=torch.bool),
            )
        valid = state.memory.valid
        # [B,S,D] -> kept as fixed S slots. Inactive slots are zero and are masked
        # out before concatenation by per-batch slicing in forward.
        base = state.memory.feature.to(dtype=self._dummy.dtype if self._dummy.numel() else torch.float32)
        base = base.to(device=state.memory.feature.device)
        return self.query_norm(base), valid

    # ------------------------ query construction ------------------------
    def _make_track_queries(
        self, state: MeMOTRuntimeState | None, dtype: torch.dtype
    ) -> tuple[list[Tensor], list[Tensor]]:
        if state is None:
            b = 0
            return [], []
        valid = state.memory.valid
        base = state.memory.feature.to(dtype=dtype)
        tracks: list[Tensor] = []
        slots: list[Tensor] = []
        for bi in range(base.shape[0]):
            idx = valid[bi].nonzero(as_tuple=False).flatten()
            tracks.append(self.query_norm(base[bi, idx]))
            slots.append(idx)
        return tracks, slots

    def _encode_tracks(
        self,
        state: MeMOTRuntimeState | None,
        context: ContextBatch,
        timestamp: Tensor,
    ) -> tuple[list[Tensor], list[Tensor], MeMOTRuntimeState | None]:
        """Run the existing MeMOTMemory encoder separately for each active slot."""
        if state is None:
            return [], [], state
        b, s = state.memory.valid.shape
        dtype = self._module_dtype()
        base = state.memory.feature.to(dtype=dtype)
        # Existing memory.read is query-driven. Querying with the slot's current
        # identity feature is the closest compatible replacement for the paper's
        # explicit track-token interface.
        ctx = self.memory.read(
            base,
            state.memory,
            context,
            current_timestamp=timestamp,
        )
        if ctx.memory_state is not None:
            state.memory = ctx.memory_state
        delta = ctx.query_delta if ctx.query_delta is not None else torch.zeros_like(base)
        track_all = self.query_norm(base + delta)
        tracks: list[Tensor] = []
        slots: list[Tensor] = []
        for bi in range(b):
            idx = state.memory.valid[bi].nonzero(as_tuple=False).flatten()
            tracks.append(track_all[bi, idx])
            slots.append(idx)
        return tracks, slots, state

    def _module_dtype(self) -> torch.dtype:
        for p in self.parameters():
            if p.is_floating_point():
                return p.dtype
        return torch.float32

    # ------------------------ forward ------------------------
    def forward(
        self,
        batch: DetectionBatch,
        context: ContextBatch,
        state: MeMOTRuntimeState | None = None,
        *,
        training_targets: bool = False,
    ) -> tuple[MeMOTOutput, MeMOTRuntimeState]:
        b = batch.batch_size
        device = batch.images.device
        if state is None:
            state = self.init_state(b, device)
        elif batch.is_sequence_start is not None:
            state = state.reset(batch.is_sequence_start)

        # ---------------- hypothesis generation ----------------
        seed_queries = self.detector.initial_queries(batch)
        proposal = self.detector(batch, query_init=seed_queries)
        proposal_queries = proposal.queries
        if self.proposal_count is not None:
            proposal_queries = proposal_queries[:, : self.proposal_count]

        # ---------------- memory encoding ----------------
        track_lists, slot_lists, state = self._encode_tracks(
            state, context, batch.timestamp
        )

        # ---------------- memory decoding ----------------
        # RF-DETR requires the decoder query count to stay fixed. MeMOT therefore
        # reserves the full memory-slot budget for track queries, even when no
        # tracks are active yet. Inactive slots are masked later.
        n_prop = proposal_queries.shape[1]
        max_tracks = self.memory.num_slots
        padded_tracks = proposal_queries.new_zeros((b, max_tracks, self.dim))
        track_mask = torch.zeros((b, max_tracks), dtype=torch.bool, device=device)
        for bi, tr in enumerate(track_lists):
            if tr.numel():
                n = min(tr.shape[0], max_tracks)
                padded_tracks[bi, :n] = tr[:n]
                track_mask[bi, :n] = True

        combined_queries = torch.cat([proposal_queries, padded_tracks], dim=1)
        expected_queries = self.detector.initial_queries(batch).shape[1]
        if combined_queries.shape[1] != expected_queries:
            raise RuntimeError(
                "MeMOT query budget mismatch: "
                f"proposal={n_prop}, tracks={max_tracks}, "
                f"combined={combined_queries.shape[1]}, expected={expected_queries}"
            )
        decoded = self.detector(batch, query_init=combined_queries)
        q = decoded.queries
        proposal_q = q[:, :n_prop]
        track_q = q[:, n_prop:]

        # RF-DETR's own box head is reused for both proposal and track boxes.
        all_boxes = decoded.boxes
        proposal_boxes = all_boxes[:, :n_prop]
        track_boxes_padded = all_boxes[:, n_prop:]

        objectness = self.objectness_head(q).squeeze(-1)
        uniqueness = self.uniqueness_head(q).squeeze(-1)
        proposal_obj = objectness[:, :n_prop]
        proposal_uni = uniqueness[:, :n_prop]
        track_obj = objectness[:, n_prop:]
        track_uni = uniqueness[:, n_prop:]

        # Update memory with the actual post-association query states, not the
        # proposal-only pass. This is the identity state used by future frames.
        # Memory.write uses DetectorOutput.logits only as an objectness-like
        # write gate. Feed it the dedicated MeMOT objectness head instead of
        # RF-DETR class confidence. The public DetectorOutput contract permits
        # an arbitrary class dimension, and MeMOTMemory only consumes the max
        # sigmoid value here.
        write_logits = proposal_obj.unsqueeze(-1)
        write_output = DetectorOutput(
            logits=write_logits,
            boxes=proposal_boxes,
            queries=proposal_q,
            reference_points=decoded.reference_points[:, :n_prop],
            features=decoded.features,
            decoder_layers=decoded.decoder_layers,
            aux=decoded.aux,
        )
        state.memory = self.memory.write(
            state.memory,
            write_output,
            context,
            current_timestamp=batch.timestamp,
        )

        # Public track IDs: existing slots keep their identity; new occupied
        # slots receive monotonically increasing IDs. Memory.write can replace a
        # slot, so compare slot validity before/after and assign IDs to newly
        # occupied slots.
        new_valid = state.memory.valid
        for bi in range(b):
            old_ids = state.track_ids[bi]
            for slot in new_valid[bi].nonzero(as_tuple=False).flatten().tolist():
                if int(old_ids[slot]) < 0:
                    old_ids[slot] = state.next_track_id[bi]
                    state.next_track_id[bi] += 1
            invalid = ~new_valid[bi]
            old_ids[invalid] = -1

        # Pad-track output is deliberately retained; inactive slots are masked
        # with zero confidence so callers can ignore them.
        track_obj = track_obj.masked_fill(~track_mask, -20.0)
        track_uni = track_uni.masked_fill(~track_mask, 20.0)
        track_boxes_padded = track_boxes_padded.masked_fill(~track_mask[..., None], 0.0)

        # Build a compact public identity matrix [B,max_tracks].
        track_ids = torch.full(
            (b, max_tracks), -1, dtype=torch.long, device=device
        )
        for bi, idx in enumerate(slot_lists):
            if idx.numel():
                track_ids[bi, : idx.numel()] = state.track_ids[bi, idx]

        return (
            MeMOTOutput(
                detector=decoded,
                proposal=proposal,
                proposal_boxes=proposal_boxes,
                proposal_objectness=proposal_obj,
                proposal_uniqueness=proposal_uni,
                track_boxes=track_boxes_padded,
                track_objectness=track_obj,
                track_uniqueness=track_uni,
                track_ids=track_ids,
                track_slot_indices=torch.nn.utils.rnn.pad_sequence(
                    slot_lists, batch_first=True, padding_value=-1
                ) if slot_lists else torch.empty((b, 0), dtype=torch.long, device=device),
                proposal_queries=proposal_q,
                track_queries=track_q,
            ),
            state.detach() if self.training else state,
        )

    # ------------------------ losses ------------------------
    def loss(
        self,
        output: MeMOTOutput,
        target: dict[str, Tensor],
        *,
        seen_ids: set[int],
        track_target_ids: Tensor | None = None,
        lambda_cls: float = 2.0,
        lambda_l1: float = 5.0,
        lambda_iou: float = 2.0,
        lambda_det: float = 1.0,
        lambda_tck: float = 1.0,
    ) -> dict[str, Tensor]:
        """Paper-style tracking + auxiliary proposal detection loss.

        `track_target_ids` is [S_active] and identifies which GT instance each
        active memory slot currently represents. The trainer computes it from
        the previous frame's slot-to-GT matching, keeping this method stateless.
        """
        gt_boxes = target["boxes"].to(output.proposal_boxes.device)
        gt_labels = target.get("labels")
        if gt_labels is None:
            gt_labels = torch.zeros(gt_boxes.shape[0], dtype=torch.long, device=gt_boxes.device)
        else:
            gt_labels = gt_labels.to(gt_boxes.device)
        gt_ids = target.get("instance_ids")
        if gt_ids is None:
            raise ValueError("MeMOT training requires target['instance_ids']")
        gt_ids = gt_ids.to(gt_boxes.device).long()

        # ---- proposal auxiliary DETR-like loss ----
        if self.proposal_aux:
            criterion = SetCriterion(
                num_classes=output.proposal.logits.shape[-1],
                cls_weight=lambda_cls,
                bbox_weight=lambda_l1,
                giou_weight=lambda_iou,
            )
            aux_loss = criterion(
                output.proposal.logits,
                output.proposal.boxes,
                [target],
            )["loss_total"]
        else:
            aux_loss = output.proposal_boxes.sum() * 0.0

        # ---- main proposal matching ----
        p_obj_t, p_uni_t, p_box_t, p_pos = self._proposal_targets(
            output.proposal_boxes[0], gt_boxes, gt_ids, seen_ids
        )
        # Use batch_size=1 here; trainer calls loss per frame. This avoids hidden
        # assumptions about cross-sequence IDs in one loss call.
        p_obj = output.proposal_objectness[0]
        p_uni = output.proposal_uniqueness[0]
        p_box = output.proposal_boxes[0]
        loss_prop = self._tracking_entry_loss(
            p_obj, p_uni, p_box, p_obj_t, p_uni_t, p_box_t,
            lambda_cls, lambda_l1, lambda_iou,
        )

        # ---- track-query targets ----
        t_obj = output.track_objectness[0]
        t_uni = output.track_uniqueness[0]
        t_box = output.track_boxes[0]
        n_t = t_obj.shape[0]
        track_obj_t = torch.zeros(n_t, device=t_obj.device)
        track_uni_t = torch.ones(n_t, device=t_obj.device)
        track_box_t = torch.zeros(n_t, 4, device=t_obj.device)
        if track_target_ids is not None and track_target_ids.numel():
            ids = track_target_ids.to(gt_ids.device)
            for j, gid in enumerate(ids.tolist()):
                hit = (gt_ids == gid).nonzero(as_tuple=False).flatten()
                if hit.numel():
                    k = int(hit[0])
                    track_obj_t[j] = 1.0
                    track_box_t[j] = gt_boxes[k]

        loss_track = self._tracking_entry_loss(
            t_obj, t_uni, t_box, track_obj_t, track_uni_t, track_box_t,
            lambda_cls, lambda_l1, lambda_iou,
        )

        total = lambda_tck * (loss_track + loss_prop) + lambda_det * aux_loss
        return {
            "loss_total": total,
            "loss_track": loss_track,
            "loss_proposal": loss_prop,
            "loss_aux_det": aux_loss,
        }

    @staticmethod
    def _tracking_entry_loss(
        obj: Tensor,
        uni: Tensor,
        boxes: Tensor,
        obj_t: Tensor,
        uni_t: Tensor,
        box_t: Tensor,
        lambda_cls: float,
        lambda_l1: float,
        lambda_iou: float,
    ) -> Tensor:
        loss_obj = _focal_binary(obj, obj_t)
        loss_uni = _focal_binary(uni, uni_t)
        pos = obj_t > 0.5
        if bool(pos.any()):
            l1 = F.l1_loss(boxes[pos], box_t[pos], reduction="mean")
            giou = (1.0 - _giou_diag(boxes[pos], box_t[pos])).mean()
        else:
            l1 = boxes.sum() * 0.0
            giou = boxes.sum() * 0.0
        return lambda_cls * (loss_obj + loss_uni) + lambda_l1 * l1 + lambda_iou * giou

    @staticmethod
    def _proposal_targets(
        pred_boxes: Tensor,
        gt_boxes: Tensor,
        gt_ids: Tensor,
        seen_ids: set[int],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        n = pred_boxes.shape[0]
        obj = pred_boxes.new_zeros(n)
        uni = pred_boxes.new_ones(n)
        boxes = pred_boxes.new_zeros((n, 4))
        if gt_boxes.numel() == 0:
            return obj, uni, boxes, pred_boxes.new_zeros(0, dtype=torch.long)

        cost = torch.cdist(pred_boxes, gt_boxes, p=1) - _pairwise_iou(pred_boxes, gt_boxes)
        row, col = _hungarian(cost)
        obj[row] = 1.0
        boxes[row] = gt_boxes[col]
        for r, c in zip(row.tolist(), col.tolist()):
            # u=1 only for a genuinely new object; if this instance has already
            # appeared earlier in the clip, the proposal must be suppressed.
            uni[r] = 0.0 if int(gt_ids[c]) in seen_ids else 1.0
        return obj, uni, boxes, row

    @staticmethod
    def _set_detection_loss(
        logits: Tensor,
        boxes: Tensor,
        gt_boxes: Tensor,
        lambda_cls: float,
        lambda_l1: float,
        lambda_iou: float,
    ) -> Tensor:
        if gt_boxes.numel() == 0:
            return _focal_binary(logits, torch.zeros_like(logits)) * lambda_cls
        cost = torch.cdist(boxes, gt_boxes, p=1) - _pairwise_iou(boxes, gt_boxes)
        row, col = _hungarian(cost)
        target = torch.zeros_like(logits)
        target[row] = 1.0
        loss_cls = _focal_binary(logits, target)
        loss_l1 = F.l1_loss(boxes[row], gt_boxes[col], reduction="mean")
        loss_giou = (1.0 - _giou_diag(boxes[row], gt_boxes[col])).mean()
        return lambda_cls * loss_cls + lambda_l1 * loss_l1 + lambda_iou * loss_giou

    # ------------------------ inference ------------------------
    @torch.no_grad()
    def predict(
        self,
        output: MeMOTOutput,
        *,
        proposal_threshold: float = 0.7,
        track_threshold: float = 0.6,
    ) -> list[dict[str, Tensor]]:
        """Return MOT outputs without NMS/extra association post-processing."""
        results: list[dict[str, Tensor]] = []
        p_score = output.proposal_confidence
        t_score = output.track_confidence
        for b in range(p_score.shape[0]):
            keep_p = p_score[b] >= proposal_threshold
            keep_t = t_score[b] >= track_threshold
            p_boxes = output.proposal_boxes[b, keep_p]
            p_scores = p_score[b, keep_p]
            p_ids = torch.full_like(p_scores, -1, dtype=torch.long)
            t_boxes = output.track_boxes[b, keep_t]
            t_scores = t_score[b, keep_t]
            t_ids = output.track_ids[b, keep_t]
            boxes = torch.cat([t_boxes, p_boxes], dim=0)
            scores = torch.cat([t_scores, p_scores], dim=0)
            ids = torch.cat([t_ids, p_ids], dim=0)
            results.append({"boxes": boxes, "scores": scores, "track_ids": ids})
        return results


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

class MeMOTClipTrainer:
    """Minimal clip-centric trainer compatible with the project's contracts.

    The loader must yield frames in causal order. `batch_size=1` is recommended
    for MOT. For each frame, `target['instance_ids']` must be present.
    """

    def __init__(
        self,
        model: RFDETRMeMOT,
        optimizer: torch.optim.Optimizer,
        *,
        amp: bool = True,
        grad_accum: int = 1,
        detach_state: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.amp = amp and torch.cuda.is_available()
        self.grad_accum = max(1, grad_accum)
        self.detach_state = detach_state
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

    def train_epoch(self, loader: Iterable[tuple[DetectionBatch, ContextBatch]]) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        state: MeMOTRuntimeState | None = None
        seen_by_batch: list[set[int]] | None = None
        total = {"loss_total": 0.0, "loss_track": 0.0, "loss_proposal": 0.0, "loss_aux_det": 0.0}
        steps = 0

        for step, (batch, context) in enumerate(loader):
            if batch.batch_size != 1:
                raise ValueError("MeMOTClipTrainer intentionally requires batch_size=1")
            if state is None:
                state = self.model.init_state(1, batch.images.device)
                seen_by_batch = [set()]
            elif batch.is_sequence_start is not None and bool(batch.is_sequence_start[0]):
                state = state.reset(batch.is_sequence_start)
                seen_by_batch = [set()]

            with torch.autocast(device_type=batch.images.device.type, enabled=self.amp):
                output, new_state = self.model(batch, context, state, training_targets=True)
                target = batch.targets[0]
                track_target_ids = self._slot_ids_for_current_frame(state, target)
                losses = self.model.loss(
                    output,
                    target,
                    seen_ids=seen_by_batch[0],
                    track_target_ids=track_target_ids,
                )
                loss = losses["loss_total"] / self.grad_accum

            self.scaler.scale(loss).backward()
            if (step + 1) % self.grad_accum == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            for k, v in losses.items():
                total[k] += float(v.detach())
            steps += 1
            seen_by_batch[0].update(int(x) for x in target["instance_ids"].detach().cpu().tolist())
            state = new_state.detach() if self.detach_state else new_state

        if steps == 0:
            return {k: 0.0 for k in total}
        return {k: v / steps for k, v in total.items()}

    @staticmethod
    def _slot_ids_for_current_frame(
        state: MeMOTRuntimeState,
        target: dict[str, Tensor],
    ) -> Tensor:
        """Map active public track IDs to current GT instance IDs by box IoU.

        This is only used to construct the supervised target for track queries;
        it does not modify the memory association itself.
        """
        slot_boxes = state.memory.box[0, state.memory.valid[0]]
        slot_ids = state.track_ids[0, state.memory.valid[0]]
        gt_boxes = target["boxes"].to(slot_boxes.device)
        gt_ids = target["instance_ids"].to(slot_boxes.device).long()
        if slot_boxes.numel() == 0 or gt_boxes.numel() == 0:
            return slot_ids.new_empty(0)
        iou = _pairwise_iou(slot_boxes, gt_boxes)
        best = iou.argmax(dim=1)
        return torch.where(iou.max(dim=1).values >= 0.3, gt_ids[best], torch.full_like(slot_ids, -1))
