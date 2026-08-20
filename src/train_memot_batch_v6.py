from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import rfdetr
import torch
import yaml
from clearml import Task
from src.context_detection.config import load_config
from src.context_detection.contracts import ContextBatch, DetectionBatch, DetectorOutput
from src.context_detection.models.memory import MeMOTMemory
from src.context_detection.models.memot_rfdetr import MeMOTOutput, RFDETRMeMOT
from src.context_detection.models.protocols import build_registered_detector
from src.datasets.video_dataloader.dataloader import create_video_dataloader
from torch.utils.data import DataLoader, Subset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument(
        "--video-config",
        type=Path,
        default=Path("datasets/video_dataloader/config.yaml"),
    )
    p.add_argument("--max-samples", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save", type=Path, default=Path("outputs/memot_overfit.pt"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clearml", action="store_true")
    p.add_argument("--clearml-project", default="context-detection")
    p.add_argument("--clearml-name", default="MeMOT RF-DETR batch training")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_video_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    config_dir = path.parent.resolve()
    for key in ("videos_dir", "annotations_dir"):
        value = Path(cfg["video"][key])
        if not value.is_absolute():
            cfg["video"][key] = str((config_dir / value).resolve())

    # VideoDataset currently reads this from the root.
    cfg["normalize_boxes"] = cfg["dataset"]["normalize_boxes"]
    return cfg


def make_loader(
    cfg: dict[str, Any],
    max_samples: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:

    base = create_video_dataloader(cfg)

    if len(base.dataset) == 0:
        raise RuntimeError("VideoDataset contains 0 samples.")

    dataset = base.dataset
    if max_samples > 0:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=base.collate_fn,
    )


def build_config(path: Path, clip_len: int):
    return load_config(
        path,
        [
            "data.name=dummy",
            "data.root=null",
            f"data.clip_len={clip_len}",
            "data.context_k=0",
        ],
    )


def empty_target(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
        "labels": torch.empty((0,), dtype=torch.long, device=device),
    }


def detection_metrics(
    pred_boxes: torch.Tensor,
    pred_objectness: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Simple reference-frame diagnostics, not COCO mAP."""
    keep = pred_objectness.sigmoid() >= 0.5
    pred_boxes = pred_boxes[keep]

    if gt_boxes.numel() == 0:
        return {
            "precision": 1.0 if pred_boxes.numel() == 0 else 0.0,
            "recall": 1.0,
            "mean_best_iou": 0.0,
            "pred_count": float(pred_boxes.shape[0]),
            "gt_count": 0.0,
        }

    if pred_boxes.numel() == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "mean_best_iou": 0.0,
            "pred_count": 0.0,
            "gt_count": float(gt_boxes.shape[0]),
        }

    iou = box_iou(pred_boxes, gt_boxes)
    best_pred = iou.max(dim=1).values
    best_gt = iou.max(dim=0).values
    tp = float((best_pred >= iou_threshold).sum())

    return {
        "precision": tp / max(float(pred_boxes.shape[0]), 1.0),
        "recall": float((best_gt >= iou_threshold).sum())
        / max(float(gt_boxes.shape[0]), 1.0),
        "mean_best_iou": float(best_gt.mean()),
        "pred_count": float(pred_boxes.shape[0]),
        "gt_count": float(gt_boxes.shape[0]),
    }


def make_context(
    device: torch.device,
    batch_size: int,
    dim: int = 256,
) -> ContextBatch:
    # ContextBatch contract: valid_mask/time_offsets carry the batch dimension.
    return ContextBatch(
        images=None,
        valid_mask=torch.zeros(
            (batch_size, 1),
            dtype=torch.bool,
            device=device,
        ),
        time_offsets=torch.zeros(
            (batch_size, 1),
            dtype=torch.float32,
            device=device,
        ),
    )



def make_batch(
    images: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    frame_index: int,
    fps: float,
    sequence_start: torch.Tensor,
    sequence_ids: list[str],
    device: torch.device,
) -> DetectionBatch:
    b = images.shape[0]
    return DetectionBatch(
        images=images.to(device, non_blocking=True),
        targets=[
            {
                "boxes": target["boxes"].to(device, non_blocking=True),
                "labels": target["labels"].to(device, non_blocking=True),
            }
            for target in targets
        ],
        sequence_id=sequence_ids,
        frame_id=torch.full((b,), frame_index, dtype=torch.long, device=device),
        timestamp=torch.full(
            (b,), frame_index / max(fps, 1e-6),
            dtype=torch.float32, device=device,
        ),
        is_sequence_start=sequence_start.to(device),
    )


def make_model(cfg) -> RFDETRMeMOT:
    detector = build_registered_detector(cfg)
    num_slots = cfg.context.num_slots

    memory = MeMOTMemory(
        dim=detector.dim,
        num_heads=cfg.detector.num_heads,
        num_slots=num_slots,
        memory_length=cfg.context.memory_length,
        short_memory_length=cfg.context.short_memory_length,
        write_threshold=cfg.context.write_threshold,
        max_missed=cfg.context.max_missed,
        association_iou_threshold=cfg.context.association_iou_threshold,
        association_cosine_threshold=cfg.context.association_cosine_threshold,
        association_appearance_weight=cfg.context.association_appearance_weight,
        motion_momentum=cfg.context.motion_momentum,
    )
    # RF-DETR has a fixed decoder query count. MeMOT appends track queries,
    # so reserve memory slots from the RF-DETR query budget:
    #   proposal_queries + track_slots == detector.num_queries
    # RF-DETR exposes 300 queries in its public/eval config, but in training
    # the adapter uses the full learned query embedding. In this installation
    # that embedding contains 3900 queries. The exact count is determined from
    # detector.initial_queries(batch), so set proposal_count in the train loop
    # once the first real batch is available.

    return RFDETRMeMOT(
        detector,
        memory,
        proposal_count=None,
    )


def fix_rfdetr_boxes(model: RFDETRMeMOT) -> None:
    """Keep upstream RF-DETR predictions compatible with DetectorOutput."""
    upstream = getattr(model.detector, "model", None)
    if upstream is None:
        return

    def hook(_module, _inputs, output):
        if not isinstance(output, dict) or "pred_boxes" not in output:
            return output

        result = dict(output)
        result["pred_boxes"] = result["pred_boxes"].clamp(0.0, 1.0)

        aux = result.get("aux_outputs")
        if aux is not None:
            result["aux_outputs"] = [
                {**layer, "pred_boxes": layer["pred_boxes"].clamp(0.0, 1.0)}
                for layer in aux
            ]
        return result

    upstream.register_forward_hook(hook)


def add_instance_ids(target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    n = target["boxes"].shape[0]
    return {
        "boxes": target["boxes"],
        "labels": target["labels"],
        "instance_ids": torch.arange(
            n, dtype=torch.long, device=target["boxes"].device
        ),
    }


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    def xyxy(boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack(
            [
                cx - w / 2,
                cy - h / 2,
                cx + w / 2,
                cy + h / 2,
            ],
            dim=-1,
        )

    a = xyxy(boxes1)
    b = xyxy(boxes2)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]

    area_a = (a[:, 2] - a[:, 0]).clamp_min(0) * (
        a[:, 3] - a[:, 1]
    ).clamp_min(0)
    area_b = (b[:, 2] - b[:, 0]).clamp_min(0) * (
        b[:, 3] - b[:, 1]
    ).clamp_min(0)

    return inter / (
        area_a[:, None] + area_b[None, :] - inter
    ).clamp_min(1e-6)


def match_active_tracks(
    state,
    gt_boxes: torch.Tensor,
    batch_index: int,
) -> torch.Tensor:
    active = state.memory.valid[batch_index].nonzero(as_tuple=False).flatten()
    if active.numel() == 0 or gt_boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=gt_boxes.device)

    tracks = state.memory.box[batch_index, active]
    iou = box_iou(tracks, gt_boxes)
    best_iou, best_gt = iou.max(dim=1)
    return torch.where(
        best_iou >= 0.3,
        best_gt.to(torch.long),
        torch.full_like(best_gt, -1),
    )


def _slice_detector_output(output: DetectorOutput, index: int) -> DetectorOutput:
    def slice_value(value):
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] > index:
            return value[index:index + 1]
        return value

    return DetectorOutput(
        logits=output.logits[index:index + 1],
        boxes=output.boxes[index:index + 1],
        queries=output.queries[index:index + 1],
        reference_points=output.reference_points[index:index + 1],
        features=[
            feature[index:index + 1]
            if torch.is_tensor(feature) and feature.ndim > 0
            and feature.shape[0] > index else feature
            for feature in output.features
        ],
        decoder_layers=[
            {key: slice_value(value) for key, value in layer.items()}
            for layer in output.decoder_layers
        ],
        aux={key: slice_value(value) for key, value in output.aux.items()},
    )


def slice_memot_output(output: MeMOTOutput, index: int) -> MeMOTOutput:
    return MeMOTOutput(
        detector=_slice_detector_output(output.detector, index),
        proposal=_slice_detector_output(output.proposal, index),
        proposal_boxes=output.proposal_boxes[index:index + 1],
        proposal_objectness=output.proposal_objectness[index:index + 1],
        proposal_uniqueness=output.proposal_uniqueness[index:index + 1],
        track_boxes=output.track_boxes[index:index + 1],
        track_objectness=output.track_objectness[index:index + 1],
        track_uniqueness=output.track_uniqueness[index:index + 1],
        track_ids=output.track_ids[index:index + 1],
        track_slot_indices=output.track_slot_indices[index:index + 1],
        proposal_queries=output.proposal_queries[index:index + 1],
        track_queries=output.track_queries[index:index + 1],
    )


def train(args: argparse.Namespace) -> None:
    clearml_logger = None
    clearml_task = None

    if args.clearml:
        if Task is None:
            raise RuntimeError(
                "ClearML is not installed. Run: pip install clearml"
            )

        Task.set_credentials(
            api_host="http://111.88.249.148:8008",
            web_host="http://111.88.249.148:8080",
            files_host="http://111.88.249.148:8081",
            key="ZI6MUIPPJM6RIMVNIPJDSAN6BOGYWI",
            secret="PAaN4sOYmVblhagbpZjskQycY7WXzFFc2O-nzQEAoD_fedUb3j7W3krPFrV2Tsq5aDs",
            store_conf_file=False,
        )
        clearml_task = Task.init(
            project_name=args.clearml_project,
            task_name=args.clearml_name,
        )
        clearml_logger = clearml_task.get_logger()
        clearml_logger = clearml_task.get_logger()

    seed_everything(args.seed)

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    video_cfg = load_video_config(args.video_config)
    clip_len = int(video_cfg["video"]["frames_before"]) + 1
    fps = float(video_cfg["video"]["fps"])

    loader = make_loader(
        video_cfg,
        args.max_samples,
        args.batch_size,
        args.num_workers,
    )

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    cfg = build_config(args.config, clip_len)

    if cfg.detector.name != "rfdetr":
        raise ValueError(f"Expected rfdetr, got {cfg.detector.name!r}")
    if cfg.context.name != "memot":
        raise ValueError(f"Expected memot, got {cfg.context.name!r}")

    model = make_model(cfg).to(device)
    fix_rfdetr_boxes(model)
    model.train()

    model.detector.freeze(
        backbone=cfg.detector.freeze_backbone,
        decoder=cfg.detector.freeze_decoder,
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp and device.type == "cuda",
    )

    args.save.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Clip length: {clip_len}")
    print(f"Samples used: {len(loader.dataset)}")
    print(
        "Training: 4 history frames -> MeMOT -> reference frame loss"
    )

    for epoch in range(1, args.epochs + 1):
        sums = {
            "loss_total": 0.0,
            "loss_track": 0.0,
            "loss_proposal": 0.0,
            "loss_aux_det": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mean_best_iou": 0.0,
            "pred_count": 0.0,
            "gt_count": 0.0,
        }

        for batch_index, (frames, targets) in enumerate(loader):
            if frames.ndim != 5:
                raise RuntimeError(
                    f"Expected [B,T,C,H,W], got {tuple(frames.shape)}"
                )

            batch_size, time_steps = frames.shape[:2]

            reference_targets = [
                add_instance_ids(
                    {
                        "boxes": target["boxes"].to(device),
                        "labels": target["labels"].to(device),
                    }
                )
                for target in targets
            ]

            sequence_ids = [
                f"sample_{batch_index}_{i}" for i in range(batch_size)
            ]
            state = None
            context = make_context(device, batch_size, dim=model.detector.dim)
            optimizer.zero_grad(set_to_none=True)

            reference_output = None
            state_before_reference = None

            # RF-DETR training uses the actual learned query bank size.
            probe_targets = [empty_target(device) for _ in range(batch_size)]
            probe = make_batch(
                frames[:, 0],
                probe_targets,
                0,
                fps,
                torch.ones(batch_size, dtype=torch.bool),
                sequence_ids,
                device,
            )
            total_queries = model.detector.initial_queries(probe).shape[1]
            model.proposal_count = total_queries - model.memory.num_slots
            if model.proposal_count <= 0:
                raise ValueError(
                    f"RF-DETR query count ({total_queries}) must be greater "
                    f"than MeMOT track slots ({model.memory.num_slots})."
                )

            for frame_index in range(clip_len):
                frame_targets = (
                    reference_targets
                    if frame_index == clip_len - 1
                    else [empty_target(device) for _ in range(batch_size)]
                )

                batch = make_batch(
                    frames[:, frame_index],
                    frame_targets,
                    frame_index,
                    fps,
                    torch.full(
                        (batch_size,),
                        frame_index == 0,
                        dtype=torch.bool,
                    ),
                    sequence_ids,
                    device,
                )

                if frame_index == clip_len - 1:
                    state_before_reference = state

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=args.amp and device.type == "cuda",
                ):
                    reference_output, state = model(batch, context, state)

            if reference_output is None:
                raise RuntimeError("Reference frame did not produce output.")

            # RFDETRMeMOT.loss currently accepts one target at a time because
            # its SetCriterion call is intentionally single-target. The model
            # forward itself is genuinely batched; only the inexpensive scalar
            # loss is evaluated per sample and then averaged.
            sample_losses = []
            for bi in range(batch_size):
                target = reference_targets[bi]

                if state_before_reference is None:
                    track_target_ids = None
                else:
                    track_target_ids = match_active_tracks(
                        state_before_reference,
                        target["boxes"],
                        bi,
                    )

                sample_output = slice_memot_output(reference_output, bi)
                sample_losses.append(
                    model.loss(
                        sample_output,
                        target,
                        seen_ids=set(),
                        track_target_ids=track_target_ids,
                    )
                )

            losses = {
                name: torch.stack(
                    [item[name] for item in sample_losses]
                ).mean()
                for name in sample_losses[0]
            }

            loss = losses["loss_total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, batch {batch_index}"
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            vals = {
                name: float(value.detach().cpu())
                for name, value in losses.items()
            }

            metric_values = {
                "precision": 0.0,
                "recall": 0.0,
                "mean_best_iou": 0.0,
                "pred_count": 0.0,
                "gt_count": 0.0,
            }
            for bi in range(batch_size):
                sample_output = slice_memot_output(reference_output, bi)
                m = detection_metrics(
                    sample_output.proposal_boxes[0].detach(),
                    sample_output.proposal_objectness[0].detach(),
                    reference_targets[bi]["boxes"].detach(),
                )
                for key in metric_values:
                    metric_values[key] += m[key]
            for key in metric_values:
                metric_values[key] /= batch_size

            if clearml_logger is not None:
                step = (epoch - 1) * len(loader) + batch_index
                for name, value in vals.items():
                    clearml_logger.report_scalar(
                        title="loss", series=name,
                        value=value, iteration=step,
                    )
                for name, value in metric_values.items():
                    clearml_logger.report_scalar(
                        title="metrics", series=name,
                        value=float(value), iteration=step,
                    )

            # Extra diagnostics on the reference frame.
            for name in ("loss_total", "loss_track", "loss_proposal", "loss_aux_det"):
                sums[name] += vals[name]
            for name, value in metric_values.items():
                sums[name] += value

            print(
                f"[epoch {epoch}/{args.epochs}] "
                f"batch {batch_index + 1}/{len(loader)} "
                f"B={batch_size} "
                f"loss={vals['loss_total']:.5f} "
                f"track={vals['loss_track']:.5f} "
                f"proposal={vals['loss_proposal']:.5f} "
                f"aux_det={vals['loss_aux_det']:.5f} "
                f"prec={metric_values['precision']:.3f} "
                f"rec={metric_values['recall']:.3f} "
                f"iou={metric_values['mean_best_iou']:.3f}"
            )

        n_batches = len(loader)
        epoch_metrics = {
            name: value / n_batches
            for name, value in sums.items()
        }
        print(
            f"Epoch {epoch}: "
            f"loss={epoch_metrics['loss_total']:.6f} "
            f"track={epoch_metrics['loss_track']:.6f} "
            f"proposal={epoch_metrics['loss_proposal']:.6f} "
            f"aux_det={epoch_metrics['loss_aux_det']:.6f} "
            f"prec={epoch_metrics['precision']:.3f} "
            f"rec={epoch_metrics['recall']:.3f} "
            f"iou={epoch_metrics['mean_best_iou']:.3f}"
        )

        if clearml_logger is not None:
            epoch_step = epoch * len(loader)
            for name, value in epoch_metrics.items():
                clearml_logger.report_scalar(
                    title="epoch",
                    series=name,
                    value=float(value),
                    iteration=epoch_step,
                )

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": epoch_metrics,
            },
            args.save,
        )
        print(f"Checkpoint saved: {args.save}")


def unused_function():
    print(rfdetr.sys)

if __name__ == "__main__":
    train(parse_args())
