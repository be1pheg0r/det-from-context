"""Thin adapters from project components to RF-DETR's official Lightning stack."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import LightningDataModule
from rfdetr.config import TrainConfig as RFDetrTrainConfig
from rfdetr.training import RFDETRModelModule
from rfdetr.utilities.tensors import NestedTensor, nested_tensor_from_tensor_list
from torch import Tensor
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..contracts import (
    ContextBatch,
    DetectionBatch,
    DetectionClipBatch,
    DetectorOutput,
)
from ..tracking import tracking_metrics, tracking_output_to_predictions
from .memory import MeMOTState
from .memot import MeMOTTracker
from .rfdetr import RFDetrAdapter


class ComponentRFDetrModule(RFDETRModelModule):
    """Run the component-built detector through RF-DETR's native train lifecycle.

    RF-DETR 1.9.0 does not accept a pre-built model in ``RFDETRModelModule``.
    The superclass is therefore initialized with a no-checkpoint template and
    its temporary CPU model is immediately replaced with the official model
    already created by :class:`RFDetrAdapter`. All training_step, validation,
    criterion, optimizer, scheduler, AMP, EMA, and checkpoint logic remains
    upstream code.
    """

    def __init__(
        self,
        detector: RFDetrAdapter,
        train_config: RFDetrTrainConfig,
    ) -> None:
        model_config = detector.model_config
        if bool(getattr(model_config, "compile", False)):
            raise ValueError(
                "component RF-DETR training does not support model.compile=true; "
                "compile the component model before injection instead"
            )
        if bool(getattr(model_config, "backbone_lora", False)):
            raise ValueError(
                "component RF-DETR training does not support backbone_lora=true"
            )
        template = model_config.model_copy(deep=True)
        template.pretrain_weights = None
        template.compile = False
        template.backbone_lora = False
        super().__init__(template, train_config)
        self.model = detector.model
        self.model_config = model_config


class ComponentRFDetrMeMOTModule(ComponentRFDetrModule):
    """Train external MeMOT on clips while retaining RF-DETR's native stack.

    The upstream criterion, matcher, optimizer grouping, scheduler, AMP, EMA,
    and checkpoint callbacks remain owned by ``RFDETRModelModule``. Only the
    sequential clip step and the two MeMOT-specific losses are added here.
    """

    def __init__(
        self,
        tracker: MeMOTTracker,
        train_config: RFDetrTrainConfig,
        config: ExperimentConfig,
    ) -> None:
        if not isinstance(tracker.detector, RFDetrAdapter):
            raise TypeError("RF-DETR + MeMOT training requires RFDetrAdapter")
        super().__init__(tracker.detector, train_config)
        self.config = config
        # Register MeMOT below the official model so its optimizer, EMA and
        # checkpoints see the added parameters without replacing upstream code.
        self.model.add_module("project_memot_encoder", tracker.memory_encoder)
        self.model.add_module("project_memot_decoder", tracker.memory_decoder)
        # Avoid registering the detector/model a second time on the Lightning
        # module; the tracker references the exact modules attached above.
        self.__dict__["_tracker"] = tracker
        self.validation_tracking_result: dict[str, float | str] = {}
        self._validation_tracking_predictions: list[dict[str, Any]] = []
        self._validation_tracking_targets: list[dict[str, Any]] = []
        self._validation_annotation_modes: set[str] = set()

    @property
    def tracker(self) -> MeMOTTracker:
        """Return the non-owning orchestration wrapper."""
        tracker = self.__dict__.get("_tracker")
        if not isinstance(tracker, MeMOTTracker):
            raise RuntimeError("MeMOT tracker was not initialized")
        return tracker

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: torch.device,
        dataloader_idx: int,
    ) -> Any:
        """Move Pydantic clip contracts, which Lightning cannot traverse."""
        if isinstance(batch, DetectionClipBatch):
            return _clip_to_device(batch, device)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def training_step(
        self, batch: DetectionClipBatch, batch_idx: int
    ) -> Tensor | dict[str, Any]:
        """Accumulate detection and association supervision over one clip."""
        result = self._clip_step(batch, stage="train")
        loss = result["loss"]
        try:
            accumulation = max(1, int(self.trainer.accumulate_grad_batches))
        except RuntimeError:
            accumulation = 1
        returned_loss = loss / accumulation
        if self.train_config.compute_train_metrics:
            return {
                **result,
                "loss": returned_loss,
            }
        return returned_loss

    def validation_step(
        self, batch: DetectionClipBatch, batch_idx: int
    ) -> dict[str, Any]:
        """Evaluate the last supervised frame and log clip-level losses."""
        del batch_idx
        result = self._clip_step(batch, stage="val")
        self._validation_tracking_predictions.extend(result["tracking_predictions"])
        self._validation_tracking_targets.extend(result["tracking_targets"])
        self._validation_annotation_modes.add(batch.mode)
        return result

    def on_validation_epoch_start(self) -> None:
        """Reset sequence records before accumulating a validation epoch."""
        self._validation_tracking_predictions = []
        self._validation_tracking_targets = []
        self._validation_annotation_modes = set()

    def on_validation_epoch_end(self) -> None:
        """Log true tracking metrics or preserve an explicit unavailable reason."""
        mode = (
            next(iter(self._validation_annotation_modes))
            if len(self._validation_annotation_modes) == 1
            else "mixed"
        )
        result = tracking_metrics(
            self._validation_tracking_predictions,
            self._validation_tracking_targets,
            annotation_mode=mode,
        )
        self.validation_tracking_result = result.as_dict()
        if result.available:
            self.log_dict(
                {
                    f"val/tracking_{name}": value
                    for name, value in result.metrics.items()
                },
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

    def test_step(self, batch: DetectionClipBatch, batch_idx: int) -> dict[str, Any]:
        """Reuse validation semantics for a fixed held-out tracking split."""
        del batch_idx
        return self._clip_step(batch, stage="test")

    def _clip_step(self, batch: DetectionClipBatch, *, stage: str) -> dict[str, Any]:
        if not isinstance(batch, DetectionClipBatch):
            raise TypeError("RF-DETR + MeMOT requires DetectionClipBatch")
        self.tracker.train(self.training)
        state: MeMOTState | None = None
        gt_slots: list[dict[int, int]] = [dict() for _ in range(batch.batch_size)]
        totals: dict[str, Tensor] = {}
        supervised_steps = 0
        last_output: DetectorOutput | None = None
        last_targets: list[dict[str, Tensor]] | None = None
        tracking_predictions: list[dict[str, Any]] = []
        tracking_targets: list[dict[str, Any]] = []

        for step_index, (detection, context) in enumerate(batch.steps):
            prior_state = state
            output, state = self.tracker(detection, context, state)
            supervision = batch.supervision_mask[step_index]
            if not bool(supervision.any()):
                continue
            if not bool(supervision.all()):
                raise ValueError(
                    "mixed supervised/unsupervised samples within one clip step "
                    "are not supported"
                )
            upstream = output.aux.get("upstream_outputs")
            if not isinstance(upstream, Mapping):
                raise TypeError("RF-DETR adapter did not preserve upstream outputs")
            targets = detection.targets
            loss_dict = self.criterion(upstream, targets)
            weighted_detection = torch.stack(
                [
                    value * self.criterion.weight_dict[name]
                    for name, value in loss_dict.items()
                    if name in self.criterion.weight_dict
                ]
            ).sum()
            _accumulate_loss(totals, "loss_detection", weighted_detection)
            for name, value in loss_dict.items():
                _accumulate_loss(totals, f"rfdetr_{name}", value)

            if batch.mode == "tracking":
                association, uniqueness, indices = self._tracking_losses(
                    output,
                    upstream,
                    targets,
                    prior_state,
                    gt_slots,
                )
                _accumulate_loss(totals, "loss_association", association)
                _accumulate_loss(totals, "loss_uniqueness", uniqueness)
                self._update_gt_slots(output, targets, indices, gt_slots)
            supervised_steps += 1
            last_output = output
            last_targets = targets
            tracking_predictions.extend(
                tracking_output_to_predictions(
                    output,
                    detection,
                    score_threshold=self.config.logging.prediction_score_threshold,
                )
            )
            tracking_targets.extend(_tracking_targets(detection))

        if not supervised_steps or last_output is None or last_targets is None:
            raise ValueError("video clip contains no supervised frame")
        averaged = {name: value / supervised_steps for name, value in totals.items()}
        association = averaged.get("loss_association", last_output.logits.new_zeros(()))
        uniqueness = averaged.get("loss_uniqueness", last_output.logits.new_zeros(()))
        loss = (
            averaged["loss_detection"]
            + self.config.context.association_loss_weight * association
            + self.config.context.uniqueness_loss_weight * uniqueness
        )
        self._log_clip_losses(stage, loss, averaged, batch.batch_size)
        orig_sizes = torch.stack([target["orig_size"] for target in last_targets])
        results = self.postprocess(
            {
                "pred_logits": last_output.logits,
                "pred_boxes": last_output.boxes,
            },
            orig_sizes,
        )
        return {
            "loss": loss,
            "results": results,
            "targets": last_targets,
            "tracking_predictions": tracking_predictions,
            "tracking_targets": tracking_targets,
        }

    def _tracking_losses(
        self,
        output: DetectorOutput,
        upstream: Mapping[str, Any],
        targets: list[dict[str, Tensor]],
        prior_state: MeMOTState | None,
        gt_slots: list[dict[int, int]],
    ) -> tuple[Tensor, Tensor, list[tuple[Tensor, Tensor]]]:
        memot = output.aux.get("memot")
        if not isinstance(memot, dict):
            raise TypeError("MeMOT output diagnostics are missing")
        association_logits = memot.get("association_logits")
        if not isinstance(association_logits, Tensor):
            raise TypeError("MeMOT association_logits are missing")
        proposal_count = association_logits.shape[1]
        matcher_outputs = {
            "pred_logits": upstream["pred_logits"][:, :proposal_count],
            "pred_boxes": upstream["pred_boxes"][:, :proposal_count],
        }
        indices = self.criterion.matcher(matcher_outputs, targets)
        association_terms: list[Tensor] = []
        new_track_class = association_logits.shape[-1] - 1
        for batch_index, (proposal_indices, target_indices) in enumerate(indices):
            track_ids = targets[batch_index].get("track_ids")
            if track_ids is None:
                continue
            selected_ids = track_ids[target_indices]
            valid = selected_ids >= 0
            if not bool(valid.any()):
                continue
            proposal_indices = proposal_indices[valid]
            selected_ids = selected_ids[valid]
            labels = torch.full_like(selected_ids, new_track_class)
            for index, track_id in enumerate(selected_ids.tolist()):
                slot = gt_slots[batch_index].get(int(track_id))
                if slot is not None and (
                    prior_state is None or bool(prior_state.valid[batch_index, slot])
                ):
                    labels[index] = slot
            association_terms.append(
                torch.nn.functional.cross_entropy(
                    association_logits[batch_index, proposal_indices], labels
                )
            )
        association = (
            torch.stack(association_terms).mean()
            if association_terms
            else association_logits.sum() * 0.0
        )
        existing_probability = association_logits.softmax(dim=-1)[..., :-1]
        occupancy = existing_probability.sum(dim=1)
        uniqueness = torch.relu(occupancy - 1.0).square().mean()
        return association, uniqueness, indices

    @staticmethod
    def _update_gt_slots(
        output: DetectorOutput,
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        gt_slots: list[dict[int, int]],
    ) -> None:
        slots = output.aux["memot"].get("slot_indices")
        if not isinstance(slots, Tensor):
            raise TypeError("MeMOT slot_indices are missing after memory update")
        for batch_index, (proposal_indices, target_indices) in enumerate(indices):
            track_ids = targets[batch_index].get("track_ids")
            if track_ids is None:
                continue
            for proposal, target in zip(
                proposal_indices.tolist(), target_indices.tolist(), strict=True
            ):
                track_id = int(track_ids[target])
                slot = int(slots[batch_index, proposal])
                if track_id >= 0 and slot >= 0:
                    gt_slots[batch_index][track_id] = slot

    def _log_clip_losses(
        self,
        stage: str,
        loss: Tensor,
        components: dict[str, Tensor],
        batch_size: int,
    ) -> None:
        on_step = stage == "train" and bool(self.train_config.train_log_on_step)
        sync_dist = stage != "train" or bool(self.train_config.train_log_sync_dist)
        self.log_dict(
            {f"{stage}/{name}": value for name, value in components.items()},
            on_step=on_step,
            on_epoch=True,
            sync_dist=sync_dist,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=on_step,
            on_epoch=True,
            sync_dist=sync_dist,
            batch_size=batch_size,
        )


class ProjectRFDetrDataModule(LightningDataModule):
    """Expose project ``DataLoader`` endpoints to RF-DETR without COCO export."""

    def __init__(
        self,
        dataloaders: Mapping[str, DataLoader[Any]],
        *,
        block_size: int,
        class_names: list[str],
    ) -> None:
        super().__init__()
        if block_size <= 0:
            raise ValueError("RF-DETR block_size must be positive")
        self._dataloaders = {
            "validation" if name == "val" else name: loader
            for name, loader in dataloaders.items()
        }
        for required in ("train", "validation"):
            if required not in self._dataloaders:
                raise ValueError(f"RF-DETR DataModule requires split {required!r}")
        self.block_size = block_size
        self.class_names = list(class_names)
        self._dataset_train = self._dataloaders["train"].dataset
        self._dataset_val = self._dataloaders["validation"].dataset
        test_loader = self._dataloaders.get("test")
        self._dataset_test = test_loader.dataset if test_loader is not None else None

    def train_dataloader(self) -> DataLoader[Any]:
        """Return the project training loader unchanged."""
        return self._dataloaders["train"]

    def val_dataloader(self) -> DataLoader[Any]:
        """Return the independent project validation loader."""
        return self._dataloaders["validation"]

    def test_dataloader(self) -> DataLoader[Any]:
        """Return the fixed test loader when the dataset declares one."""
        try:
            return self._dataloaders["test"]
        except KeyError as error:
            raise RuntimeError("RF-DETR test split was not configured") from error

    def predict_dataloader(self) -> DataLoader[Any]:
        """Use validation samples for prediction diagnostics."""
        return self.val_dataloader()

    def on_before_batch_transfer(
        self,
        batch: DetectionClipBatch | tuple[DetectionBatch, ContextBatch] | list[Any],
        dataloader_idx: int,
    ) -> DetectionClipBatch | tuple[NestedTensor, list[dict[str, Any]]]:
        """Convert project contracts to RF-DETR's ``NestedTensor, targets`` batch.

        PyTorch's pin-memory walker converts plain tuples to lists for backwards
        compatibility. GPU DataLoaders therefore reach this hook as a list even
        though :class:`DetectionCollator` returns a tuple on the CPU path.
        """
        del dataloader_idx
        if isinstance(batch, DetectionClipBatch):
            return batch
        if not isinstance(batch, tuple | list) or len(batch) != 2:
            raise TypeError(
                "project RF-DETR loader must return (DetectionBatch, ContextBatch)"
            )
        detection, context = batch
        if not isinstance(detection, DetectionBatch) or not isinstance(
            context, ContextBatch
        ):
            raise TypeError(
                "project RF-DETR loader returned incompatible batch contracts"
            )
        if context.valid_mask.numel() and bool(context.valid_mask.any()):
            raise ValueError(
                "native RF-DETR image training does not accept context frames; "
                "set data.context_k=0"
            )
        samples = nested_tensor_from_tensor_list(
            list(detection.images.unbind(0)), block_size=self.block_size
        )
        targets = [dict(target) for target in detection.targets]
        return samples, targets


def _accumulate_loss(totals: dict[str, Tensor], name: str, value: Tensor) -> None:
    totals[name] = totals.get(name, value.new_zeros(())) + value


def _clip_to_device(
    batch: DetectionClipBatch, device: torch.device
) -> DetectionClipBatch:
    """Rebuild a clip contract with every model-facing tensor on one device."""
    steps: list[tuple[DetectionBatch, ContextBatch]] = []
    for detection, context in batch.steps:
        moved_targets = [
            {name: value.to(device) for name, value in target.items()}
            for target in detection.targets
        ]
        moved_detection = DetectionBatch(
            images=detection.images.to(device),
            targets=moved_targets,
            sequence_id=detection.sequence_id,
            frame_id=detection.frame_id.to(device),
            timestamp=detection.timestamp.to(device),
            is_sequence_start=(
                None
                if detection.is_sequence_start is None
                else detection.is_sequence_start.to(device)
            ),
        )
        moved_context = ContextBatch(
            images=None if context.images is None else context.images.to(device),
            valid_mask=context.valid_mask.to(device),
            time_offsets=context.time_offsets.to(device),
            targets=(
                None
                if context.targets is None
                else [
                    [
                        {name: value.to(device) for name, value in target.items()}
                        for target in sample
                    ]
                    for sample in context.targets
                ]
            ),
            extras=context.extras,
        )
        steps.append((moved_detection, moved_context))
    return DetectionClipBatch(
        steps=steps,
        supervision_mask=batch.supervision_mask.to(device),
        mode=batch.mode,
    )


def _tracking_targets(batch: DetectionBatch) -> list[dict[str, Any]]:
    """Attach sequence/frame identity to target dictionaries for MOT metrics."""
    return [
        {
            **{name: value.detach().cpu() for name, value in target.items()},
            "sequence_id": batch.sequence_id[index],
            "frame_id": int(batch.frame_id[index]),
        }
        for index, target in enumerate(batch.targets)
    ]


def build_rfdetr_train_config(
    config: ExperimentConfig,
    output_dir: str | Path,
    *,
    class_names: list[str],
    has_test_split: bool,
) -> RFDetrTrainConfig:
    """Map the project experiment schema onto RF-DETR 1.9.0 TrainConfig."""
    return RFDetrTrainConfig(
        dataset_dir=None,
        output_dir=str(output_dir),
        lr=config.train.lr,
        lr_encoder=config.train.lr * config.train.backbone_lr_multiplier,
        lr_component_decay=config.train.decoder_lr_multiplier,
        batch_size=config.train.batch_size,
        grad_accum_steps=config.train.grad_accum,
        epochs=config.train.epochs,
        checkpoint_interval=config.output.checkpoint_every_n_epochs,
        skip_best_epochs=0,
        warmup_epochs=float(config.train.warmup_epochs),
        weight_decay=config.train.weight_decay,
        num_workers=config.train.num_workers,
        seed=config.train.seed,
        optimizer=config.train.optimizer.value,
        use_ema=True,
        multi_scale=False,
        expanded_scales=False,
        tensorboard=False,
        wandb=False,
        mlflow=False,
        clearml=False,
        project=config.clearml.project_name,
        run=config.name,
        class_names=class_names,
        run_test=has_test_split,
        eval_interval=config.validation.every_n_epochs,
        log_per_class_metrics=True,
        progress_bar="tqdm",
        train_log_on_step=True,
        compute_train_metrics=True,
        compute_val_loss=True,
        compute_test_loss=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.train.num_workers > 0,
    )
