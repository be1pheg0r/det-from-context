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
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..contracts import ContextBatch, DetectionBatch
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
        batch: tuple[DetectionBatch, ContextBatch],
        dataloader_idx: int,
    ) -> tuple[NestedTensor, list[dict[str, Any]]]:
        """Convert project contracts to RF-DETR's ``NestedTensor, targets`` batch."""
        del dataloader_idx
        if not isinstance(batch, tuple) or len(batch) != 2:
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
