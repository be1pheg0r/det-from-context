"""Contract tests for the project-to-upstream RF-DETR training bridge."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import torch
from pytorch_lightning import LightningModule
from rfdetr.training import RFDETRModelModule
from rfdetr.utilities.tensors import NestedTensor
from torch import nn
from torch.utils.data import DataLoader, Dataset

from context_detection.config import ExperimentConfig
from context_detection.data.collate import DetectionCollator
from context_detection.models.rfdetr_training import (
    ComponentRFDetrModule,
    ProjectRFDetrDataModule,
    build_rfdetr_train_config,
)


class _SampleDataset(Dataset[dict[str, Any]]):
    """Small project-contract dataset used without external images or weights."""

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": torch.full((3, 16, 16), float(index)),
            "target": {
                "boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
                "labels": torch.tensor([0]),
                "image_id": torch.tensor(index),
                "orig_size": torch.tensor([16, 16]),
                "size": torch.tensor([16, 16]),
            },
            "sequence_id": "images",
            "frame_id": index,
            "timestamp": float(index),
            "is_sequence_start": True,
            "context_valid_mask": torch.zeros(0, dtype=torch.bool),
            "context_time_offsets": torch.zeros(0),
        }


def _loader() -> DataLoader[Any]:
    return DataLoader(_SampleDataset(), batch_size=2, collate_fn=DetectionCollator())


def _config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "rfdetr-native-training-test",
            "data": {
                "name": "dummy",
                "context_k": 0,
                "context_strategy": "empty",
                "clip_len": 1,
                "image_size": 32,
            },
            "detector": {
                "name": "dummy",
                "dim": 8,
                "num_heads": 2,
                "num_classes": 2,
            },
            "train": {
                "epochs": 3,
                "lr": 0.001,
                "batch_size": 2,
                "grad_accum": 2,
                "warmup_epochs": 1,
                "backbone_lr_multiplier": 0.1,
                "decoder_lr_multiplier": 0.5,
                "num_workers": 0,
            },
            "validation": {"every_n_epochs": 2, "batch_size": 2},
            "output": {"checkpoint_every_n_epochs": 2},
        }
    )


def test_project_datamodule_converts_batches_to_native_nested_tensor() -> None:
    loader = _loader()
    datamodule = ProjectRFDetrDataModule(
        {"train": loader, "validation": loader},
        block_size=32,
        class_names=["car"],
    )

    nested, targets = datamodule.on_before_batch_transfer(next(iter(loader)), 0)

    assert isinstance(nested, NestedTensor)
    assert nested.tensors.shape == (2, 3, 32, 32)
    assert nested.mask is not None and nested.mask.shape == (2, 32, 32)
    assert len(targets) == 2
    assert targets[0]["boxes"].shape == (1, 4)
    assert datamodule.class_names == ["car"]


def test_project_datamodule_rejects_valid_context_frames() -> None:
    loader = _loader()
    datamodule = ProjectRFDetrDataModule(
        {"train": loader, "validation": loader},
        block_size=16,
        class_names=["car"],
    )
    detection, context = next(iter(loader))
    context.valid_mask = torch.ones(2, 1, dtype=torch.bool)

    with pytest.raises(ValueError, match="context frames"):
        datamodule.on_before_batch_transfer((detection, context), 0)


def test_project_config_maps_to_official_train_config(tmp_path: Any) -> None:
    config = _config()

    upstream = build_rfdetr_train_config(
        config,
        tmp_path,
        class_names=["car", "person"],
        has_test_split=True,
    )

    assert upstream.dataset_dir is None
    assert upstream.epochs == 3
    assert upstream.grad_accum_steps == 2
    assert upstream.lr_encoder == pytest.approx(0.0001)
    assert upstream.lr_component_decay == 0.5
    assert upstream.checkpoint_interval == 2
    assert upstream.eval_interval == 2
    assert upstream.class_names == ["car", "person"]
    assert upstream.run_test is True
    assert upstream.clearml is False
    assert upstream.tensorboard is False


class _ModelConfig:
    compile = False
    backbone_lora = False
    pretrain_weights = "must-not-be-loaded.pth"

    def model_copy(self, *, deep: bool) -> _ModelConfig:
        assert deep is True
        return deepcopy(self)


def test_training_module_injects_component_model_without_loading_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component_model = nn.Linear(2, 2)
    detector = type(
        "Detector",
        (),
        {"model": component_model, "model_config": _ModelConfig()},
    )()
    seen: dict[str, Any] = {}

    def fake_upstream_init(
        self: RFDETRModelModule,
        model_config: _ModelConfig,
        train_config: Any,
    ) -> None:
        LightningModule.__init__(self)
        seen["pretrain_weights"] = model_config.pretrain_weights
        self.model = nn.Linear(1, 1)
        self.model_config = model_config
        self.train_config = train_config

    monkeypatch.setattr(RFDETRModelModule, "__init__", fake_upstream_init)
    upstream_config = build_rfdetr_train_config(
        _config(), ".", class_names=["car"], has_test_split=False
    )

    module = ComponentRFDetrModule(detector, upstream_config)  # type: ignore[arg-type]

    assert seen["pretrain_weights"] is None
    assert module.model is component_model
    assert module.model_config is detector.model_config
