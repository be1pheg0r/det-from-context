"""Contract tests for the project-to-upstream RF-DETR training bridge."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from experiments.rfdetr_image.run import _configured_splits
from experiments.rfdetr_image.worker import RFDetrImageExperiment, _loader_manifest
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule, Trainer
from rfdetr.training import RFDETRModelModule
from rfdetr.utilities.tensors import NestedTensor
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

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


def test_project_datamodule_accepts_pin_memory_list_batch() -> None:
    loader = _loader()
    datamodule = ProjectRFDetrDataModule(
        {"train": loader, "validation": loader},
        block_size=16,
        class_names=["car"],
    )
    pinned_shape = list(next(iter(loader)))

    nested, targets = datamodule.on_before_batch_transfer(pinned_shape, 0)

    assert isinstance(nested, NestedTensor)
    assert len(targets) == 2


class _ValidationModule(LightningModule):
    """Exercise Lightning's real CombinedLoader batch wrapping."""

    def validation_step(
        self,
        batch: tuple[NestedTensor, list[dict[str, Any]]],
        batch_idx: int,
    ) -> None:
        del batch_idx
        nested, targets = batch
        assert isinstance(nested, NestedTensor)
        assert len(targets) == 2


def test_project_datamodule_accepts_lightning_combined_loader_batch() -> None:
    loader = _loader()
    datamodule = ProjectRFDetrDataModule(
        {"train": loader, "validation": loader},
        block_size=16,
        class_names=["car"],
    )
    trainer = Trainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        limit_val_batches=1,
    )

    trainer.validate(_ValidationModule(), datamodule=datamodule)


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


class _ManifestSource(Dataset[dict[str, Any]]):
    """Expose deterministic provenance without loading real image files."""

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"index": index}

    @staticmethod
    def manifest() -> list[dict[str, str]]:
        return [
            {"image": f"image-{index}.jpg", "annotation": f"image-{index}.json"}
            for index in range(3)
        ]


def test_loader_manifest_preserves_subset_order() -> None:
    wrapped = SimpleNamespace(dataset=_ManifestSource())
    loader = SimpleNamespace(dataset=Subset(wrapped, [2, 0]))

    assert _loader_manifest(loader) == [
        {"image": "image-2.jpg", "annotation": "image-2.json"},
        {"image": "image-0.jpg", "annotation": "image-0.json"},
    ]


@pytest.mark.parametrize(
    ("split_config", "expected"),
    [
        (
            {"mode": "generated", "test_fraction": 0.0},
            ("train", "validation"),
        ),
        (
            {"mode": "generated", "test_fraction": 0.1},
            ("train", "validation", "test"),
        ),
        (
            {"mode": "predefined", "test_fraction": 0.0},
            ("train", "validation", "test"),
        ),
    ],
)
def test_configured_splits_include_declared_test(
    tmp_path: Path,
    split_config: dict[str, Any],
    expected: tuple[str, ...],
) -> None:
    config_path = tmp_path / "dataset.yaml"
    OmegaConf.save({"splits": split_config}, config_path)

    assert _configured_splits(config_path) == expected


def test_checkpoint_publication_applies_periodic_retention(tmp_path: Path) -> None:
    for name in ("checkpoint_0.ckpt", "checkpoint_1.ckpt", "checkpoint_2.ckpt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    (tmp_path / "best.ckpt").write_text("best", encoding="utf-8")
    published: list[str] = []
    metadata: dict[str, Any] = {}
    experiment = SimpleNamespace(
        checkpoints_dir=tmp_path,
        save_artifact=lambda name, source: published.append(name),
        record_metadata=lambda name, value: metadata.__setitem__(name, value),
    )
    runner = RFDetrImageExperiment.__new__(RFDetrImageExperiment)
    runner.experiment = experiment
    runner.config = ExperimentConfig(
        output={"keep_last_checkpoints": 2},
    )

    runner._publish_checkpoints()

    assert not (tmp_path / "checkpoint_0.ckpt").exists()
    assert (tmp_path / "checkpoint_1.ckpt").is_file()
    assert (tmp_path / "checkpoint_2.ckpt").is_file()
    assert published == [
        "checkpoint__best.ckpt",
        "checkpoint__checkpoint_1.ckpt",
        "checkpoint__checkpoint_2.ckpt",
    ]
    assert metadata["checkpoints"] == published
