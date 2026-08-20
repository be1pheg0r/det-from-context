"""Tests for the generic image dataset, annotation readers, and split modes."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from context_detection.config import ExperimentConfig
from context_detection.data.protocols import DatasetSplit

_COMPONENT_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "image_dataloader"
if str(_COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMPONENT_ROOT))

from annotation_readers import BDD100KJsonReader  # noqa: E402
from dataset import ImageDetectionDataset  # noqa: E402
from provider import ImageDataLoaderProtocol  # noqa: E402
from settings import ImageDataLoaderSettings  # noqa: E402


def _settings(
    images_dir: Path,
    annotations_dir: Path,
    *,
    split_mode: str = "generated",
    test_fraction: float = 0.0,
    strict_pairs: bool = True,
) -> ImageDataLoaderSettings:
    """Build a compact valid component config for tests."""
    validation_fraction = 0.2 if test_fraction == 0.0 else 0.2
    return ImageDataLoaderSettings.model_validate(
        {
            "name": "image_dataloader",
            "dataset": {
                "images_dir": str(images_dir),
                "annotations_dir": str(annotations_dir),
                "annotation_format": "bdd100k_json",
                "image_size": {"width": 32, "height": 32},
                "patch_size": 16,
                "num_windows": 2,
                "image_extensions": [".png"],
                "strict_pairs": strict_pairs,
            },
            "splits": {
                "mode": split_mode,
                "train_fraction": 1.0 - validation_fraction - test_fraction,
                "validation_fraction": validation_fraction,
                "test_fraction": test_fraction,
            },
            "dataloader": {"batch_size": 1, "num_workers": 0},
            "classes": {"car": 0, "person": 1},
        }
    )


def _write_pair(root: Path, stem: str, objects: list[dict[str, Any]]) -> None:
    """Write one 20x10 RGB image and its BDD100K-style JSON file."""
    images = root / "images"
    annotations = root / "annotations"
    images.mkdir(parents=True, exist_ok=True)
    annotations.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 10), color=(40, 80, 120)).save(images / f"{stem}.png")
    (annotations / f"{stem}.json").write_text(
        json.dumps({"frames": [{"objects": objects}]}),
        encoding="utf-8",
    )


def _normalize_transform(
    image: Image.Image, target: dict[str, Tensor]
) -> tuple[Tensor, dict[str, Tensor]]:
    """Small deterministic stand-in for RF-DETR's normalized cxcywh transform."""
    width, height = image.size
    boxes = target["boxes"].clone()
    if boxes.numel():
        x1, y1, x2, y2 = boxes.unbind(-1)
        boxes = torch.stack(
            (
                (x1 + x2) / (2 * width),
                (y1 + y2) / (2 * height),
                (x2 - x1) / width,
                (y2 - y1) / height,
            ),
            dim=-1,
        )
    transformed = dict(target)
    transformed["boxes"] = boxes
    transformed["size"] = torch.tensor([height, width])
    return torch.zeros(3, height, width), transformed


def test_bdd100k_reader_clamps_boxes_and_counts_rejected_objects(
    tmp_path: Path,
) -> None:
    _write_pair(
        tmp_path,
        "frame",
        [
            {"category": "car", "box2d": {"x1": -2, "y1": 1, "x2": 25, "y2": 9}},
            {"category": "unknown", "box2d": {"x1": 1, "y1": 1, "x2": 2, "y2": 2}},
            {"category": "person", "box2d": {"x1": 5, "y1": 5, "x2": 4, "y2": 6}},
        ],
    )

    parsed = BDD100KJsonReader().read(
        tmp_path / "annotations/frame.json", {"car": 0, "person": 1}, (20, 10)
    )

    assert parsed.boxes == [[0.0, 1.0, 20, 9.0]]
    assert parsed.labels == [0]
    assert parsed.areas == [160.0]
    assert parsed.rejected_objects == 2


def test_dataset_pairs_sorted_files_and_validates_transformed_targets(
    tmp_path: Path,
) -> None:
    box = {"category": "car", "box2d": {"x1": 2, "y1": 1, "x2": 10, "y2": 7}}
    _write_pair(tmp_path, "b", [box])
    _write_pair(tmp_path, "a", [box])
    settings = _settings(tmp_path / "images", tmp_path / "annotations")

    dataset = ImageDetectionDataset(
        tmp_path / "images",
        tmp_path / "annotations",
        settings,
        image_set="train",
        transform=_normalize_transform,
    )
    image, target = dataset[0]

    assert [Path(item["image"]).stem for item in dataset.manifest()] == ["a", "b"]
    assert image.shape == (3, 10, 20)
    assert target["boxes"].shape == (1, 4)
    assert target["boxes"].min() >= 0 and target["boxes"].max() <= 1
    assert target["rejected_objects"].item() == 0


def test_dataset_strict_pair_validation_reports_missing_and_orphan_files(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    Image.new("RGB", (8, 8)).save(images / "missing.png")
    settings = _settings(images, annotations)

    with pytest.raises(ValueError, match="missing annotations"):
        ImageDetectionDataset(
            images,
            annotations,
            settings,
            image_set="val",
            transform=_normalize_transform,
        )

    (images / "missing.png").unlink()
    (annotations / "orphan.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="orphan annotations"):
        ImageDetectionDataset(
            images,
            annotations,
            settings,
            image_set="val",
            transform=_normalize_transform,
        )


def test_generated_splits_are_deterministic_disjoint_and_include_test(
    tmp_path: Path,
) -> None:
    settings = _settings(
        tmp_path / "images",
        tmp_path / "annotations",
        test_fraction=0.2,
    )
    protocol = ImageDataLoaderProtocol()

    first = {
        split: protocol._split_indices(settings, 20, split, seed=42)
        for split in DatasetSplit
    }
    second = {
        split: protocol._split_indices(settings, 20, split, seed=42)
        for split in DatasetSplit
    }

    assert first == second
    assert {split: len(values) for split, values in first.items()} == {
        DatasetSplit.TRAIN: 12,
        DatasetSplit.VALIDATION: 4,
        DatasetSplit.TEST: 4,
    }
    assert set(first[DatasetSplit.TRAIN]).isdisjoint(first[DatasetSplit.VALIDATION])
    assert set(first[DatasetSplit.TRAIN]).isdisjoint(first[DatasetSplit.TEST])
    assert set(first[DatasetSplit.VALIDATION]).isdisjoint(first[DatasetSplit.TEST])
    assert set().union(*map(set, first.values())) == set(range(20))


def test_generated_test_split_with_zero_fraction_fails_explicitly(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "images", tmp_path / "annotations")

    with pytest.raises(ValueError, match="zero configured fraction"):
        ImageDataLoaderProtocol._split_indices(settings, 10, DatasetSplit.TEST, seed=0)


def test_predefined_mode_selects_independent_train_val_and_test_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    for split_name in ("train", "val", "test"):
        (images / split_name).mkdir(parents=True)
        (annotations / split_name).mkdir(parents=True)
    settings = _settings(images, annotations, split_mode="predefined")
    config_path = tmp_path / "dataset.yaml"
    OmegaConf.save(settings.model_dump(mode="json"), config_path)
    experiment = ExperimentConfig.model_validate(
        {
            "data": {
                "name": "image_dataloader",
                "component_path": str(_COMPONENT_ROOT),
                "config_path": str(config_path),
                "context_k": 0,
                "context_strategy": "empty",
                "clip_len": 1,
                "image_size": 32,
            },
            "train": {"batch_size": 1, "num_workers": 0},
            "validation": {"batch_size": 1},
        }
    )
    calls: list[tuple[Path, Path, str]] = []

    class _FakeDataset(Dataset[tuple[Tensor, dict[str, Tensor]]]):
        def __init__(
            self,
            images_dir: Path,
            annotations_dir: Path,
            component_settings: ImageDataLoaderSettings,
            *,
            image_set: str,
        ) -> None:
            del component_settings
            calls.append((Path(images_dir), Path(annotations_dir), image_set))

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
            return torch.zeros(3, 32, 32), {
                "boxes": torch.zeros(0, 4),
                "labels": torch.zeros(0, dtype=torch.long),
                "image_id": torch.tensor(index),
            }

    monkeypatch.setattr("provider.ImageDetectionDataset", _FakeDataset)
    protocol = ImageDataLoaderProtocol()
    for split in DatasetSplit:
        protocol.build(experiment, split)

    assert calls == [
        (images / "train", annotations / "train", "train"),
        (images / "val", annotations / "val", "val"),
        (images / "test", annotations / "test", "test"),
    ]
