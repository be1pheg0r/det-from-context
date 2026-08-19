"""Protocol adapter for the existing BDD100K-style image dataloader."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset

from context_detection.config import ExperimentConfig
from context_detection.data.collate import DetectionCollator
from context_detection.data.protocols import DatasetSplit

_COMPONENT_ROOT = Path(__file__).resolve().parent
if str(_COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMPONENT_ROOT))

from dataset import BDD100KDataset  # noqa: E402


class _ProtocolDataset(Dataset[dict[str, Any]]):
    """Adds project batch metadata to samples returned by ``BDD100KDataset``."""

    def __init__(self, dataset: BDD100KDataset, context_slots: int) -> None:
        self.dataset = dataset
        self.context_slots = context_slots

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, target = self.dataset[index]
        target["image_size"] = torch.tensor(
            [image.shape[1], image.shape[2]], dtype=torch.int64
        )
        return {
            "image": image,
            "target": target,
            "sequence_id": "image-dataloader",
            "frame_id": index,
            "timestamp": float(index),
            "is_sequence_start": True,
            "context_valid_mask": torch.zeros(self.context_slots, dtype=torch.bool),
            "context_time_offsets": torch.zeros(
                self.context_slots, dtype=torch.float32
            ),
        }


class ImageDataLoaderProtocol:
    """Build reproducible train/validation loaders from the component config."""

    def build(self, config: ExperimentConfig, split: DatasetSplit) -> DataLoader[Any]:
        raw: Any = OmegaConf.to_container(
            OmegaConf.load(config.data.config_path), resolve=True
        )
        if not isinstance(raw, dict):
            raise TypeError("image_dataloader config must be a mapping")
        dataset_config = raw.get("dataset")
        if not isinstance(dataset_config, dict):
            raise ValueError("image_dataloader config requires dataset")

        image_size = config.data.image_size
        if image_size % 32:
            raise ValueError(
                "RF-DETR image_dataloader requires data.image_size divisible by 32, "
                f"got {image_size}"
            )
        # The experiment owns the model input resolution.  The component's
        # image_size is only a local default for standalone dataloader usage.
        dataset_config["image_size"] = {"width": image_size, "height": image_size}

        dataset_config["image_set"] = "train" if split is DatasetSplit.TRAIN else "val"

        self._resolve_paths(raw, Path(config.data.config_path).parent)
        source = BDD100KDataset(
            raw["dataset"]["images_dir"],
            raw["dataset"]["annotations_dir"],
            raw,
        )
        if not len(source):
            raise ValueError("image_dataloader found no image/annotation pairs")
        indices = self._split_indices(raw, len(source), split, config.train.seed)
        dataset = Subset(_ProtocolDataset(source, config.data.context_k), indices)
        workers = config.train.num_workers
        return DataLoader(
            dataset,
            batch_size=(
                config.train.batch_size
                if split is DatasetSplit.TRAIN
                else config.validation.batch_size
            ),
            shuffle=split is DatasetSplit.TRAIN,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
            collate_fn=DetectionCollator(),
        )

    @staticmethod
    def _resolve_paths(raw: dict[str, Any], root: Path) -> None:
        dataset_config = raw["dataset"]
        for key in ("images_dir", "annotations_dir"):
            value = Path(dataset_config[key])
            if not value.is_absolute():
                dataset_config[key] = str((root / value).resolve())

    @staticmethod
    def _split_indices(
        raw: dict[str, Any], length: int, split: DatasetSplit, seed: int
    ) -> list[int]:
        fraction = float(raw.get("splits", {}).get("train_fraction", 0.9))
        if not 0.0 < fraction < 1.0:
            raise ValueError("splits.train_fraction must be between 0 and 1")
        if length < 2:
            raise ValueError("image_dataloader requires at least two samples")
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(length, generator=generator).tolist()
        train_count = min(max(round(length * fraction), 1), length - 1)
        if split is DatasetSplit.TRAIN:
            return order[:train_count]
        return order[train_count:]


PROTOCOL = ImageDataLoaderProtocol()
