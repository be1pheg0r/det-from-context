"""Project dataset protocol for arbitrary paired image/annotation directories."""

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

from dataset import ImageDetectionDataset  # noqa: E402
from settings import ImageDataLoaderSettings  # noqa: E402


class _ProtocolDataset(Dataset[dict[str, Any]]):
    """Add project batch metadata to RF-DETR-compatible image samples."""

    def __init__(self, dataset: ImageDetectionDataset, context_slots: int) -> None:
        self.dataset = dataset
        self.context_slots = context_slots

    def __len__(self) -> int:
        """Return the wrapped dataset length."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one sample compatible with :class:`DetectionCollator`."""
        image, target = self.dataset[index]
        target["image_size"] = torch.tensor(
            [image.shape[1], image.shape[2]], dtype=torch.int64
        )
        return {
            "image": image,
            "target": target,
            "sequence_id": "image-dataloader",
            "frame_id": int(target["image_id"]),
            "timestamp": float(target["image_id"]),
            "is_sequence_start": True,
            "context_valid_mask": torch.zeros(self.context_slots, dtype=torch.bool),
            "context_time_offsets": torch.zeros(
                self.context_slots, dtype=torch.float32
            ),
            "extras": {"dataset_index": int(target["image_id"])},
        }


class ImageDataLoaderProtocol:
    """Build deterministic generated or fixed directory split DataLoaders."""

    def build(self, config: ExperimentConfig, split: DatasetSplit) -> DataLoader[Any]:
        """Build one project-standard DataLoader endpoint."""
        settings = self._load_settings(Path(config.data.config_path))
        self._set_experiment_resolution(settings, config.data.image_size)
        component_root = Path(config.data.config_path).parent
        images_root, annotations_root = self._resolved_roots(settings, component_root)
        image_set = {
            DatasetSplit.TRAIN: "train",
            DatasetSplit.VALIDATION: "val",
            DatasetSplit.TEST: "test",
        }[split]

        if settings.splits.mode == "predefined":
            child = settings.splits.directory(split.value)
            source = ImageDetectionDataset(
                images_root / child,
                annotations_root / child,
                settings,
                image_set=image_set,
            )
            dataset: Dataset[Any] = _ProtocolDataset(source, config.data.context_k)
        else:
            source = ImageDetectionDataset(
                images_root,
                annotations_root,
                settings,
                image_set=image_set,
            )
            indices = self._split_indices(
                settings, len(source), split, config.train.seed
            )
            dataset = Subset(_ProtocolDataset(source, config.data.context_k), indices)

        if not len(dataset):
            raise ValueError(f"image_dataloader split {split.value!r} is empty")
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
            generator=torch.Generator().manual_seed(config.train.seed),
        )

    @staticmethod
    def _load_settings(path: Path) -> ImageDataLoaderSettings:
        """Load and validate the component YAML."""
        raw: Any = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if not isinstance(raw, dict):
            raise TypeError("image_dataloader config must be a mapping")
        return ImageDataLoaderSettings.model_validate(raw)

    @staticmethod
    def _set_experiment_resolution(
        settings: ImageDataLoaderSettings, resolution: int
    ) -> None:
        """Apply the experiment-owned resolution after checking RF-DETR divisibility."""
        block_size = settings.dataset.patch_size * settings.dataset.num_windows
        if resolution % block_size:
            raise ValueError(
                "RF-DETR image_size must be divisible by patch_size * num_windows "
                f"({block_size}), got {resolution}"
            )
        settings.dataset.image_size.width = resolution
        settings.dataset.image_size.height = resolution

    @staticmethod
    def _resolved_roots(
        settings: ImageDataLoaderSettings, component_root: Path
    ) -> tuple[Path, Path]:
        """Resolve the two independent dataset roots relative to component YAML."""

        def resolve(value: str) -> Path:
            path = Path(value)
            return path if path.is_absolute() else (component_root / path).resolve()

        return resolve(settings.dataset.images_dir), resolve(
            settings.dataset.annotations_dir
        )

    @staticmethod
    def _split_indices(
        settings: ImageDataLoaderSettings,
        length: int,
        split: DatasetSplit,
        seed: int,
    ) -> list[int]:
        """Partition one stable permutation into disjoint generated splits."""
        fractions = [
            settings.splits.train_fraction,
            settings.splits.validation_fraction,
            settings.splits.test_fraction,
        ]
        positive = sum(fraction > 0 for fraction in fractions)
        if length < positive:
            raise ValueError(
                "image_dataloader needs at least "
                f"{positive} samples for configured splits"
            )
        raw_counts = [length * fraction for fraction in fractions]
        counts = [int(value) for value in raw_counts]
        for index, fraction in enumerate(fractions):
            if fraction > 0 and counts[index] == 0:
                counts[index] = 1
        while sum(counts) > length:
            candidates = [i for i, count in enumerate(counts) if count > 1]
            counts[max(candidates, key=lambda i: counts[i] - raw_counts[i])] -= 1
        while sum(counts) < length:
            candidates = [i for i, fraction in enumerate(fractions) if fraction > 0]
            counts[max(candidates, key=lambda i: raw_counts[i] - counts[i])] += 1

        index = list(DatasetSplit).index(split)
        if fractions[index] == 0:
            raise ValueError(
                f"generated split {split.value!r} has zero configured fraction"
            )
        order = torch.randperm(
            length, generator=torch.Generator().manual_seed(seed)
        ).tolist()
        start = sum(counts[:index])
        return order[start : start + counts[index]]


PROTOCOL = ImageDataLoaderProtocol()
