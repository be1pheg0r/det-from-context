"""Project dataset protocol for arbitrary paired image/annotation directories."""

from __future__ import annotations

import math
import random
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from context_detection.config import ExperimentConfig
from context_detection.data.collate import DetectionCollator
from context_detection.data.protocols import DatasetSplit
from context_detection.models.rfdetr import rfdetr_pretrained_resolution

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
        if "image_size" in config.data.model_fields_set:
            raise ValueError(
                "data.image_size must not be set for RF-DETR; the pretrained variant "
                "owns its input resolution"
            )
        variant = config.detector.variant
        if config.detector.name != "rfdetr" or variant is None:
            raise ValueError("image_dataloader requires an RF-DETR pretrained variant")
        resolution = rfdetr_pretrained_resolution(variant)
        self._validate_resolution(settings, resolution)
        component_root = Path(config.data.config_path).parent
        images_root, annotations_root = self._resolved_roots(settings, component_root)
        image_set = {
            DatasetSplit.TRAIN: "train",
            DatasetSplit.VALIDATION: "val",
            DatasetSplit.TEST: "test",
        }[split]

        selected_indices: list[int] | None = None
        if settings.splits.mode == "predefined":
            child = settings.splits.directory(split.value)
            source = ImageDetectionDataset(
                images_root / child,
                annotations_root / child,
                settings,
                resolution=resolution,
                image_set=image_set,
            )
        else:
            source = ImageDetectionDataset(
                images_root,
                annotations_root,
                settings,
                resolution=resolution,
                image_set=image_set,
            )
            labels_for_split = (
                source.labels_by_image()
                if settings.class_balance.stratify_generated
                else None
            )
            selected_indices = self._split_indices(
                settings,
                len(source),
                split,
                config.train.seed,
                labels_by_image=labels_for_split,
            )
        protocol_dataset = _ProtocolDataset(source, config.data.context_k)
        dataset: Dataset[Any] = (
            protocol_dataset
            if selected_indices is None
            else Subset(protocol_dataset, selected_indices)
        )

        if not len(dataset):
            raise ValueError(f"image_dataloader split {split.value!r} is empty")
        labels_by_image: list[tuple[int, ...]] | None = None
        sampler: WeightedRandomSampler | None = None
        balance_summary: dict[str, Any] | None = None
        balance_enabled = (
            settings.class_balance.stratify_generated
            or settings.class_balance.sampling != "none"
        )
        if balance_enabled:
            source_labels = source.labels_by_image()
            labels_by_image = (
                source_labels
                if selected_indices is None
                else [source_labels[index] for index in selected_indices]
            )
            sample_weights: list[float] | None = None
            class_weights: dict[int, float] = {}
            if (
                split is DatasetSplit.TRAIN
                and settings.class_balance.sampling != "none"
            ):
                sample_weights, class_weights = self._sample_weights(
                    labels_by_image,
                    len(settings.classes),
                    settings.class_balance.sampling,
                    settings.class_balance.max_sample_weight,
                )
                if max(sample_weights) - min(sample_weights) > 1e-12:
                    sampler = WeightedRandomSampler(
                        sample_weights,
                        num_samples=len(sample_weights),
                        replacement=True,
                        generator=torch.Generator().manual_seed(config.train.seed),
                    )
            balance_summary = self._balance_summary(
                labels_by_image,
                settings.classes,
                settings.class_balance.sampling
                if split is DatasetSplit.TRAIN
                else "none",
                class_weights,
                sample_weights,
            )
        workers = config.train.num_workers
        loader = DataLoader(
            dataset,
            batch_size=(
                config.train.batch_size
                if split is DatasetSplit.TRAIN
                else config.validation.batch_size
            ),
            shuffle=split is DatasetSplit.TRAIN and sampler is None,
            sampler=sampler,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
            collate_fn=DetectionCollator(),
            generator=torch.Generator().manual_seed(config.train.seed),
        )
        if balance_summary is not None:
            loader.class_balance_summary = balance_summary  # type: ignore[attr-defined]
        return loader

    @staticmethod
    def _load_settings(path: Path) -> ImageDataLoaderSettings:
        """Load and validate the component YAML."""
        raw: Any = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if not isinstance(raw, dict):
            raise TypeError("image_dataloader config must be a mapping")
        return ImageDataLoaderSettings.model_validate(raw)

    @staticmethod
    def _validate_resolution(
        settings: ImageDataLoaderSettings, resolution: int
    ) -> None:
        """Check that the variant-owned resolution matches preprocessing geometry."""
        block_size = settings.dataset.patch_size * settings.dataset.num_windows
        if resolution % block_size:
            raise ValueError(
                "RF-DETR image_size must be divisible by patch_size * num_windows "
                f"({block_size}), got {resolution}"
            )

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
        *,
        labels_by_image: Sequence[Sequence[int]] | None = None,
    ) -> list[int]:
        """Partition randomly or with deterministic multilabel stratification."""
        fractions = [
            settings.splits.train_fraction,
            settings.splits.validation_fraction,
            settings.splits.test_fraction,
        ]
        counts = ImageDataLoaderProtocol._proportional_counts(
            length, fractions, ensure_nonempty=True
        )

        index = list(DatasetSplit).index(split)
        if fractions[index] == 0:
            raise ValueError(
                f"generated split {split.value!r} has zero configured fraction"
            )
        if settings.class_balance.stratify_generated:
            if labels_by_image is None or len(labels_by_image) != length:
                raise ValueError(
                    "stratified generated splits require labels for every image"
                )
            partitions = ImageDataLoaderProtocol._stratified_partitions(
                labels_by_image, counts, fractions, seed
            )
            return partitions[index]
        order = torch.randperm(
            length, generator=torch.Generator().manual_seed(seed)
        ).tolist()
        start = sum(counts[:index])
        return order[start : start + counts[index]]

    @staticmethod
    def _proportional_counts(
        total: int,
        fractions: Sequence[float],
        *,
        ensure_nonempty: bool,
    ) -> list[int]:
        """Allocate an exact integer total using deterministic largest remainders."""
        positive = sum(fraction > 0 for fraction in fractions)
        if ensure_nonempty and total < positive:
            raise ValueError(
                "image_dataloader needs at least "
                f"{positive} samples for configured splits"
            )
        raw = [total * fraction for fraction in fractions]
        counts = [math.floor(value) for value in raw]
        if ensure_nonempty:
            counts = [
                max(count, 1) if fraction > 0 else 0
                for count, fraction in zip(counts, fractions, strict=True)
            ]
        while sum(counts) > total:
            candidates = [
                index
                for index, count in enumerate(counts)
                if count > (1 if ensure_nonempty and fractions[index] > 0 else 0)
            ]
            selected = min(
                candidates,
                key=lambda index: (raw[index] - counts[index], index),
            )
            counts[selected] -= 1
        while sum(counts) < total:
            candidates = [
                index for index, fraction in enumerate(fractions) if fraction > 0
            ]
            selected = max(
                candidates,
                key=lambda index: (raw[index] - counts[index], -index),
            )
            counts[selected] += 1
        return counts

    @staticmethod
    def _stratified_partitions(
        labels_by_image: Sequence[Sequence[int]],
        split_sizes: Sequence[int],
        fractions: Sequence[float],
        seed: int,
    ) -> list[list[int]]:
        """Greedily preserve multilabel image presence at exact split sizes.

        The rarest remaining label is assigned first. Each assignment consumes
        the integer per-class quota of every label present in that image, which
        is the multilabel analogue of stratified single-label splitting.
        """
        if sum(split_sizes) != len(labels_by_image):
            raise ValueError("stratified split sizes must cover every image")
        normalized = [
            frozenset(int(label) for label in labels) for labels in labels_by_image
        ]
        label_to_remaining: dict[int, set[int]] = {}
        for sample_index, labels in enumerate(normalized):
            for label in labels:
                label_to_remaining.setdefault(label, set()).add(sample_index)
        desired: dict[int, list[int]] = {
            label: ImageDataLoaderProtocol._proportional_counts(
                len(indices), fractions, ensure_nonempty=False
            )
            for label, indices in label_to_remaining.items()
        }
        rng = random.Random(seed)
        remaining = set(range(len(normalized)))
        capacities = list(split_sizes)
        partitions: list[list[int]] = [[] for _ in split_sizes]

        while True:
            active = [
                (len(indices & remaining), label)
                for label, indices in label_to_remaining.items()
                if indices & remaining
            ]
            if not active:
                break
            rarest_count = min(count for count, _ in active)
            rarest_labels = sorted(
                label for count, label in active if count == rarest_count
            )
            focus_label = rng.choice(rarest_labels)
            candidates = list(label_to_remaining[focus_label] & remaining)
            rng.shuffle(candidates)
            for sample_index in candidates:
                if sample_index not in remaining:
                    continue
                available = [
                    index for index, capacity in enumerate(capacities) if capacity > 0
                ]
                labels = normalized[sample_index]
                scores = {
                    split_index: (
                        desired[focus_label][split_index],
                        sum(max(desired[label][split_index], 0) for label in labels),
                        capacities[split_index] / max(split_sizes[split_index], 1),
                    )
                    for split_index in available
                }
                best_score = max(scores.values())
                best_splits = sorted(
                    index for index, score in scores.items() if score == best_score
                )
                selected_split = rng.choice(best_splits)
                partitions[selected_split].append(sample_index)
                capacities[selected_split] -= 1
                remaining.remove(sample_index)
                for label in labels:
                    desired[label][selected_split] -= 1
                    label_to_remaining[label].discard(sample_index)

        leftovers = list(remaining)
        rng.shuffle(leftovers)
        for sample_index in leftovers:
            largest = max(capacities)
            choices = [
                index
                for index, capacity in enumerate(capacities)
                if capacity == largest
            ]
            selected_split = rng.choice(choices)
            partitions[selected_split].append(sample_index)
            capacities[selected_split] -= 1
        if any(capacities):
            raise RuntimeError(
                "stratified split allocation did not fill all capacities"
            )
        return partitions

    @staticmethod
    def _sample_weights(
        labels_by_image: Sequence[Sequence[int]],
        num_classes: int,
        strategy: str,
        max_weight: float,
    ) -> tuple[list[float], dict[int, float]]:
        """Build image weights from inverse class presence frequency."""
        presence = Counter(
            label
            for labels in labels_by_image
            for label in set(int(value) for value in labels)
        )
        observed = [count for count in presence.values() if count > 0]
        if not observed:
            return [1.0] * len(labels_by_image), {}
        majority = max(observed)
        exponent = 0.5 if strategy == "inverse_sqrt" else 1.0
        class_weights = {
            class_id: min(
                max_weight,
                (majority / presence[class_id]) ** exponent,
            )
            for class_id in range(num_classes)
            if presence[class_id] > 0
        }
        sample_weights = [
            max((class_weights[label] for label in set(labels)), default=1.0)
            for labels in labels_by_image
        ]
        return sample_weights, class_weights

    @staticmethod
    def _balance_summary(
        labels_by_image: Sequence[Sequence[int]],
        classes: dict[str, int],
        sampling: str,
        class_weights: dict[int, float],
        sample_weights: Sequence[float] | None,
    ) -> dict[str, Any]:
        """Return JSON-safe class counts and effective train sampling weights."""
        image_presence = Counter(
            label for labels in labels_by_image for label in set(labels)
        )
        object_counts = Counter(label for labels in labels_by_image for label in labels)
        ordered = sorted(classes.items(), key=lambda item: item[1])
        total_images = len(labels_by_image)
        return {
            "sampling": sampling,
            "num_images": total_images,
            "replacement": bool(
                sample_weights and max(sample_weights) - min(sample_weights) > 1e-12
            ),
            "sample_weight_range": (
                [min(sample_weights), max(sample_weights)] if sample_weights else None
            ),
            "classes": {
                name: {
                    "id": class_id,
                    "images": image_presence[class_id],
                    "objects": object_counts[class_id],
                    "image_frequency": (
                        image_presence[class_id] / total_images if total_images else 0.0
                    ),
                    "sampling_weight": class_weights.get(class_id),
                }
                for name, class_id in ordered
            },
        }


PROTOCOL = ImageDataLoaderProtocol()
