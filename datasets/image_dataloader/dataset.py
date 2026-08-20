"""Directory-backed detection dataset with RF-DETR-native preprocessing."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from annotation_readers import AnnotationReader, get_annotation_reader
from PIL import Image
from rfdetr.datasets.coco import make_coco_transforms_square_div_64
from settings import ImageDataLoaderSettings
from torch import Tensor
from torch.utils.data import Dataset

Transform = Callable[[Image.Image, dict[str, Tensor]], tuple[Tensor, dict[str, Tensor]]]


class ImageDetectionDataset(Dataset[tuple[Tensor, dict[str, Tensor]]]):
    """Pair images with per-image annotations and apply upstream RF-DETR transforms."""

    def __init__(
        self,
        images_dir: str | Path,
        annotations_dir: str | Path,
        config: ImageDataLoaderSettings | dict[str, Any],
        *,
        resolution: int,
        image_set: str | None = None,
        transform: Transform | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.settings = (
            config
            if isinstance(config, ImageDataLoaderSettings)
            else ImageDataLoaderSettings.model_validate(config)
        )
        dataset_config = self.settings.dataset
        if resolution <= 0:
            raise ValueError("RF-DETR resolution must be positive")
        self.resolution = resolution
        self.image_set = self._normalize_image_set(
            image_set or self._infer_image_set(self.images_dir)
        )
        self.reader: AnnotationReader = get_annotation_reader(
            dataset_config.annotation_format
        )
        self.transform: Transform = transform or make_coco_transforms_square_div_64(
            image_set=self.image_set,
            resolution=self.resolution,
            multi_scale=dataset_config.multi_scale,
            expanded_scales=dataset_config.expanded_scales,
            skip_random_resize=dataset_config.skip_random_resize,
            patch_size=dataset_config.patch_size,
            num_windows=dataset_config.num_windows,
            aug_config=dataset_config.aug_config,
            scale_jitter=dataset_config.scale_jitter,
            gpu_postprocess=False,
        )
        self.samples = self._build_samples()
        self._labels_cache: dict[int, tuple[int, ...]] = {}

    @staticmethod
    def _normalize_image_set(value: str) -> str:
        """Translate project split names to RF-DETR transform names."""
        normalized = value.lower()
        if normalized in {"validation", "valid"}:
            normalized = "val"
        if normalized not in {"train", "val", "test", "val_speed"}:
            raise ValueError(f"unsupported RF-DETR image_set: {value!r}")
        return normalized

    @staticmethod
    def _infer_image_set(images_dir: Path) -> str:
        """Infer a split only for backwards-compatible standalone use."""
        parts = {part.lower() for part in images_dir.parts}
        if "train" in parts:
            return "train"
        if parts.intersection({"validation", "valid", "val"}):
            return "val"
        if "test" in parts:
            return "test"
        raise ValueError("image_set is required when the path has no split directory")

    def _build_samples(self) -> list[dict[str, Path]]:
        """Build a sorted one-to-one image/annotation manifest."""
        if not self.images_dir.is_dir():
            raise FileNotFoundError(self.images_dir)
        if not self.annotations_dir.is_dir():
            raise FileNotFoundError(self.annotations_dir)
        dataset_config = self.settings.dataset
        iterator = (
            self.images_dir.rglob("*")
            if dataset_config.recursive
            else self.images_dir.iterdir()
        )
        images = sorted(
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in dataset_config.image_extensions
        )
        samples: list[dict[str, Path]] = []
        missing: list[Path] = []
        expected_annotations: set[Path] = set()
        for image_path in images:
            relative = image_path.relative_to(self.images_dir)
            annotation_path = (self.annotations_dir / relative).with_suffix(
                dataset_config.annotation_extension
            )
            expected_annotations.add(annotation_path.resolve())
            if not annotation_path.is_file():
                missing.append(annotation_path)
                continue
            samples.append({"image": image_path, "annotation": annotation_path})
        if dataset_config.strict_pairs and missing:
            preview = ", ".join(str(path) for path in missing[:3])
            raise ValueError(
                f"missing annotations for {len(missing)} images: {preview}"
            )
        if dataset_config.strict_pairs:
            annotation_iterator = (
                self.annotations_dir.rglob(f"*{dataset_config.annotation_extension}")
                if dataset_config.recursive
                else self.annotations_dir.glob(
                    f"*{dataset_config.annotation_extension}"
                )
            )
            orphans = sorted(
                path
                for path in annotation_iterator
                if path.resolve() not in expected_annotations
            )
            if orphans:
                preview = ", ".join(str(path) for path in orphans[:3])
                raise ValueError(f"orphan annotations without images: {preview}")
        return samples

    def __len__(self) -> int:
        """Return the number of paired samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        """Load one image, parse its annotation, and transform both consistently."""
        sample = self.samples[index]
        with Image.open(sample["image"]) as source:
            image = source.convert("RGB")
        width, height = image.size
        annotation = self.reader.read(
            sample["annotation"], self.settings.classes, (width, height)
        )
        target: dict[str, Tensor] = {
            "boxes": torch.tensor(annotation.boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(annotation.labels, dtype=torch.long),
            "image_id": torch.tensor(index, dtype=torch.long),
            "area": torch.tensor(annotation.areas, dtype=torch.float32),
            "iscrowd": torch.zeros(len(annotation.boxes), dtype=torch.long),
            "orig_size": torch.tensor([height, width], dtype=torch.long),
            "size": torch.tensor([height, width], dtype=torch.long),
            "rejected_objects": torch.tensor(annotation.rejected_objects),
        }
        image_tensor, transformed = self.transform(image, target)
        self._validate_transformed_target(image_tensor, transformed)
        return image_tensor, transformed

    def labels_by_image(self) -> list[tuple[int, ...]]:
        """Return raw object labels per image for splitting and train sampling."""
        return [self._labels_for_index(index) for index in range(len(self))]

    def _labels_for_index(self, index: int) -> tuple[int, ...]:
        """Read and cache only labels, avoiding transformed image materialization."""
        cached = self._labels_cache.get(index)
        if cached is not None:
            return cached
        sample = self.samples[index]
        with Image.open(sample["image"]) as source:
            image_size = source.size
        annotation = self.reader.read(
            sample["annotation"], self.settings.classes, image_size
        )
        labels = tuple(annotation.labels)
        self._labels_cache[index] = labels
        return labels

    def manifest(self) -> list[dict[str, str]]:
        """Return stable paths suitable for split provenance artifacts."""
        return [
            {
                "image": str(sample["image"]),
                "annotation": str(sample["annotation"]),
            }
            for sample in self.samples
        ]

    @staticmethod
    def _validate_transformed_target(image: Tensor, target: dict[str, Tensor]) -> None:
        """Fail before training when transforms violate the RF-DETR target contract."""
        if image.ndim != 3 or image.shape[0] != 3 or not torch.isfinite(image).all():
            raise ValueError("transformed image must be a finite [3,H,W] tensor")
        boxes = target.get("boxes")
        labels = target.get("labels")
        if boxes is None or boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("transformed target boxes must have shape [N,4]")
        if labels is None or labels.shape != (boxes.shape[0],):
            raise ValueError("transformed target labels must have shape [N]")
        if boxes.numel() and (
            not torch.isfinite(boxes).all() or boxes.min() < 0 or boxes.max() > 1
        ):
            raise ValueError("RF-DETR transformed boxes must be normalized to [0,1]")


# Backwards-compatible import used by the existing notebook and standalone loader.
BDD100KDataset = ImageDetectionDataset
