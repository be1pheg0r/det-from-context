import json
from math import isfinite
from pathlib import Path

import torch
from PIL import Image
from rfdetr.datasets.coco import make_coco_transforms_square_div_64
from torch.utils.data import Dataset


class BDD100KDataset(Dataset):
    """BDD100K-style dataset using RF-DETR's native preprocessing pipeline."""

    def __init__(self, images_dir, annotations_dir, config):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)

        dataset_config = config.get("dataset", config)

        image_size = dataset_config.get("image_size", {})
        if not isinstance(image_size, dict):
            image_size = {"width": image_size, "height": image_size}

        target_width = image_size.get("width")
        target_height = image_size.get("height")
        if target_width is None or target_height is None:
            raise ValueError("dataset.image_size.width/height must be configured")
        if target_width != target_height:
            raise ValueError(
                "RF-DETR square preprocessing requires image_size.width "
                "== image_size.height"
            )

        self.resolution = int(target_width)
        self.image_extensions = set(dataset_config.get("image_extensions", []))
        self.classes = config.get("classes", dataset_config.get("classes", {}))
        self.image_set = self._resolve_image_set(dataset_config)

        self.transform = make_coco_transforms_square_div_64(
            image_set=self.image_set,
            resolution=self.resolution,
            multi_scale=bool(dataset_config.get("multi_scale", False)),
            expanded_scales=bool(dataset_config.get("expanded_scales", False)),
            skip_random_resize=bool(dataset_config.get("skip_random_resize", False)),
            patch_size=int(dataset_config.get("patch_size", 16)),
            num_windows=int(dataset_config.get("num_windows", 4)),
            aug_config=dataset_config.get("aug_config"),
            scale_jitter=bool(dataset_config.get("scale_jitter", True)),
            gpu_postprocess=False,
        )

        self.samples = self._build_samples()

    def _resolve_image_set(self, dataset_config):
        configured = dataset_config.get("image_set")
        if configured is not None:
            value = str(configured).lower()
            if value == "validation":
                value = "val"
            if value not in {"train", "val", "test", "val_speed"}:
                raise ValueError(f"Unsupported RF-DETR image_set: {configured!r}")
            return value

        parts = {part.lower() for part in self.images_dir.parts}
        if "train" in parts:
            return "train"
        if "validation" in parts or "valid" in parts or "val" in parts:
            return "val"
        if "test" in parts:
            return "test"

        raise ValueError(
            "Cannot infer RF-DETR image_set from images_dir. "
            "Set dataset.image_set to train/val/test in the dataset config."
        )

    def _build_samples(self):
        samples = []

        for image_path in self.images_dir.iterdir():
            if image_path.suffix.lower() not in self.image_extensions:
                continue

            annotation_path = self.annotations_dir / f"{image_path.stem}.json"
            if not annotation_path.exists():
                continue

            samples.append({"image": image_path, "annotation": annotation_path})

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image"]
        annotation_path = sample["annotation"]

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        original_width, original_height = image.size

        with open(annotation_path, encoding="utf-8") as f:
            annotation = json.load(f)

        frames = annotation.get("frames", [])
        frame = frames[0] if isinstance(frames, list) and frames else {}
        objects = frame.get("objects", []) if isinstance(frame, dict) else []

        boxes = []
        labels = []
        areas = []

        for obj in objects:
            if not isinstance(obj, dict):
                continue

            category = obj.get("category")
            if category not in self.classes:
                continue

            box = obj.get("box2d")
            if not isinstance(box, dict):
                continue

            try:
                x1, y1, x2, y2 = (float(box[key]) for key in ("x1", "y1", "x2", "y2"))
            except (KeyError, TypeError, ValueError):
                continue

            if not all(isfinite(value) for value in (x1, y1, x2, y2)):
                continue

            x1 = min(max(x1, 0.0), original_width)
            x2 = min(max(x2, 0.0), original_width)
            y1 = min(max(y1, 0.0), original_height)
            y2 = min(max(y2, 0.0), original_height)

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self.classes[category])
            areas.append((x2 - x1) * (y2 - y1))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "image_id": torch.tensor(idx, dtype=torch.long),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.long),
            "orig_size": torch.tensor(
                [original_height, original_width], dtype=torch.long
            ),
            "size": torch.tensor([original_height, original_width], dtype=torch.long),
        }

        image, target = self.transform(image, target)
        return image, target
