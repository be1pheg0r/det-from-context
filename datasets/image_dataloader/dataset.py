import json
from math import isfinite
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class BDD100KDataset(Dataset):
    """BDD100K-style dataset built from explicit image and annotation folders."""

    def __init__(self, images_dir, annotations_dir, config):
        """Build the dataset from independent image and JSON annotation folders.

        ``config`` contains parsing and preprocessing options only.  It does
        not control where either dataset directory is located.

        The component config stores dataset-specific fields under
        ``dataset``. Older flat configs still pass them at the top level,
        so we accept both layouts for compatibility.
        """
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)

        dataset_config = config.get("dataset", config)

        image_size = dataset_config.get("image_size", {})
        if not isinstance(image_size, dict):
            image_size = {"width": image_size, "height": image_size}

        self.target_width = image_size.get("width")
        self.target_height = image_size.get("height")

        self.image_extensions = set(dataset_config.get("image_extensions", []))
        self.classes = config.get("classes", dataset_config.get("classes", {}))
        self.normalize_boxes = dataset_config.get(
            "normalize_boxes", config.get("normalize_boxes", False)
        )
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []

        for image_path in self.images_dir.iterdir():
            if image_path.suffix.lower() not in self.image_extensions:
                continue

            # Предполагаем, что:
            # image.jpg -> image.json
            annotation_path = self.annotations_dir / f"{image_path.stem}.json"

            if not annotation_path.exists():
                continue

            samples.append(
                {
                    "image": image_path,
                    "annotation": annotation_path,
                }
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_path = sample["image"]
        annotation_path = sample["annotation"]

        # -------------------------
        # 1. Читаем изображение
        # -------------------------

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f"Не удалось прочитать изображение: {image_path}")

        # OpenCV: H x W x C, BGR
        original_height, original_width = image.shape[:2]

        # -------------------------
        # 2. Читаем JSON
        # -------------------------

        with open(annotation_path, encoding="utf-8") as f:
            annotation = json.load(f)

        # Одному изображению соответствует первый frame.  BDD100K JSON может
        # также содержать polygon-only objects without ``box2d``.
        frames = annotation.get("frames", [])
        frame = frames[0] if isinstance(frames, list) and frames else {}
        objects = frame.get("objects", []) if isinstance(frame, dict) else []

        # -------------------------
        # 3. Получаем bbox
        # -------------------------

        boxes = []
        labels = []

        for obj in objects:
            if not isinstance(obj, dict):
                continue
            category = obj.get("category")

            # Пропускаем неизвестные классы
            if category not in self.classes:
                continue

            # Lines, drivable areas and other BDD100K objects may contain
            # ``poly2d`` only.  RF-DETR trains on finite, non-empty boxes.
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

        # -------------------------
        # 4. Resize изображения
        # -------------------------

        scale_x = self.target_width / original_width
        scale_y = self.target_height / original_height

        image = cv2.resize(image, (self.target_width, self.target_height))

        # -------------------------
        # 5. Масштабируем bbox
        # -------------------------

        resized_boxes = []

        for x1, y1, x2, y2 in boxes:
            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            if self.normalize_boxes:
                cx /= self.target_width
                cy /= self.target_height
                w /= self.target_width
                h /= self.target_height

            resized_boxes.append([cx, cy, w, h])

        # -------------------------
        # 6. Tensor
        # -------------------------

        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # HWC -> CHW
        image = torch.from_numpy(image).permute(2, 0, 1)

        # uint8 -> float32
        image = image.float() / 255.0

        # -------------------------
        # 7. Target
        # -------------------------

        target = {
            "boxes": torch.tensor(resized_boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": torch.tensor(idx, dtype=torch.long),
        }

        return image, target
