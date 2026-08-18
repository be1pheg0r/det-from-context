import json
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class BDD100KDataset(Dataset):
    def __init__(self, config):
        self.images_dir = Path(config["dataset"]["images_dir"])
        self.annotations_dir = Path(config["dataset"]["annotations_dir"])

        self.target_width = config["dataset"]["image_size"]["width"]
        self.target_height = config["dataset"]["image_size"]["height"]

        self.image_extensions = set(config["dataset"]["image_extensions"])

        self.classes = config["classes"]

        self.normalize_boxes = config["normalize_boxes"]
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

        # Пока предполагаем один frame на изображение
        frame = annotation["frames"][0]

        objects = frame.get("objects", [])

        # -------------------------
        # 3. Получаем bbox
        # -------------------------

        boxes = []
        labels = []

        for obj in objects:
            category = obj.get("category")
            if not category or category not in self.classes:
                continue

            # 2. Проверяем наличие блока координат
            box = obj.get("box2d")
            if not box:
                continue

            # 3. Проверяем наличие всех нужных точек
            required_keys = ("x1", "y1", "x2", "y2")
            if not all(k in box for k in required_keys):
                continue
            x1 = box["x1"]
            y1 = box["y1"]
            x2 = box["x2"]
            y2 = box["y2"]

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
            "boxes": torch.tensor(resized_boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": idx,
        }

        return image, target
