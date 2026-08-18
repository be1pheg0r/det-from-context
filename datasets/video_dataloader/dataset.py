# dataset.py

import json
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """
    Dataset для object detection по видео.

    Один sample содержит последовательность кадров:

        [frame_0, frame_1, ..., reference_frame]

    Последний кадр является опорным.
    Target содержит разметку только опорного кадра.

    Возвращает:

        frames:
            Tensor [T, C, H, W]

        target:
            {
                "boxes": Tensor [N, 4],
                "labels": Tensor [N],
                "image_id": int
            }
    """

    def __init__(self, config):
        self.videos_dir = Path(config["video"]["videos_dir"])
        self.annotations_dir = Path(config["video"]["annotations_dir"])

        self.reference_frame = config["video"]["reference_frame"]
        self.frames_before = config["video"]["frames_before"]
        self.target_fps = config["video"]["fps"]

        self.target_width = config["dataset"]["image_size"]["width"]
        self.target_height = config["dataset"]["image_size"]["height"]

        self.classes = config["classes"]

        self.samples = self._build_samples()

    # ============================================================
    # BUILD SAMPLES
    # ============================================================

    def _build_samples(self):
        """
        Создаёт список samples.

        Каждый sample содержит:

            video_path
            frames
            reference_frame

        frames — список кадров, которые нужно загрузить.
        """

        samples = []

        video_extensions = {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
        }
        from tqdm import tqdm
        for video_path in tqdm(sorted(self.videos_dir.iterdir())):
            if video_path.suffix.lower() not in video_extensions:
                continue

            annotation_path = self.annotations_dir / f"{video_path.stem}.json"

            if not annotation_path.exists():
                continue

            with open(annotation_path, encoding="utf-8") as f:
                annotation = json.load(f)

            frames = annotation.get("frames", [])

            if not frames:
                continue

            video_samples = self._build_video_samples(
                video_path, annotation_path, frames
            )

            samples.extend(video_samples)

        return samples

    # ============================================================
    # BUILD VIDEO SAMPLES
    # ============================================================

    def _build_video_samples(self, video_path, annotation_path, frames):
        """
        Формирует samples для одного видео.

        Для каждого опорного кадра выбираются
        предыдущие кадры с заданным временным интервалом.
        """

        samples = []

        # timestamp в JSON — миллисекунды.
        interval_ms = 1000.0 / self.target_fps

        timestamps = [frame["timestamp"] for frame in frames]

        for reference_idx in range(len(frames)):
            reference_timestamp = timestamps[reference_idx]

            selected_frames = []

            # ------------------------------------------------
            # Выбираем предыдущие кадры
            # ------------------------------------------------

            for i in range(self.frames_before, -1, -1):
                target_timestamp = reference_timestamp - i * interval_ms

                frame_idx = self._find_closest_frame(
                    timestamps, target_timestamp, reference_idx
                )

                if frame_idx is None:
                    continue

                selected_frames.append(frame_idx)

            # ------------------------------------------------
            # Если не хватает кадров в начале видео,
            # sample пропускаем
            # ------------------------------------------------

            if len(selected_frames) != self.frames_before + 1:
                continue

            samples.append(
                {
                    "video_path": video_path,
                    "annotation_path": annotation_path,
                    "frame_indices": selected_frames,
                    # Последний frame — reference
                    "reference_idx": selected_frames[-1],
                    "reference_timestamp": reference_timestamp,
                }
            )

        return samples

    # ============================================================
    # FIND CLOSEST FRAME
    # ============================================================

    def _find_closest_frame(self, timestamps, target_timestamp, max_idx):
        """
        Ищет кадр, timestamp которого наиболее близок
        к target_timestamp.

        Ищем только среди кадров до reference frame.
        """

        if max_idx < 0:
            return None

        best_idx = None
        best_distance = float("inf")

        for idx in range(max_idx + 1):
            distance = abs(timestamps[idx] - target_timestamp)

            if distance < best_distance:
                best_distance = distance
                best_idx = idx

        return best_idx

    # ============================================================
    # LEN
    # ============================================================

    def __len__(self):
        return len(self.samples)

    # ============================================================
    # GET ITEM
    # ============================================================

    def __getitem__(self, idx):

        sample = self.samples[idx]

        video_path = sample["video_path"]
        annotation_path = sample["annotation_path"]

        frame_indices = sample["frame_indices"]
        reference_idx = sample["reference_idx"]

        # ------------------------------------------------
        # Загружаем JSON
        # ------------------------------------------------

        with open(annotation_path, encoding="utf-8") as f:
            annotation = json.load(f)

        annotation_frames = annotation["frames"]

        # ------------------------------------------------
        # Открываем видео
        # ------------------------------------------------

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {video_path}")

        frames = []

        # ------------------------------------------------
        # Загружаем нужные кадры
        # ------------------------------------------------

        for frame_idx in frame_indices:
            frame_info = annotation_frames[frame_idx]

            timestamp_ms = frame_info["timestamp"]

            # OpenCV позиционируется по миллисекундам
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)

            success, image = cap.read()

            if not success:
                cap.release()

                raise RuntimeError(
                    f"Не удалось прочитать кадр {frame_idx} из {video_path}"
                )

            # --------------------------------------------
            # Resize
            # --------------------------------------------

            original_height, original_width = image.shape[:2]

            image = cv2.resize(image, (self.target_width, self.target_height))

            # --------------------------------------------
            # BGR -> RGB
            # --------------------------------------------

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # --------------------------------------------
            # HWC -> CHW
            # --------------------------------------------

            image = torch.from_numpy(image).permute(2, 0, 1)

            # --------------------------------------------
            # uint8 -> float
            # --------------------------------------------

            image = image.float() / 255.0

            frames.append(image)

        cap.release()

        # ------------------------------------------------
        # [T, C, H, W]
        # ------------------------------------------------

        frames = torch.stack(frames)

        # ------------------------------------------------
        # Target reference frame
        # ------------------------------------------------

        reference_frame = annotation_frames[reference_idx]

        target = self._create_target(
            reference_frame, original_width, original_height, idx
        )

        return frames, target

    # ============================================================
    # CREATE TARGET
    # ============================================================

    def _create_target(self, frame, original_width, original_height, image_id):
        """
        Создаёт target для reference frame.
        """

        boxes = []
        labels = []

        # Коэффициенты resize
        scale_x = self.target_width / original_width

        scale_y = self.target_height / original_height

        for obj in frame.get("objects", []):
            category = obj["category"]

            # Неизвестный класс пропускаем
            if category not in self.classes:
                continue

            box = obj["box2d"]

            x1 = box["x1"] * scale_x
            y1 = box["y1"] * scale_y
            x2 = box["x2"] * scale_x
            y2 = box["y2"] * scale_y

            boxes.append([x1, y1, x2, y2])

            labels.append(self.classes[category])

        # ------------------------------------------------
        # Empty target
        # ------------------------------------------------

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)

        if labels:
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            labels = torch.empty((0,), dtype=torch.long)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
        }

        return target
