"""Video clips with BDD100K-style annotations and RF-DETR preprocessing."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol

import cv2
import numpy as np
import torch
from PIL import Image
from rfdetr.datasets.coco import make_coco_transforms_square_div_64
from torch import Tensor
from torch.utils.data import Dataset


def _progress(message: str) -> None:
    print(f"[video_dataloader] {message}", flush=True)


if TYPE_CHECKING:
    from settings import VideoDataLoaderSettings


@dataclass(frozen=True)
class AnnotatedFrame:
    """One timestamp and its normalized source objects."""

    timestamp_ms: float
    objects: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class VideoRecord:
    """One paired video/annotation sequence."""

    sequence_id: str
    video_path: Path
    annotation_path: Path
    frames: tuple[AnnotatedFrame, ...]
    mode: str


@dataclass(frozen=True)
class ClipSpec:
    """Timestamps and optional annotation indices for one fixed-length clip."""

    record_index: int
    timestamps_ms: tuple[float, ...]
    annotation_indices: tuple[int | None, ...]
    sequence_start: bool


class FrameReader(Protocol):
    """Decode requested video timestamps as RGB images."""

    def read(self, path: Path, timestamps_ms: tuple[float, ...]) -> list[Image.Image]:
        """Return one decoded image for every requested timestamp."""


class OpenCVFrameReader:
    """Timestamp-seeking OpenCV reader with container rotation enabled."""

    def __init__(self, *, auto_rotate: bool, max_seek_error_ms: float) -> None:
        self.auto_rotate = auto_rotate
        self.max_seek_error_ms = max_seek_error_ms

    def read(self, path: Path, timestamps_ms: tuple[float, ...]) -> list[Image.Image]:
        """Decode monotonically ordered timestamps and validate seek accuracy."""
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        try:
            orientation_auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
            if orientation_auto is not None:
                capture.set(orientation_auto, 1.0 if self.auto_rotate else 0.0)
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            images: list[Image.Image] = []
            for timestamp_ms in timestamps_ms:
                capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
                success, frame = capture.read()
                if not success:
                    raise RuntimeError(f"cannot decode {path} at {timestamp_ms:.3f} ms")
                if fps > 0 and self.max_seek_error_ms >= 0:
                    frame_position = max(
                        float(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1.0, 0.0
                    )
                    actual_ms = frame_position * 1000.0 / fps
                    if abs(actual_ms - timestamp_ms) > self.max_seek_error_ms:
                        raise RuntimeError(
                            f"video seek error for {path}: requested "
                            f"{timestamp_ms:.3f} ms, decoded {actual_ms:.3f} ms"
                        )
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(Image.fromarray(rgb))
            return images
        finally:
            capture.release()


class VideoClipDataset(Dataset[dict[str, Any]]):
    """Build fixed-length clips for full tracking or reference-frame training.

    Tracking annotations contribute supervision at every clip step. A
    reference-frame annotation contributes supervision only at the final step;
    preceding frames are decoded from the real video and exist solely to warm
    temporal memory.
    """

    def __init__(
        self,
        videos_root: str | Path,
        annotations_root: str | Path,
        settings: VideoDataLoaderSettings,
        *,
        split: str,
        split_names: frozenset[str],
        clip_len: int,
        resolution: int,
        frame_reader: FrameReader | None = None,
        transform: Any | None = None,
    ) -> None:
        if clip_len < 1:
            raise ValueError("clip_len must be positive")
        self.videos_root = Path(videos_root)
        self.annotations_root = Path(annotations_root)
        self.settings = settings
        self.split = split
        self.split_names = split_names
        self.clip_len = clip_len
        self.resolution = resolution
        dataset_settings = settings.dataset
        _progress(
            f"initializing split={split}, videos_root={self.videos_root}, "
            f"annotations_root={self.annotations_root}, "
            f"recursive={dataset_settings.recursive}, "
            f"strict_pairs={dataset_settings.strict_pairs}"
        )
        self.frame_reader = frame_reader or OpenCVFrameReader(
            auto_rotate=dataset_settings.auto_rotate,
            max_seek_error_ms=dataset_settings.max_seek_error_ms,
        )
        image_set = "val" if split == "validation" else split
        self.transform = transform or make_coco_transforms_square_div_64(
            image_set=image_set,
            resolution=resolution,
            multi_scale=dataset_settings.multi_scale,
            expanded_scales=dataset_settings.expanded_scales,
            skip_random_resize=dataset_settings.skip_random_resize,
            patch_size=dataset_settings.patch_size,
            num_windows=dataset_settings.num_windows,
            aug_config=dataset_settings.aug_config,
            scale_jitter=dataset_settings.scale_jitter,
            gpu_postprocess=False,
        )
        self.records = self._build_records()
        _progress(
            f"split={split}: building clip index from {len(self.records)} records"
        )
        self.samples = self._build_clips()
        if not self.samples:
            raise ValueError(f"video_dataloader split {split!r} contains no clips")
        _progress(
            f"split={split} ready: records={len(self.records)}, "
            f"clips={len(self.samples)}"
        )

    def _build_records(self) -> list[VideoRecord]:
        """Pair files by stem inside the requested logical split."""
        if not self.videos_root.is_dir():
            raise FileNotFoundError(self.videos_root)
        if not self.annotations_root.is_dir():
            raise FileNotFoundError(self.annotations_root)
        dataset_settings = self.settings.dataset
        started = perf_counter()
        limit = dataset_settings.max_sequences.get(self.split)
        if limit is not None and not dataset_settings.strict_pairs:
            videos, annotations = self._bounded_pairs(limit)
            _progress(
                f"split={self.split}: bounded discovery stopped after "
                f"{len(set(videos).intersection(annotations))} matched sequences "
                f"in {perf_counter() - started:.1f}s"
            )
        else:
            _progress(f"split={self.split}: scanning video files")
            videos = self._unique_by_stem(
                self._files_for_split(
                    self.videos_root, set(dataset_settings.video_extensions)
                ),
                "videos",
                strict=dataset_settings.strict_pairs,
            )
            _progress(
                f"split={self.split}: found {len(videos)} videos in "
                f"{perf_counter() - started:.1f}s; scanning annotations"
            )
            started = perf_counter()
            annotations = self._unique_by_stem(
                self._files_for_split(
                    self.annotations_root, {dataset_settings.annotation_extension}
                ),
                "annotations",
                strict=dataset_settings.strict_pairs,
            )
            _progress(
                f"split={self.split}: found {len(annotations)} annotations in "
                f"{perf_counter() - started:.1f}s"
            )
        missing = sorted(set(videos) - set(annotations))
        orphans = sorted(set(annotations) - set(videos))
        if dataset_settings.strict_pairs and (missing or orphans):
            details: list[str] = []
            if missing:
                details.append(f"missing annotations: {missing[:3]}")
            if orphans:
                details.append(f"orphan annotations: {orphans[:3]}")
            raise ValueError("; ".join(details))

        records: list[VideoRecord] = []
        paired_stems = sorted(set(videos).intersection(annotations))
        if limit is not None:
            paired_stems = paired_stems[:limit]
            _progress(
                f"split={self.split}: limiting annotation parsing to "
                f"{len(paired_stems)} sequences"
            )
        _progress(
            f"split={self.split}: paired={len(paired_stems)}, missing={len(missing)}, "
            f"orphans={len(orphans)}; parsing JSON annotations"
        )
        started = perf_counter()
        report_every = max(1, len(paired_stems) // 20)
        for index, stem in enumerate(paired_stems, start=1):
            frames = self._read_annotation(annotations[stem])
            mode = self._resolve_mode(frames, annotations[stem])
            records.append(
                VideoRecord(
                    sequence_id=stem,
                    video_path=videos[stem],
                    annotation_path=annotations[stem],
                    frames=frames,
                    mode=mode,
                )
            )
            if index == 1 or index % report_every == 0 or index == len(paired_stems):
                _progress(
                    f"split={self.split}: parsed {index}/{len(paired_stems)} "
                    f"annotations ({perf_counter() - started:.1f}s)"
                )
        return records

    def _bounded_pairs(self, limit: int) -> tuple[dict[str, Path], dict[str, Path]]:
        """Stop lazy file discovery as soon as enough video/JSON stems match."""
        dataset_settings = self.settings.dataset
        streams = (
            iter(
                self._iter_files_for_split(
                    self.videos_root, set(dataset_settings.video_extensions)
                )
            ),
            iter(
                self._iter_files_for_split(
                    self.annotations_root, {dataset_settings.annotation_extension}
                )
            ),
        )
        results: tuple[dict[str, Path], dict[str, Path]] = ({}, {})
        exhausted = [False, False]
        while len(set(results[0]).intersection(results[1])) < limit:
            advanced = False
            for index, stream in enumerate(streams):
                if exhausted[index]:
                    continue
                try:
                    path = next(stream)
                except StopIteration:
                    exhausted[index] = True
                else:
                    results[index].setdefault(path.stem, path)
                    advanced = True
            if not advanced:
                break
        return results

    def _files_for_split(self, root: Path, extensions: set[str]) -> list[Path]:
        started = perf_counter()
        iterator = (
            root.rglob("*") if self.settings.dataset.recursive else root.iterdir()
        )
        selected: list[Path] = []
        root_is_split = root.name.lower() in self.split_names
        visited = 0
        for path in iterator:
            visited += 1
            if visited % 10_000 == 0:
                _progress(
                    f"split={self.split}: scanned {visited} paths under {root} "
                    f"({perf_counter() - started:.1f}s)"
                )
            if (
                path.is_file()
                and path.suffix.lower() in extensions
                and (
                    root_is_split
                    or any(
                        part.lower() in self.split_names
                        for part in path.relative_to(root).parts
                    )
                )
            ):
                selected.append(path)
        _progress(
            f"split={self.split}: scan complete for {root}; visited={visited}, "
            f"selected={len(selected)}, elapsed={perf_counter() - started:.1f}s"
        )
        return sorted(selected)

    def _iter_files_for_split(self, root: Path, extensions: set[str]) -> Any:
        iterator = (
            root.rglob("*") if self.settings.dataset.recursive else root.iterdir()
        )
        root_is_split = root.name.lower() in self.split_names
        for path in iterator:
            if (
                path.is_file()
                and path.suffix.lower() in extensions
                and (
                    root_is_split
                    or any(
                        part.lower() in self.split_names
                        for part in path.relative_to(root).parts
                    )
                )
            ):
                yield path

    @staticmethod
    def _unique_by_stem(
        paths: list[Path], kind: str, *, strict: bool
    ) -> dict[str, Path]:
        result: dict[str, Path] = {}
        duplicates: dict[str, list[Path]] = {}
        for path in paths:
            if path.stem in result:
                duplicates.setdefault(path.stem, [result[path.stem]]).append(path)
            else:
                result[path.stem] = path
        if duplicates:
            preview = ", ".join(
                f"{stem}: {[str(path) for path in values]}"
                for stem, values in list(duplicates.items())[:3]
            )
            if strict:
                raise ValueError(f"duplicate {kind} stems inside split: {preview}")
            _progress(
                f"ignoring duplicate {kind} paths for {len(duplicates)} stems; "
                f"using first sorted path. Examples: {preview}"
            )
        return result

    def _read_annotation(self, path: Path) -> tuple[AnnotatedFrame, ...]:
        """Parse BDD100K-like frames and assign contiguous local track IDs."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        source_frames = raw.get("frames") if isinstance(raw, dict) else None
        if not isinstance(source_frames, list) or not source_frames:
            raise ValueError(f"{path}: non-empty 'frames' list is required")
        track_ids: dict[str, int] = {}
        parsed: list[AnnotatedFrame] = []
        for frame_index, frame in enumerate(source_frames):
            if not isinstance(frame, dict) or not isinstance(
                frame.get("timestamp"), int | float
            ):
                raise ValueError(f"{path}: frames[{frame_index}] needs timestamp")
            objects: list[dict[str, Any]] = []
            source_objects = frame.get("objects", [])
            if not isinstance(source_objects, list):
                raise ValueError(
                    f"{path}: frames[{frame_index}].objects must be a list"
                )
            for source in source_objects:
                if not isinstance(source, dict):
                    continue
                category = source.get("category")
                box = source.get("box2d")
                if category not in self.settings.classes or not isinstance(box, dict):
                    continue
                try:
                    coordinates = tuple(
                        float(box[key]) for key in ("x1", "y1", "x2", "y2")
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                if not all(np.isfinite(coordinates)):
                    continue
                x1, y1, x2, y2 = coordinates
                if x2 <= x1 or y2 <= y1:
                    continue
                source_id = source.get("id")
                normalized_id = -1
                if source_id is not None:
                    key = str(source_id)
                    normalized_id = track_ids.setdefault(key, len(track_ids))
                objects.append(
                    {
                        "category": str(category),
                        "box": coordinates,
                        "track_id": normalized_id,
                    }
                )
            parsed.append(AnnotatedFrame(float(frame["timestamp"]), tuple(objects)))
        parsed.sort(key=lambda frame: frame.timestamp_ms)
        timestamps = [frame.timestamp_ms for frame in parsed]
        if len(timestamps) != len(set(timestamps)):
            raise ValueError(f"{path}: duplicate frame timestamps")
        return tuple(parsed)

    def _resolve_mode(self, frames: tuple[AnnotatedFrame, ...], path: Path) -> str:
        requested = self.settings.dataset.annotation_mode
        objects = [obj for frame in frames for obj in frame.objects]
        has_ids = bool(objects) and all(obj["track_id"] >= 0 for obj in objects)
        tracking_capable = len(frames) >= 2 and has_ids
        if requested == "tracking" and not tracking_capable:
            raise ValueError(
                f"{path}: tracking mode needs at least two frames and track IDs"
            )
        if requested == "reference_frame":
            return "reference_frame"
        return "tracking" if tracking_capable else "reference_frame"

    def _build_clips(self) -> list[ClipSpec]:
        clips: list[ClipSpec] = []
        interval_ms = 1000.0 / self.settings.dataset.target_fps
        for record_index, record in enumerate(self.records):
            if record.mode == "tracking":
                if len(record.frames) < self.clip_len:
                    raise ValueError(
                        f"{record.annotation_path}: tracking sequence has "
                        f"{len(record.frames)} frames, clip_len={self.clip_len}"
                    )
                for end in range(self.clip_len - 1, len(record.frames)):
                    start = end - self.clip_len + 1
                    selected = record.frames[start : end + 1]
                    clips.append(
                        ClipSpec(
                            record_index=record_index,
                            timestamps_ms=tuple(
                                frame.timestamp_ms for frame in selected
                            ),
                            annotation_indices=tuple(range(start, end + 1)),
                            sequence_start=True,
                        )
                    )
            else:
                for annotation_index, frame in enumerate(record.frames):
                    timestamps = tuple(
                        frame.timestamp_ms - interval_ms * offset
                        for offset in range(self.clip_len - 1, -1, -1)
                    )
                    if timestamps[0] < 0:
                        continue
                    clips.append(
                        ClipSpec(
                            record_index=record_index,
                            timestamps_ms=timestamps,
                            annotation_indices=(None,) * (self.clip_len - 1)
                            + (annotation_index,),
                            sequence_start=True,
                        )
                    )
        return clips

    def __len__(self) -> int:
        """Return the number of fixed-length clips."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Decode, jointly transform, and return one clip as step dictionaries."""
        spec = self.samples[index]
        record = self.records[spec.record_index]
        images = self.frame_reader.read(record.video_path, spec.timestamps_ms)
        targets = [
            self._target_for(
                record,
                annotation_index,
                image.size,
                image_id=index * self.clip_len + step,
            )
            for step, (annotation_index, image) in enumerate(
                zip(spec.annotation_indices, images, strict=True)
            )
        ]
        transformed = self._transform_clip(images, targets)
        steps: list[dict[str, Any]] = []
        supervision: list[bool] = []
        clip_sequence_id = (
            f"{record.sequence_id}@{spec.timestamps_ms[0]:.3f}"
            if record.mode == "tracking"
            else record.sequence_id
        )
        for step, ((image, target), timestamp_ms, annotation_index) in enumerate(
            zip(
                transformed,
                spec.timestamps_ms,
                spec.annotation_indices,
                strict=True,
            )
        ):
            target["image_size"] = torch.tensor(
                [image.shape[1], image.shape[2]], dtype=torch.int64
            )
            steps.append(
                {
                    "image": image,
                    "target": target,
                    "sequence_id": clip_sequence_id,
                    "frame_id": int(round(timestamp_ms)),
                    "timestamp": timestamp_ms / 1000.0,
                    "is_sequence_start": spec.sequence_start and step == 0,
                    "context_valid_mask": torch.zeros(0, dtype=torch.bool),
                    "context_time_offsets": torch.zeros(0, dtype=torch.float32),
                    "extras": {
                        "video": str(record.video_path),
                        "annotation": str(record.annotation_path),
                        "annotation_mode": record.mode,
                    },
                }
            )
            supervision.append(annotation_index is not None)
        return {
            "steps": steps,
            "supervision_mask": torch.tensor(supervision, dtype=torch.bool),
            "mode": record.mode,
        }

    def _target_for(
        self,
        record: VideoRecord,
        annotation_index: int | None,
        image_size: tuple[int, int],
        *,
        image_id: int,
    ) -> dict[str, Tensor]:
        width, height = image_size
        objects = (
            () if annotation_index is None else record.frames[annotation_index].objects
        )
        boxes: list[list[float]] = []
        labels: list[int] = []
        track_ids: list[int] = []
        for obj in objects:
            x1, y1, x2, y2 = obj["box"]
            x1 = min(max(x1, 0.0), float(width))
            x2 = min(max(x2, 0.0), float(width))
            y1 = min(max(y1, 0.0), float(height))
            y2 = min(max(y2, 0.0), float(height))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.settings.classes[obj["category"]])
            track_ids.append(int(obj["track_id"]))
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        area = (
            (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            if boxes_tensor.numel()
            else torch.empty(0, dtype=torch.float32)
        )
        return {
            "boxes": boxes_tensor,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "track_ids": torch.tensor(track_ids, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
            "size": torch.tensor([height, width], dtype=torch.int64),
        }

    def _transform_clip(
        self,
        images: list[Image.Image],
        targets: list[dict[str, Tensor]],
    ) -> list[tuple[Tensor, dict[str, Tensor]]]:
        """Replay identical random choices for every frame in the clip."""
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        next_states: tuple[Any, Any, Tensor] | None = None
        transformed: list[tuple[Tensor, dict[str, Tensor]]] = []
        for image, target in zip(images, targets, strict=True):
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            image_tensor, transformed_target = self.transform(image, target)
            transformed.append((image_tensor, transformed_target))
            if next_states is None:
                next_states = (
                    random.getstate(),
                    np.random.get_state(),
                    torch.random.get_rng_state(),
                )
        if next_states is not None:
            random.setstate(next_states[0])
            np.random.set_state(next_states[1])
            torch.random.set_rng_state(next_states[2])
        return transformed

    def manifest(self) -> list[dict[str, Any]]:
        """Return exact clip provenance for experiment artifacts."""
        result: list[dict[str, Any]] = []
        for sample in self.samples:
            record = self.records[sample.record_index]
            result.append(
                {
                    "video": str(record.video_path),
                    "annotation": str(record.annotation_path),
                    "sequence_id": record.sequence_id,
                    "mode": record.mode,
                    "timestamps_ms": list(sample.timestamps_ms),
                    "supervised": [
                        value is not None for value in sample.annotation_indices
                    ],
                }
            )
        return result


# Backwards-compatible name used by the original notebook.
VideoDataset = VideoClipDataset
