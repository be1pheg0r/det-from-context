"""Extensible readers for per-image object-detection annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class DetectionAnnotation:
    """Validated absolute ``xyxy`` boxes and contiguous class labels."""

    boxes: list[list[float]]
    labels: list[int]
    areas: list[float]
    rejected_objects: int = 0


@runtime_checkable
class AnnotationReader(Protocol):
    """Parse one annotation file without knowing anything about model transforms."""

    def read(
        self,
        path: Path,
        classes: dict[str, int],
        image_size: tuple[int, int],
    ) -> DetectionAnnotation:
        """Return boxes in absolute pixel ``xyxy`` coordinates."""


class BDD100KJsonReader:
    """Read the first BDD100K frame from a per-image JSON annotation file."""

    def read(
        self,
        path: Path,
        classes: dict[str, int],
        image_size: tuple[int, int],
    ) -> DetectionAnnotation:
        """Parse, clamp, and filter malformed BDD100K objects."""
        with path.open(encoding="utf-8") as stream:
            annotation = json.load(stream)
        if not isinstance(annotation, dict):
            raise ValueError(f"{path}: annotation root must be a mapping")

        frames = annotation.get("frames", [])
        frame = frames[0] if isinstance(frames, list) and frames else {}
        objects = frame.get("objects", []) if isinstance(frame, dict) else []
        if not isinstance(objects, list):
            raise ValueError(f"{path}: frames[0].objects must be a list")

        width, height = image_size
        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        rejected = 0
        for obj in objects:
            parsed = self._parse_object(obj, classes, width, height)
            if parsed is None:
                rejected += 1
                continue
            box, label, area = parsed
            boxes.append(box)
            labels.append(label)
            areas.append(area)
        return DetectionAnnotation(boxes, labels, areas, rejected)

    @staticmethod
    def _parse_object(
        obj: object,
        classes: dict[str, int],
        width: int,
        height: int,
    ) -> tuple[list[float], int, float] | None:
        """Return one valid object or ``None`` for ignored/malformed input."""
        if not isinstance(obj, dict):
            return None
        category = obj.get("category")
        if not isinstance(category, str) or category not in classes:
            return None
        raw_box = obj.get("box2d")
        if not isinstance(raw_box, dict):
            return None
        try:
            x1, y1, x2, y2 = (float(raw_box[key]) for key in ("x1", "y1", "x2", "y2"))
        except (KeyError, TypeError, ValueError):
            return None
        if not all(isfinite(value) for value in (x1, y1, x2, y2)):
            return None
        x1 = min(max(x1, 0.0), width)
        x2 = min(max(x2, 0.0), width)
        y1 = min(max(y1, 0.0), height)
        y2 = min(max(y2, 0.0), height)
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2], classes[category], (x2 - x1) * (y2 - y1)


_READERS: dict[str, AnnotationReader] = {"bdd100k_json": BDD100KJsonReader()}


def register_annotation_reader(name: str, reader: AnnotationReader) -> None:
    """Register a project-local annotation format without changing the dataset."""
    if not name:
        raise ValueError("annotation reader name must not be empty")
    if not isinstance(reader, AnnotationReader):
        raise TypeError(
            "annotation reader must implement read(path, classes, image_size)"
        )
    _READERS[name] = reader


def get_annotation_reader(name: str) -> AnnotationReader:
    """Resolve a configured reader with an actionable error message."""
    try:
        return _READERS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown annotation format {name!r}; available: {sorted(_READERS)}"
        ) from error
