"""Tests for clip construction, video annotation modes, and collation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from torch import Tensor

from context_detection.contracts import DetectionClipBatch

_COMPONENT_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "video_dataloader"
_PROVIDER_PATH = _COMPONENT_ROOT / "provider.py"
_SPEC = importlib.util.spec_from_file_location("test_video_provider", _PROVIDER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(_PROVIDER_PATH)
_PROVIDER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PROVIDER
_SPEC.loader.exec_module(_PROVIDER)

VideoClipCollator = _PROVIDER.VideoClipCollator
VideoClipDataset = _PROVIDER.VideoClipDataset
VideoDataLoaderSettings = _PROVIDER.VideoDataLoaderSettings


class _FakeFrameReader:
    """Return deterministic synthetic RGB frames and record requested times."""

    def __init__(self) -> None:
        self.requests: list[tuple[float, ...]] = []

    def read(self, path: Path, timestamps_ms: tuple[float, ...]) -> list[Image.Image]:
        self.requests.append(timestamps_ms)
        return [
            Image.new("RGB", (20, 10), color=(int(timestamp) % 255, 20, 30))
            for timestamp in timestamps_ms
        ]


def _transform(
    image: Image.Image, target: dict[str, Tensor]
) -> tuple[Tensor, dict[str, Tensor]]:
    """Normalize xyxy boxes while preserving all per-object target fields."""
    width, height = image.size
    boxes = target["boxes"].clone()
    if boxes.numel():
        x1, y1, x2, y2 = boxes.unbind(-1)
        boxes = torch.stack(
            (
                (x1 + x2) / (2 * width),
                (y1 + y2) / (2 * height),
                (x2 - x1) / width,
                (y2 - y1) / height,
            ),
            dim=-1,
        )
    transformed = dict(target)
    transformed["boxes"] = boxes
    transformed["size"] = torch.tensor([height, width])
    pixel = torch.rand(())
    return torch.full((3, height, width), pixel), transformed


def _settings(
    videos_dir: Path,
    annotations_dir: Path,
    *,
    annotation_mode: str = "auto",
    strict_pairs: bool = True,
) -> Any:
    return VideoDataLoaderSettings.model_validate(
        {
            "name": "video_dataloader",
            "dataset": {
                "videos_dir": str(videos_dir),
                "annotations_dir": str(annotations_dir),
                "video_extensions": [".mov"],
                "annotation_mode": annotation_mode,
                "strict_pairs": strict_pairs,
                "target_fps": 5,
                "patch_size": 16,
                "num_windows": 2,
            },
            "splits": {},
            "dataloader": {"batch_size": 1, "num_workers": 0},
            "classes": {"car": 0, "person": 1},
        }
    )


def _write_pair(
    root: Path,
    stem: str,
    frames: list[dict[str, Any]],
    *,
    split: str = "train",
) -> tuple[Path, Path]:
    videos = root / "videos" / "site" / split / "camera"
    annotations = root / "annotations" / "export" / split
    videos.mkdir(parents=True, exist_ok=True)
    annotations.mkdir(parents=True, exist_ok=True)
    video = videos / f"{stem}.mov"
    annotation = annotations / f"{stem}.json"
    video.touch()
    annotation.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    return root / "videos", root / "annotations"


def _obj(identifier: str | None = "vehicle-42") -> dict[str, Any]:
    result: dict[str, Any] = {
        "category": "car",
        "box2d": {"x1": 2, "y1": 1, "x2": 10, "y2": 7},
    }
    if identifier is not None:
        result["id"] = identifier
    return result


def test_tracking_annotations_build_sliding_supervised_clips(tmp_path: Path) -> None:
    frames = [
        {"timestamp": timestamp, "objects": [_obj()]} for timestamp in (0, 200, 400)
    ]
    videos, annotations = _write_pair(tmp_path, "sequence", frames)
    reader = _FakeFrameReader()
    dataset = VideoClipDataset(
        videos,
        annotations,
        _settings(videos, annotations),
        split="train",
        split_names=frozenset({"train"}),
        clip_len=2,
        resolution=32,
        frame_reader=reader,
        transform=_transform,
    )

    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["mode"] == "tracking"
    assert sample["supervision_mask"].tolist() == [True, True]
    assert [step["target"]["track_ids"].tolist() for step in sample["steps"]] == [
        [0],
        [0],
    ]
    assert reader.requests == [(0.0, 200.0)]
    assert torch.equal(sample["steps"][0]["image"], sample["steps"][1]["image"])


def test_reference_frame_mode_warms_memory_without_fake_labels(tmp_path: Path) -> None:
    frames = [
        {"timestamp": timestamp, "objects": [_obj(None)]} for timestamp in (1000, 1200)
    ]
    videos, annotations = _write_pair(tmp_path, "reference", frames)
    reader = _FakeFrameReader()
    dataset = VideoClipDataset(
        videos,
        annotations,
        _settings(videos, annotations),
        split="train",
        split_names=frozenset({"train"}),
        clip_len=3,
        resolution=32,
        frame_reader=reader,
        transform=_transform,
    )

    first, second = dataset[0], dataset[1]
    batch = VideoClipCollator()([first, second])

    assert isinstance(batch, DetectionClipBatch)
    assert batch.mode == "reference_frame"
    assert batch.clip_len == 3 and batch.batch_size == 2
    assert batch.supervision_mask.tolist() == [
        [False, False],
        [False, False],
        [True, True],
    ]
    assert reader.requests[0] == (600.0, 800.0, 1000.0)
    assert batch.steps[0][0].targets is not None
    assert batch.steps[0][0].targets[0]["boxes"].shape == (0, 4)
    assert batch.steps[-1][0].targets[0]["boxes"].shape == (1, 4)


def test_split_discovery_and_strict_pairing_are_independent_between_roots(
    tmp_path: Path,
) -> None:
    frames = [{"timestamp": 1000, "objects": [_obj(None)]}]
    videos, annotations = _write_pair(tmp_path, "paired", frames, split="val")
    train_dir = videos / "elsewhere" / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "missing.mov").touch()

    dataset = VideoClipDataset(
        videos,
        annotations,
        _settings(videos, annotations),
        split="validation",
        split_names=frozenset({"val", "validation"}),
        clip_len=1,
        resolution=32,
        frame_reader=_FakeFrameReader(),
        transform=_transform,
    )
    assert len(dataset) == 1

    with pytest.raises(ValueError, match="missing annotations"):
        VideoClipDataset(
            videos,
            annotations,
            _settings(videos, annotations),
            split="train",
            split_names=frozenset({"train"}),
            clip_len=1,
            resolution=32,
            frame_reader=_FakeFrameReader(),
            transform=_transform,
        )


def test_non_strict_pairing_uses_first_duplicate_video_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frames = [{"timestamp": 1000, "objects": [_obj(None)]}]
    videos, annotations = _write_pair(tmp_path, "duplicate", frames)
    duplicate_dir = videos / "second" / "train"
    duplicate_dir.mkdir(parents=True)
    (duplicate_dir / "duplicate.mov").touch()

    dataset = VideoClipDataset(
        videos,
        annotations,
        _settings(videos, annotations, strict_pairs=False),
        split="train",
        split_names=frozenset({"train"}),
        clip_len=1,
        resolution=32,
        frame_reader=_FakeFrameReader(),
        transform=_transform,
    )

    assert len(dataset) == 1
    assert "ignoring duplicate videos paths for 1 stems" in capsys.readouterr().out


def test_forced_tracking_rejects_reference_only_annotations(tmp_path: Path) -> None:
    videos, annotations = _write_pair(
        tmp_path,
        "reference",
        [{"timestamp": 1000, "objects": [_obj(None)]}],
    )

    with pytest.raises(ValueError, match="tracking mode needs"):
        VideoClipDataset(
            videos,
            annotations,
            _settings(videos, annotations, annotation_mode="tracking"),
            split="train",
            split_names=frozenset({"train"}),
            clip_len=1,
            resolution=32,
            frame_reader=_FakeFrameReader(),
            transform=_transform,
        )
