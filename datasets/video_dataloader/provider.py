"""Project dataset protocol for fixed-length video clips."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from context_detection.config import ExperimentConfig
from context_detection.contracts import DetectionClipBatch
from context_detection.data.collate import DetectionCollator
from context_detection.data.protocols import DatasetSplit
from context_detection.models.rfdetr import rfdetr_pretrained_resolution

_COMPONENT_ROOT = Path(__file__).resolve().parent


def _load_local_module(filename: str, module_name: str) -> ModuleType:
    """Load a sibling without colliding with other directory components."""
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = _COMPONENT_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import video_dataloader module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


_settings = _load_local_module("settings.py", "context_detection_video_settings")
_dataset = _load_local_module("dataset.py", "context_detection_video_dataset")
VideoDataLoaderSettings = _settings.VideoDataLoaderSettings
VideoClipDataset = _dataset.VideoClipDataset


class VideoClipCollator:
    """Transpose sample-major clips into time-major project batch contracts."""

    def __init__(self) -> None:
        self.step_collator = DetectionCollator()

    def __call__(self, samples: list[dict[str, Any]]) -> DetectionClipBatch:
        """Collate fixed-length clips without mixing batch and time axes."""
        if not samples:
            raise ValueError("cannot collate an empty video batch")
        clip_lengths = {len(sample["steps"]) for sample in samples}
        if len(clip_lengths) != 1:
            raise ValueError(f"video clips have different lengths: {clip_lengths}")
        modes = {str(sample["mode"]) for sample in samples}
        if len(modes) != 1:
            raise ValueError(f"cannot mix annotation modes in one batch: {modes}")

        clip_len = clip_lengths.pop()
        steps = [
            self.step_collator([sample["steps"][step] for sample in samples])
            for step in range(clip_len)
        ]
        supervision_mask = torch.stack(
            [sample["supervision_mask"] for sample in samples], dim=1
        ).to(device=steps[0][0].images.device, dtype=torch.bool)
        return DetectionClipBatch(
            steps=steps,
            supervision_mask=supervision_mask,
            mode=modes.pop(),
        )


class VideoDataLoaderProtocol:
    """Build RF-DETR-ready clip DataLoaders from independent directory roots."""

    def build(self, config: ExperimentConfig, split: DatasetSplit) -> DataLoader[Any]:
        """Build one deterministic project-standard DataLoader endpoint."""
        print(
            f"[video_dataloader] loading component settings for split={split.value} "
            f"from {config.data.config_path}",
            flush=True,
        )
        settings = self._load_settings(Path(config.data.config_path))
        if "image_size" in config.data.model_fields_set:
            raise ValueError(
                "data.image_size must not be set for RF-DETR; the pretrained "
                "variant owns its input resolution"
            )
        variant = config.detector.variant
        if config.detector.name != "rfdetr" or variant is None:
            raise ValueError("video_dataloader requires an RF-DETR pretrained variant")
        resolution = rfdetr_pretrained_resolution(variant)
        self._validate_resolution(settings, resolution)
        videos_root, annotations_root = self._resolved_roots(
            settings, Path(config.data.config_path).parent, split.value
        )
        dataset = VideoClipDataset(
            videos_root,
            annotations_root,
            settings,
            split=split.value,
            split_names=settings.splits.names(split.value),
            clip_len=config.data.clip_len,
            resolution=resolution,
        )
        workers = config.train.num_workers
        loader = DataLoader(
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
            collate_fn=VideoClipCollator(),
            generator=torch.Generator().manual_seed(config.train.seed),
        )
        loader.video_manifest = dataset.manifest()  # type: ignore[attr-defined]
        print(
            f"[video_dataloader] DataLoader ready for split={split.value}: "
            f"samples={len(dataset)}, batches={len(loader)}, workers={workers}",
            flush=True,
        )
        return loader

    @staticmethod
    def _load_settings(path: Path) -> Any:
        raw: Any = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if not isinstance(raw, dict):
            raise TypeError("video_dataloader config must be a mapping")
        return VideoDataLoaderSettings.model_validate(raw)

    @staticmethod
    def _validate_resolution(settings: Any, resolution: int) -> None:
        block_size = settings.dataset.patch_size * settings.dataset.num_windows
        if resolution % block_size:
            raise ValueError(
                "RF-DETR image_size must be divisible by patch_size * num_windows "
                f"({block_size}), got {resolution}"
            )

    @staticmethod
    def _resolved_roots(
        settings: Any, component_root: Path, split: str
    ) -> tuple[Path, Path]:
        def resolve(value: str) -> Path:
            path = Path(value)
            return path if path.is_absolute() else (component_root / path).resolve()

        videos = settings.dataset.split_videos_dirs.get(
            split, settings.dataset.videos_dir
        )
        annotations = settings.dataset.split_annotations_dirs.get(
            split, settings.dataset.annotations_dir
        )
        return resolve(videos), resolve(annotations)


PROTOCOL = VideoDataLoaderProtocol()
