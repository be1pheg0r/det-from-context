"""Standalone helper for inspecting the video component outside an experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from provider import (
    VideoClipCollator,
    VideoClipDataset,
    VideoDataLoaderProtocol,
    VideoDataLoaderSettings,
)
from torch.utils.data import DataLoader


def create_video_dataloader(
    config: str | Path | dict[str, Any],
    *,
    split: str = "train",
    clip_len: int = 4,
    resolution: int = 560,
) -> DataLoader[Any]:
    """Create a clip loader directly from a component YAML or mapping."""
    if isinstance(config, dict):
        raw = config
        component_root = Path.cwd()
    else:
        config_path = Path(config).resolve()
        raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        component_root = config_path.parent
    if not isinstance(raw, dict):
        raise TypeError("video_dataloader config must be a mapping")
    settings = VideoDataLoaderSettings.model_validate(raw)
    VideoDataLoaderProtocol._validate_resolution(settings, resolution)
    videos_root, annotations_root = VideoDataLoaderProtocol._resolved_roots(
        settings, component_root
    )
    dataset = VideoClipDataset(
        videos_root,
        annotations_root,
        settings,
        split=split,
        split_names=settings.splits.names(split),
        clip_len=clip_len,
        resolution=resolution,
    )
    return DataLoader(
        dataset,
        batch_size=settings.dataloader.batch_size,
        shuffle=settings.dataloader.shuffle and split == "train",
        num_workers=settings.dataloader.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=settings.dataloader.num_workers > 0,
        collate_fn=VideoClipCollator(),
    )
