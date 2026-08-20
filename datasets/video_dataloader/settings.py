"""Validated settings for the directory-backed video dataset component."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Settings(BaseModel):
    """Reject unknown options instead of silently accepting misspellings."""

    model_config = ConfigDict(extra="forbid")


class VideoDatasetSettings(_Settings):
    """Locations, annotation semantics, sampling, and RF-DETR preprocessing."""

    videos_dir: str = Field(min_length=1)
    annotations_dir: str = Field(min_length=1)
    split_videos_dirs: dict[str, str] = Field(default_factory=dict)
    split_annotations_dirs: dict[str, str] = Field(default_factory=dict)
    max_sequences: dict[str, int] = Field(default_factory=dict)
    annotation_extension: str = Field(".json", pattern=r"^\.[^.]+$")
    video_extensions: list[str] = Field(
        default_factory=lambda: [".mp4", ".mov", ".avi", ".mkv"],
        min_length=1,
    )
    recursive: bool = True
    strict_pairs: bool = True
    annotation_mode: Literal["auto", "tracking", "reference_frame"] = "auto"
    target_fps: float = Field(5.0, gt=0.0)
    max_seek_error_ms: float = Field(100.0, ge=0.0)
    auto_rotate: bool = True
    patch_size: int = Field(16, gt=0)
    num_windows: int = Field(2, gt=0)
    multi_scale: bool = False
    expanded_scales: bool = False
    skip_random_resize: bool = False
    scale_jitter: bool = True
    aug_config: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _normalize_extensions(self) -> VideoDatasetSettings:
        normalized: list[str] = []
        for extension in self.video_extensions:
            value = extension.lower()
            if not value.startswith(".") or len(value) == 1:
                raise ValueError("dataset.video_extensions must contain '.ext' values")
            if value not in normalized:
                normalized.append(value)
        self.video_extensions = normalized
        return self


class SplitSettings(_Settings):
    """Names used to discover predefined splits at arbitrary depth."""

    train_names: list[str] = Field(default_factory=lambda: ["train"], min_length=1)
    validation_names: list[str] = Field(
        default_factory=lambda: ["val", "validation", "valid"], min_length=1
    )
    test_names: list[str] = Field(default_factory=lambda: ["test"], min_length=1)

    @model_validator(mode="after")
    def _validate_names(self) -> SplitSettings:
        groups = (
            self.train_names,
            self.validation_names,
            self.test_names,
        )
        normalized = [[value.strip().lower() for value in group] for group in groups]
        if any(not value for group in normalized for value in group):
            raise ValueError("split names cannot be blank")
        flattened = [value for group in normalized for value in group]
        if len(flattened) != len(set(flattened)):
            raise ValueError("split aliases must be unique across train/val/test")
        self.train_names, self.validation_names, self.test_names = normalized
        return self

    def names(self, split: str) -> frozenset[str]:
        """Return directory aliases for a normalized project split."""
        return frozenset(
            {
                "train": self.train_names,
                "validation": self.validation_names,
                "test": self.test_names,
            }[split]
        )


class StandaloneDataLoaderSettings(_Settings):
    """Defaults used only by the backwards-compatible standalone helper."""

    batch_size: int = Field(1, gt=0)
    shuffle: bool = True
    num_workers: int = Field(0, ge=0)


class VideoDataLoaderSettings(_Settings):
    """Complete self-contained video component configuration."""

    name: Literal["video_dataloader"]
    dataset: VideoDatasetSettings
    splits: SplitSettings = Field(default_factory=SplitSettings)
    dataloader: StandaloneDataLoaderSettings = Field(
        default_factory=StandaloneDataLoaderSettings
    )
    classes: dict[str, int] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_classes(self) -> VideoDataLoaderSettings:
        class_ids = list(self.classes.values())
        if any(isinstance(value, bool) for value in class_ids):
            raise ValueError("class IDs must be integers")
        if set(class_ids) != set(range(len(class_ids))):
            raise ValueError("class IDs must be contiguous and start at zero")
        return self
