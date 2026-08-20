"""Validated configuration for the directory-backed image dataset component."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Settings(BaseModel):
    """Reject unknown component options instead of silently ignoring typos."""

    model_config = ConfigDict(extra="forbid")


class ImageSizeSettings(_Settings):
    """Square model input resolution before the experiment overrides it."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)


class DatasetSettings(_Settings):
    """Locations, annotation reader, and RF-DETR preprocessing controls."""

    images_dir: str = Field(min_length=1)
    annotations_dir: str = Field(min_length=1)
    annotation_format: str = Field("bdd100k_json", min_length=1)
    annotation_extension: str = Field(".json", pattern=r"^\.[^.]+$")
    image_size: ImageSizeSettings
    patch_size: int = Field(16, gt=0)
    num_windows: int = Field(2, gt=0)
    multi_scale: bool = False
    expanded_scales: bool = False
    skip_random_resize: bool = False
    scale_jitter: bool = True
    aug_config: dict[str, Any] | None = None
    image_extensions: list[str] = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png"], min_length=1
    )
    recursive: bool = False
    strict_pairs: bool = True

    @model_validator(mode="after")
    def _validate_extensions(self) -> DatasetSettings:
        normalized: list[str] = []
        for extension in self.image_extensions:
            value = extension.lower()
            if not value.startswith(".") or len(value) == 1:
                raise ValueError("dataset.image_extensions must contain '.ext' values")
            if value not in normalized:
                normalized.append(value)
        self.image_extensions = normalized
        return self


class SplitSettings(_Settings):
    """Generated fractions or names of already materialized split directories."""

    mode: Literal["generated", "predefined"] = "generated"
    train_fraction: float = Field(0.8, ge=0.0, le=1.0)
    validation_fraction: float = Field(0.2, ge=0.0, le=1.0)
    test_fraction: float = Field(0.0, ge=0.0, le=1.0)
    train_dir: str = Field("train", min_length=1)
    validation_dir: str = Field("val", min_length=1)
    test_dir: str = Field("test", min_length=1)

    @model_validator(mode="after")
    def _validate_fractions(self) -> SplitSettings:
        if self.mode == "generated":
            total = self.train_fraction + self.validation_fraction + self.test_fraction
            if abs(total - 1.0) > 1e-8:
                raise ValueError("generated split fractions must sum to 1.0")
            if self.train_fraction == 0.0 or self.validation_fraction == 0.0:
                raise ValueError(
                    "generated train and validation fractions must be positive"
                )
        names = (self.train_dir, self.validation_dir, self.test_dir)
        if len(set(names)) != len(names):
            raise ValueError("predefined split directory names must be distinct")
        return self

    def fraction(self, split: str) -> float:
        """Return the configured fraction for a normalized project split."""
        return {
            "train": self.train_fraction,
            "validation": self.validation_fraction,
            "test": self.test_fraction,
        }[split]

    def directory(self, split: str) -> str:
        """Return the child directory used by a predefined split."""
        return {
            "train": self.train_dir,
            "validation": self.validation_dir,
            "test": self.test_dir,
        }[split]


class DataLoaderSettings(_Settings):
    """Standalone loader defaults retained for notebook compatibility."""

    batch_size: int = Field(1, gt=0)
    shuffle: bool = True
    num_workers: int = Field(4, ge=0)
    return_targets: bool = True


class ClassBalanceSettings(_Settings):
    """Control multilabel split stratification and train-only resampling."""

    stratify_generated: bool = False
    sampling: Literal["none", "inverse_sqrt", "inverse_frequency"] = "none"
    max_sample_weight: float = Field(5.0, ge=1.0)


class ImageDataLoaderSettings(_Settings):
    """Complete self-contained component configuration."""

    name: Literal["image_dataloader"]
    dataset: DatasetSettings
    splits: SplitSettings = Field(default_factory=SplitSettings)
    dataloader: DataLoaderSettings = Field(default_factory=DataLoaderSettings)
    class_balance: ClassBalanceSettings = Field(default_factory=ClassBalanceSettings)
    classes: dict[str, int] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_classes(self) -> ImageDataLoaderSettings:
        class_ids = list(self.classes.values())
        if any(isinstance(value, bool) for value in class_ids):
            raise ValueError("class IDs must be integers")
        if set(class_ids) != set(range(len(class_ids))):
            raise ValueError("class IDs must be contiguous and start at zero")
        return self
