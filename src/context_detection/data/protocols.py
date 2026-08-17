"""Расширяемый dataset protocol с endpoint в виде DataLoader."""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum, StrEnum
from typing import Any, ClassVar, Protocol, runtime_checkable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ..config import ExperimentConfig
from ..registry import NEEDS_CONTEXT_FRAMES, register_dataset_name
from .collate import collate_fn
from .datasets import (
    SequenceDetectionDataset,
    build_imagenet_vid,
    build_ovis,
)
from .sequence_index import SequenceIndex


class DatasetSplit(StrEnum):
    """Поддерживаемые логические части датасета."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

    @classmethod
    def parse(cls, value: str | DatasetSplit) -> DatasetSplit:
        """Нормализовать общеупотребимый ``val`` в ``validation``."""
        if value == "val":
            return cls.VALIDATION
        return cls(value)


@runtime_checkable
class DatasetProtocol(Protocol):
    """Структурный интерфейс провайдера данных.

    Реализация может использовать любой ``torch.utils.data.Dataset``, sampler
    и collator, но наружу всегда возвращает готовый ``DataLoader``.
    """

    def build(
        self,
        config: ExperimentConfig,
        split: DatasetSplit,
    ) -> DataLoader[Any]:
        """Построить DataLoader для одного split."""


class DatasetProtocolRegistry:
    """Реестр dataset protocols без центрального условного оператора."""

    def __init__(self) -> None:
        self._protocols: dict[str, DatasetProtocol] = {}

    @property
    def names(self) -> frozenset[str]:
        """Зарегистрированные имена датасетов."""
        return frozenset(self._protocols)

    def register(
        self,
        name: str,
        protocol: DatasetProtocol,
        *,
        replace: bool = False,
    ) -> None:
        """Зарегистрировать структурно совместимый provider."""
        if not name.strip():
            raise ValueError("имя dataset protocol не может быть пустым")
        if not isinstance(protocol, DatasetProtocol):
            raise TypeError("dataset protocol обязан реализовать build(config, split)")
        if name in self._protocols and not replace:
            raise ValueError(f"dataset protocol {name!r} уже зарегистрирован")
        self._protocols[name] = protocol
        register_dataset_name(name)

    def get(self, name: str) -> DatasetProtocol:
        """Получить provider с информативной ошибкой для неизвестного имени."""
        try:
            return self._protocols[name]
        except KeyError as error:
            raise ValueError(
                f"нет dataset protocol {name!r}, доступно: {sorted(self.names)}"
            ) from error

    def build(
        self,
        config: ExperimentConfig,
        split: str | DatasetSplit,
    ) -> DataLoader[Any]:
        """Построить и проверить публичный endpoint."""
        loader: DataLoader[Any] = self.get(config.data.name).build(
            config,
            DatasetSplit.parse(split),
        )
        if not isinstance(loader, DataLoader):
            raise TypeError(
                f"dataset protocol {config.data.name!r} вернул "
                f"{type(loader).__name__}, ожидался DataLoader"
            )
        return loader


class DummyDatasetDefaults(IntEnum):
    """Небольшие размеры для быстрого CPU pipeline."""

    BATCHES = 2
    BOXES = 3


class DummyDetectionDataset(Dataset[dict[str, Any]]):
    """Детерминированный Dataset, совместимый с текущими batch-контрактами."""

    _FRAME_SECONDS: ClassVar[float] = 0.04

    def __init__(
        self,
        config: ExperimentConfig,
        split: DatasetSplit,
        length: int,
    ) -> None:
        self.config: ExperimentConfig = config
        self.split: DatasetSplit = split
        self.length: int = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, position: int) -> dict[str, Any]:
        if position < 0 or position >= self.length:
            raise IndexError(position)
        seed_offset: int = list(DatasetSplit).index(self.split) * self.length
        generator: torch.Generator = torch.Generator().manual_seed(
            self.config.train.seed + seed_offset + position
        )
        image_size: int = self.config.data.image_size
        image: Tensor = torch.rand(3, image_size, image_size, generator=generator)
        boxes: Tensor = (
            torch.rand(
                DummyDatasetDefaults.BOXES,
                4,
                generator=generator,
            )
            * 0.5
            + 0.25
        )
        labels: Tensor = torch.randint(
            self.config.detector.num_classes,
            (DummyDatasetDefaults.BOXES,),
            generator=generator,
        )
        frame_id: int = position % self.config.data.clip_len
        context = self._make_context(frame_id, generator)
        return {
            "image": image,
            "target": {"boxes": boxes, "labels": labels},
            "sequence_id": f"{self.split}-seq-{position // self.config.data.clip_len}",
            "frame_id": frame_id,
            "timestamp": frame_id * self._FRAME_SECONDS,
            "is_sequence_start": frame_id == 0,
            **context,
        }

    def _make_context(
        self,
        frame_id: int,
        generator: torch.Generator,
    ) -> dict[str, Tensor | None]:
        slots: int = self.config.data.context_k
        valid_mask: Tensor = torch.zeros(slots, dtype=torch.bool)
        time_offsets: Tensor = torch.zeros(slots, dtype=torch.float32)
        if not slots:
            return {
                "context_images": None,
                "context_valid_mask": valid_mask,
                "context_time_offsets": time_offsets,
            }
        valid_count: int = min(frame_id, slots)
        first_valid: int = slots - valid_count
        if valid_count:
            valid_mask[first_valid:] = True
            time_offsets[first_valid:] = (
                torch.arange(valid_count, 0, -1, dtype=torch.float32)
                * self._FRAME_SECONDS
            )
        image_size: int = self.config.data.image_size
        context_images: Tensor = torch.zeros(slots, 3, image_size, image_size)
        if valid_count:
            context_images[first_valid:] = torch.rand(
                valid_count,
                3,
                image_size,
                image_size,
                generator=generator,
            )
        return {
            "context_images": context_images,
            "context_valid_mask": valid_mask,
            "context_time_offsets": time_offsets,
        }


class DummyDatasetProtocol:
    """Встроенный provider для быстрого сквозного pipeline."""

    def build(
        self,
        config: ExperimentConfig,
        split: DatasetSplit,
    ) -> DataLoader[Any]:
        batch_size: int = (
            config.train.batch_size
            if split is DatasetSplit.TRAIN
            else config.validation.batch_size
        )
        dataset: DummyDetectionDataset = DummyDetectionDataset(
            config,
            split,
            length=batch_size * DummyDatasetDefaults.BATCHES,
        )
        carries_state: bool = (
            config.context.name != "none"
            and config.context.name not in NEEDS_CONTEXT_FRAMES
        )
        generator: torch.Generator = torch.Generator().manual_seed(config.train.seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split is DatasetSplit.TRAIN and not carries_state,
            num_workers=config.train.num_workers,
            collate_fn=collate_fn,
            generator=generator,
        )


IndexBuilder = Callable[[str, str], SequenceIndex]


class SequenceDatasetProtocol:
    """Адаптер существующего sequence stack к DataLoader endpoint."""

    def __init__(self, index_builder: IndexBuilder) -> None:
        self.index_builder: IndexBuilder = index_builder

    def build(
        self,
        config: ExperimentConfig,
        split: DatasetSplit,
    ) -> DataLoader[Any]:
        if config.data.root is None:
            raise ValueError(f"датасету {config.data.name!r} нужен root")
        index: SequenceIndex = self.index_builder(config.data.root, split.value)
        dataset: SequenceDetectionDataset = SequenceDetectionDataset(
            index=index,
            context_k=config.data.context_k,
            strategy=config.data.context_strategy,
        )
        batch_size: int = (
            config.train.batch_size
            if split is DatasetSplit.TRAIN
            else config.validation.batch_size
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.train.num_workers,
            collate_fn=collate_fn,
        )


DATASET_PROTOCOLS: DatasetProtocolRegistry = DatasetProtocolRegistry()
DATASET_PROTOCOLS.register("dummy", DummyDatasetProtocol())
DATASET_PROTOCOLS.register(
    "imagenet_vid",
    SequenceDatasetProtocol(build_imagenet_vid),
)
DATASET_PROTOCOLS.register("ovis", SequenceDatasetProtocol(build_ovis))


def register_dataset_protocol(
    name: str,
    protocol: DatasetProtocol,
    *,
    replace: bool = False,
) -> None:
    """Публичная точка расширения реестра датасетов."""
    DATASET_PROTOCOLS.register(name, protocol, replace=replace)


def build_dataloader(
    config: ExperimentConfig,
    split: str | DatasetSplit,
) -> DataLoader[Any]:
    """Построить DataLoader через зарегистрированный protocol."""
    return DATASET_PROTOCOLS.build(config, split)
