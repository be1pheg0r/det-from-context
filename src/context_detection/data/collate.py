"""Сборка dataset samples в существующие контракты модели."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import Tensor

from ..contracts import ContextBatch, DetectionBatch


class DetectionCollator:
    """Преобразует список sample-словарей в пару проектных batch-контрактов.

    Каждый sample обязан содержать ``image``, ``target``, ``sequence_id``,
    ``frame_id``, ``timestamp``, ``is_sequence_start``, ``context_valid_mask``
    и ``context_time_offsets``. Контекстные изображения и произвольные extras
    опциональны. Разное число контекстных слотов дополняется невалидными
    слотами до максимума внутри батча.
    """

    _REQUIRED_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "image",
            "target",
            "sequence_id",
            "frame_id",
            "timestamp",
            "is_sequence_start",
            "context_valid_mask",
            "context_time_offsets",
        }
    )

    def __call__(
        self,
        samples: list[dict[str, Any]],
    ) -> tuple[DetectionBatch, ContextBatch]:
        """Собрать непустой список samples.

        Args:
            samples: Отдельные элементы dataset, ещё не объединённые в batch.

        Returns:
            Совместимые с ``ContextDetector`` target и context batches.
        """
        if not samples:
            raise ValueError("нельзя собрать пустой batch")
        for index, sample in enumerate(samples):
            missing: set[str] = self._REQUIRED_KEYS - sample.keys()
            if missing:
                raise ValueError(
                    f"sample[{index}] не содержит ключи: {sorted(missing)}"
                )

        images: Tensor = self._stack_images(samples)
        valid_mask, time_offsets, context_images = self._collate_context(samples)
        device: torch.device = images.device
        detection = DetectionBatch(
            images=images,
            targets=[sample["target"] for sample in samples],
            sequence_id=[str(sample["sequence_id"]) for sample in samples],
            frame_id=torch.tensor(
                [int(sample["frame_id"]) for sample in samples],
                dtype=torch.int64,
                device=device,
            ),
            timestamp=torch.tensor(
                [float(sample["timestamp"]) for sample in samples],
                dtype=torch.float32,
                device=device,
            ),
            is_sequence_start=torch.tensor(
                [bool(sample["is_sequence_start"]) for sample in samples],
                dtype=torch.bool,
                device=device,
            ),
        )
        context_targets: list[list[dict[str, Tensor]]] | None = None
        if any("context_targets" in sample for sample in samples):
            context_targets = [sample.get("context_targets", []) for sample in samples]
        context = ContextBatch(
            images=context_images,
            valid_mask=valid_mask,
            time_offsets=time_offsets,
            targets=context_targets,
            extras={"samples": [sample.get("extras", {}) for sample in samples]},
        )
        return detection, context

    @staticmethod
    def _stack_images(samples: list[dict[str, Any]]) -> Tensor:
        images: list[Tensor] = [sample["image"] for sample in samples]
        shapes: set[tuple[int, ...]] = {tuple(image.shape) for image in images}
        if len(shapes) != 1:
            raise ValueError(
                "изображения разных размеров требуют dataset-specific collator; "
                f"получены формы {sorted(shapes)}"
            )
        return torch.stack(images)

    @staticmethod
    def _collate_context(
        samples: list[dict[str, Any]],
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        masks: list[Tensor] = [sample["context_valid_mask"] for sample in samples]
        offsets: list[Tensor] = [sample["context_time_offsets"] for sample in samples]
        max_slots: int = max(mask.numel() for mask in masks)
        batch_size: int = len(samples)
        device: torch.device = samples[0]["image"].device
        valid_mask: Tensor = torch.zeros(
            batch_size,
            max_slots,
            dtype=torch.bool,
            device=device,
        )
        time_offsets: Tensor = torch.zeros(
            batch_size,
            max_slots,
            dtype=torch.float32,
            device=device,
        )
        for index, (mask, offset) in enumerate(zip(masks, offsets, strict=True)):
            if mask.ndim != 1 or offset.shape != mask.shape:
                raise ValueError(
                    "context_valid_mask и context_time_offsets должны иметь "
                    "одинаковую одномерную форму"
                )
            slots: int = mask.numel()
            valid_mask[index, :slots] = mask.to(device=device, dtype=torch.bool)
            time_offsets[index, :slots] = offset.to(
                device=device,
                dtype=torch.float32,
            )

        context_values: list[Tensor | None] = [
            sample.get("context_images") for sample in samples
        ]
        present: list[Tensor] = [value for value in context_values if value is not None]
        if not present:
            return valid_mask, time_offsets, None
        frame_shapes: set[tuple[int, ...]] = {
            tuple(value.shape[1:]) for value in present
        }
        if len(frame_shapes) != 1:
            raise ValueError(
                "контекстные изображения разных размеров требуют "
                f"dataset-specific collator; получены формы {sorted(frame_shapes)}"
            )
        frame_shape: tuple[int, ...] = next(iter(frame_shapes))
        context_images: Tensor = torch.zeros(
            (batch_size, max_slots, *frame_shape),
            dtype=present[0].dtype,
            device=device,
        )
        for index, value in enumerate(context_values):
            if value is None:
                continue
            if value.shape[0] != masks[index].numel():
                raise ValueError(
                    f"sample[{index}]: число context_images не совпадает с маской"
                )
            context_images[index, : value.shape[0]] = value.to(device)
        return valid_mask, time_offsets, context_images


_COLLATOR: DetectionCollator = DetectionCollator()


def collate_fn(samples: list[dict[str, Any]]) -> tuple[DetectionBatch, ContextBatch]:
    """Совместимая функциональная точка входа для существующих импортов."""
    return _COLLATOR(samples)
