from typing import Any

import torch
from dataset import ImageDetectionDataset
from settings import ImageDataLoaderSettings
from torch.utils.data import DataLoader


def detection_collate_fn(batch):
    """
    Собирает batch для object detection.

    Каждый элемент batch:
        (image, target)

    Возвращает:
        images  - Tensor [B, C, H, W]
        targets - list[dict]
    """

    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    images = torch.stack(images, dim=0)

    return images, targets


def create_dataloader(config: dict[str, Any], split: str = "train") -> DataLoader[Any]:
    """Create the backwards-compatible standalone loader for a named split."""
    settings = ImageDataLoaderSettings.model_validate(config)
    split_name = "val" if split in {"validation", "valid"} else split
    dataset = ImageDetectionDataset(
        settings.dataset.images_dir,
        settings.dataset.annotations_dir,
        settings,
        image_set=split_name,
    )
    dataloader_config = settings.dataloader
    return DataLoader(
        dataset,
        batch_size=dataloader_config.batch_size,
        shuffle=dataloader_config.shuffle if split_name == "train" else False,
        num_workers=dataloader_config.num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=dataloader_config.num_workers > 0,
    )
