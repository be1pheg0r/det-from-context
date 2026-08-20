import torch
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


def create_dataloader(config):
    """
    Создаёт DataLoader на основе config.yaml.
    """

    from dataset import BDD100KDataset

    dataset = BDD100KDataset(config)

    dataloader_config = config["dataloader"]

    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=dataloader_config["shuffle"],
        num_workers=dataloader_config["num_workers"],
        collate_fn=detection_collate_fn,
        pin_memory=True,
    )

    return dataloader
