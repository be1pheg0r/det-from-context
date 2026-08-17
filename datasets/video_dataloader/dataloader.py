# dataloader.py

import torch
from dataset import VideoDataset
from torch.utils.data import DataLoader


def video_collate_fn(batch):
    """
    Собирает batch для VideoDataset.

    Один элемент batch:
        frames:  [T, C, H, W]
        target:  dict

    После collate:

        frames:
            [B, T, C, H, W]

        targets:
            list[dict]
    """

    frames = []
    targets = []

    for sample_frames, sample_target in batch:
        frames.append(sample_frames)
        targets.append(sample_target)

    # [B, T, C, H, W]
    frames = torch.stack(frames, dim=0)

    return frames, targets


def create_video_dataloader(config):
    """
    Создаёт VideoDataset и DataLoader
    на основе config.yaml.
    """

    dataset = VideoDataset(config)

    dataloader_config = config["dataloader"]

    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=dataloader_config["shuffle"],
        num_workers=dataloader_config["num_workers"],
        pin_memory=dataloader_config.get("pin_memory", True),
        collate_fn=video_collate_fn,
    )

    return dataloader
