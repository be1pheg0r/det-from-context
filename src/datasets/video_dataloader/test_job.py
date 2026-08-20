import yaml

from .dataloader import create_video_dataloader


def main():
    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    dataloader = create_video_dataloader(config)
    print("Количество samples:", len(dataloader.dataset))

    for frames, targets in dataloader:
        print("Frames shape:", frames.shape)
        print("Number of targets:", len(targets))
        for target in targets:
            print("boxes:", target["boxes"].shape, "labels:", target["labels"].shape)
        print(target)
        break

if __name__ == "__main__":
    main()
