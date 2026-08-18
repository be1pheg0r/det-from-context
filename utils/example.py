from postprocess import postprocess_boxes

boxes = [
    [0.5, 0.5, 0.2, 0.3],
]

result = postprocess_boxes(
    boxes=boxes,
    config_path="./datasets/image_dataloader/config.yaml",
    original_width=1920,
    original_height=1080,
)

print(result)
