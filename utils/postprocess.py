import yaml


def load_config(config_path):
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def postprocess_boxes(
    boxes,
    config_path,
    original_width,
    original_height,
):
    config = load_config(config_path)

    target_width = config["dataset"]["image_size"]["width"]
    target_height = config["dataset"]["image_size"]["height"]

    scale_x = target_width / original_width
    scale_y = target_height / original_height

    processed_boxes = []

    for cx, cy, w, h in boxes:
        # Если bbox нормализован относительно изображения,
        # поданного в модель
        if config["dataset"]["normalize_boxes"]:
            cx *= target_width
            cy *= target_height
            w *= target_width
            h *= target_height

        # cx, cy, w, h -> x1, y1, x2, y2
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Возвращаем координаты исходного изображения
        x1 /= scale_x
        x2 /= scale_x
        y1 /= scale_y
        y2 /= scale_y

        processed_boxes.append([x1, y1, x2, y2])

    return processed_boxes
