import json
from pathlib import Path


# надо настроить пути. файл json должен лежать на 1 уровне с кадрами
def convert_to_coco(
    json_dir: str,
    images_dir: str,
    output_json: str,
    image_width: int,
    image_height: int,
):
    json_dir = Path(json_dir)
    images_dir = Path(images_dir)

    # ---------------------------------------------------------
    # 1. Собираем все категории
    # ---------------------------------------------------------

    categories_set = set()

    json_files = list(json_dir.glob("*.json"))

    for json_file in json_files:
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):
                category = obj.get("category")

                if category is not None:
                    categories_set.add(category)

    categories = sorted(categories_set)

    category_to_id = {category: idx + 1 for idx, category in enumerate(categories)}

    # ---------------------------------------------------------
    # 2. COCO структура
    # ---------------------------------------------------------

    coco = {"images": [], "annotations": [], "categories": []}

    for category, category_id in category_to_id.items():
        coco["categories"].append(
            {"id": category_id, "name": category, "supercategory": "object"}
        )

    # ---------------------------------------------------------
    # 3. Обрабатываем JSON
    # ---------------------------------------------------------

    image_id = 1
    annotation_id = 1

    for json_file in json_files:
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        video_name = data["name"]

        # В твоем случае размечен ровно один кадр
        # поэтому берем первый frame
        frames = data.get("frames", [])

        if not frames:
            continue

        frame = frames[0]

        # -----------------------------------------------------
        # Ищем соответствующее изображение
        # -----------------------------------------------------

        image_path = None

        for extension in [".jpg", ".jpeg", ".png"]:
            candidate = images_dir / f"{video_name}{extension}"

            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            print(f"[WARNING] Не найдено изображение для {video_name}")
            continue

        # -----------------------------------------------------
        # IMAGE
        # -----------------------------------------------------

        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": image_width,
                "height": image_height,
            }
        )

        # -----------------------------------------------------
        # ANNOTATIONS
        # -----------------------------------------------------

        for obj in frame.get("objects", []):
            category = obj.get("category")

            if category not in category_to_id:
                continue

            box = obj.get("box2d")

            if box is None:
                continue

            x1 = float(box["x1"])
            y1 = float(box["y1"])
            x2 = float(box["x2"])
            y2 = float(box["y2"])

            width = x2 - x1
            height = y2 - y1

            # Защита от некорректных bbox
            if width <= 0 or height <= 0:
                continue

            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_to_id[category],
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )

            annotation_id += 1

        image_id += 1

    # ---------------------------------------------------------
    # 4. Сохраняем
    # ---------------------------------------------------------

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    print("Готово!")
    print(f"Images:      {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")
    print(f"Categories:  {len(coco['categories'])}")
    print(f"Saved to:    {output_path}")


def main():
    convert_to_coco(
        json_dir="../../dataset/labels/train",
        images_dir="../../dataset/10k/train/images",
        output_json="/job/s3/bt17f0k57rfieh6jugr8/",
        image_width=1280,
        image_height=720,
    )


if __name__ == "__main__":
    main()
