"""
Проходит по папке с видео, для каждого видео ищет одноимённый json
в другой папке и перемещает найденные json в указанную папку.

Пример: video_dir/clip1.mp4 -> ищем json_dir/clip1.json -> перемещаем
в moved_json_dir/clip1.json

Все пути задаются ниже в блоке "НАСТРОЙКИ".
"""

import shutil
from pathlib import Path

# ========================= НАСТРОЙКИ =========================

VIDEOS_DIR = Path(
    r"bdd100k_downloads/VIDEOS_TEST/bdd100k/videos/100k/test"
)  # папка с видео
JSON_DIR = Path(r"100k/test")  # папка с json-разметкой
OUT_JSON_DIR = Path(r"out/labels3")  # куда сохранять json по каждому кадру

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
# ===============================================================


def main():
    if not VIDEOS_DIR.exists():
        print(f"Папка с видео не найдена: {VIDEOS_DIR}")
        return
    if not JSON_DIR.exists():
        print(f"Папка с json не найдена: {JSON_DIR}")
        return

    OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        p
        for p in VIDEOS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    if not video_files:
        print(f"В {VIDEOS_DIR} не найдено видеофайлов")
        return

    moved = 0
    not_found = 0

    for video_path in video_files:
        json_path = JSON_DIR / f"{video_path.stem}.json"

        if not json_path.exists():
            print(f"[-] Json для {video_path.name} не найден ({json_path.name})")
            not_found += 1
            continue

        dest_path = OUT_JSON_DIR / json_path.name

        if dest_path.exists():
            print(f"[!] {dest_path.name} уже есть в {OUT_JSON_DIR}, пропускаю")
            continue

        shutil.move(str(json_path), str(dest_path))
        print(f"[+] {json_path.name} -> {OUT_JSON_DIR}")
        moved += 1

    print("\nГотово.")
    print(f"Перемещено json: {moved}")
    print(f"Видео без json: {not_found}")


if __name__ == "__main__":
    main()
