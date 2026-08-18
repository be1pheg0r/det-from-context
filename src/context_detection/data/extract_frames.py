"""
Скрипт проходит по папке с видео и для каждого видео вырезает один кадр
на таймстемпе TIMESTAMP_MS (в миллисекундах), сохраняя его как картинку.


Все пути и настройки задаются ниже в блоке "НАСТРОЙКИ" — просто впишите свои
и запустите: python extract_frames.py

Зависимости:
    pip install opencv-python
"""

import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    print("Нужен пакет opencv-python: pip install opencv-python", file=sys.stderr)
    sys.exit(1)


# ========================= НАСТРОЙКИ =========================
VIDEOS_DIR = Path(
    "/job/s3/bt17f0k57rfieh8/bdd100k_downloads/VIDEOS_TEST/bdd100k/videos/100k/test"
)  # папка с видео
OUT_IMAGES_DIR = Path("out/images_test_2")  # куда сохранять вырезанные кадры

TIMESTAMP_MS = 10000  # какой кадр брать (в миллисекундах от начала видео)
IMAGE_EXT = ".jpg"  # расширение картинок: ".jpg" или ".png"
JPEG_QUALITY = 95  # качество JPEG (1-100), актуально только для .jpg
# ===============================================================


VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]


def extract_frame_at_ms(cap: cv2.VideoCapture, timestamp_ms: float):
    """Достаём кадр по таймстемпу в миллисекундах."""
    cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def main():
    if not VIDEOS_DIR.exists():
        print(f"Папка с видео не найдена: {VIDEOS_DIR}")
        return

    OUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        p
        for p in VIDEOS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    if not video_files:
        print(f"В {VIDEOS_DIR} не найдено видеофайлов")
        return

    saved = 0
    failed = 0

    for video_path in video_files:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[!] Не удалось открыть видео {video_path.name}")
            failed += 1
            continue

        frame = extract_frame_at_ms(cap, TIMESTAMP_MS)
        cap.release()

        if frame is None:
            print(f"[!] Не удалось извлечь кадр {TIMESTAMP_MS} мс из {video_path.name}")
            failed += 1
            continue

        img_path = OUT_IMAGES_DIR / f"{video_path.stem}{IMAGE_EXT}"
        if IMAGE_EXT.lower() in (".jpg", ".jpeg"):
            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        else:
            cv2.imwrite(str(img_path), frame)

        print(f"[+] {video_path.name} -> {img_path.name}")
        saved += 1

    print("\nГотово.")
    print(f"Сохранено кадров: {saved}")
    print(f"Ошибок: {failed}")


if __name__ == "__main__":
    main()
