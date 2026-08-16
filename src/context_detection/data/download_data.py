
"""
Скрипт для выборочного скачивания архивов BDD100K.

Поддерживает два режима на каждую ссылку:

1) mode="stream"  — качаем архив потоком и обрываем
   по достижении лимита в ГБ. при обрыве архив невалиден
   (не откроется unzip'ом), т.к. обрезан не по границе файлов.

2) mode="partial" — через remotezip: смотрим оглавление архива,
   выбираем файлы по расширению/маске и скачиваем
   только их, целиком и корректно 



Установка зависимостей:
    pip install requests remotezip tqdm
"""

import os
import fnmatch
import requests
from remotezip import RemoteZip
from tqdm import tqdm

GB = 1024 ** 3

# ---------------------------------------------------------------------------
# 1. Список всех доступных ссылок
# ---------------------------------------------------------------------------
URLS = {
    "VIDEOS": "http://128.32.162.150/bdd100k/bdd100k_videos.zip",
    "INFO": "http://128.32.162.150/bdd100k/bdd100k_info.zip",
    "IMAGES_100K": "http://128.32.162.150/bdd100k/bdd100k_images_100k.zip",
    "IMAGES_10K": "http://128.32.162.150/bdd100k/bdd100k_images_10k.zip",
    "LABELS": "http://128.32.162.150/bdd100k/bdd100k_labels.zip",
    "MOT_2020_IMAGES_TEST_1": "http://128.32.162.150/bdd100k/mot20/images20-track-test-1.zip",
    "MOT_2020_IMAGES_TEST_2": "http://128.32.162.150/bdd100k/mot20/images20-track-test-2.zip",
    "MOT_2020_IMAGES_TRAIN_1": "http://128.32.162.150/bdd100k/mot20/images20-track-train-1.zip",
    "MOT_2020_IMAGES_TRAIN_2": "http://128.32.162.150/bdd100k/mot20/images20-track-train-2.zip",
    "MOT_2020_IMAGES_TRAIN_3": "http://128.32.162.150/bdd100k/mot20/images20-track-train-3.zip",
    "MOT_2020_IMAGES_TRAIN_4": "http://128.32.162.150/bdd100k/mot20/images20-track-train-4.zip",
    "MOT_2020_IMAGES_TRAIN_5": "http://128.32.162.150/bdd100k/mot20/images20-track-train-5.zip",
    "MOT_2020_IMAGES_TRAIN_6": "http://128.32.162.150/bdd100k/mot20/images20-track-train-6.zip",
    "MOT_2020_IMAGES_TRAIN_7": "http://128.32.162.150/bdd100k/mot20/images20-track-train-7.zip",
    "MOT_2020_IMAGES_VAL_1": "http://128.32.162.150/bdd100k/mot20/images20-track-val-1.zip",
    "DETECTION_2020_LABELS": "http://128.32.162.150/bdd100k/bdd100k_det_20_labels.zip",
}

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# ---------------------------------------------------------------------------
# 2. НАСТРОЙКИ
# ---------------------------------------------------------------------------

DEST_DIR = "./bdd100k_downloads"
CHUNK_SIZE = 4 * 1024 * 1024  # для режима "stream" и построчного чтения из zip

DOWNLOAD_PLAN = {
    "MOT_2020_IMAGES_TEST_2": {
        "mode": "partial",
        "extensions": None,
        "name_glob": None,
        "max_files": 10,
        "max_total_gb": 0.01,
    },
    # "LABELS": {
    #     "mode": "stream",
    #     "limit_gb": 100,
    # },
}

# ---------------------------------------------------------------------------
# 3. Общий прогресс-трекер по всему плану
# ---------------------------------------------------------------------------

class OverallTracker:
    

    def __init__(self, total_bytes: int):
        self.total = total_bytes
        self.pbar = tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="итого по всем ссылкам",
            position=0,
            leave=True,
        )
        self._last_milestone_gb = 0

    def update(self, n_bytes: int):
        if n_bytes <= 0:
            return
        self.pbar.update(n_bytes)

        current_gb = self.pbar.n / GB
        while current_gb >= self._last_milestone_gb + 1:
            self._last_milestone_gb += 1
            remaining = max(self.total - self.pbar.n, 0)
            tqdm.write(
                f"  [прогресс] всего скачано: {self.pbar.n / GB:.2f} ГБ, "
                f"осталось примерно: {remaining / GB:.2f} ГБ"
            )

    def close(self):
        self.pbar.close()


# ---------------------------------------------------------------------------
# 4. Режим "stream" — качаем весь файл потоком с обрывом по лимиту
# ---------------------------------------------------------------------------

def plan_stream(url: str, dest_path: str, limit_bytes: int) -> int:
    """Возвращает, сколько байт предстоит докачать"""
    already = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
    if already >= limit_bytes:
        return 0

    server_size = None
    try:
        r = requests.head(url, allow_redirects=True, timeout=15)
        if "Content-Length" in r.headers:
            server_size = int(r.headers["Content-Length"])
    except requests.exceptions.RequestException:
        pass

    planned_total = min(server_size, limit_bytes) if server_size else limit_bytes
    return max(planned_total - already, 0)


def download_stream_with_limit(url: str, dest_path: str, limit_bytes: int,
                                chunk_size: int = CHUNK_SIZE, overall: OverallTracker = None):
    already = 0
    mode = "wb"
    headers = {}

    if os.path.exists(dest_path):
        already = os.path.getsize(dest_path)
        if already >= limit_bytes:
            print(f"  [skip] уже скачано {already / GB:.2f} ГБ >= лимита")
            return
        headers["Range"] = f"bytes={already}-"
        mode = "ab"

    remaining = limit_bytes - already

    try:
        with requests.get(url, headers=headers, stream=True, timeout=30) as r:
            if headers.get("Range") and r.status_code == 200:
                print("  Сервер не поддерживает докачку (Range), начинаю заново.")
                already = 0
                remaining = limit_bytes
                mode = "wb"

            r.raise_for_status()

            total_size_header = r.headers.get("Content-Length")
            server_total = (already + int(total_size_header)) if total_size_header else None
            bar_total = min(server_total, limit_bytes) if server_total else limit_bytes

            downloaded_this_run = 0
            with open(dest_path, mode) as f, tqdm(
                total=bar_total,
                initial=already,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"  {os.path.basename(dest_path)}",
                position=1,
                leave=False,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    if downloaded_this_run + len(chunk) > remaining:
                        chunk = chunk[: remaining - downloaded_this_run]
                        f.write(chunk)
                        pbar.update(len(chunk))
                        if overall:
                            overall.update(len(chunk))
                        downloaded_this_run += len(chunk)
                        pbar.set_postfix_str("лимит достигнут, стоп")
                        break

                    f.write(chunk)
                    pbar.update(len(chunk))
                    if overall:
                        overall.update(len(chunk))
                    downloaded_this_run += len(chunk)

    except requests.exceptions.RequestException as e:
        print(f"  [ошибка] Не удалось скачать {url}: {e}")
        return

    final_size = os.path.getsize(dest_path)
    print(f"  Готово: {dest_path} ({final_size / GB:.2f} ГБ)")


# ---------------------------------------------------------------------------
# 5. Режим "partial" — выборочная докачка файлов ВНУТРИ архива
# ---------------------------------------------------------------------------

def select_files(all_infos, extensions=None, name_glob=None, max_files=None, max_total_gb=None):
    """Фильтрует записи архива и возвращает (selected_infos, total_bytes)."""
    candidates = []
    for info in all_infos:
        name = info.filename
        if name.endswith("/") or info.is_dir():
            continue
        if extensions and not name.lower().endswith(tuple(e.lower() for e in extensions)):
            continue
        if name_glob and not fnmatch.fnmatch(name, name_glob):
            continue
        candidates.append(info)

    max_total_bytes = int(max_total_gb * GB) if max_total_gb else None
    selected, total = [], 0
    for info in candidates:
        if max_files is not None and len(selected) >= max_files:
            break
        if max_total_bytes is not None and total + info.file_size > max_total_bytes:
            continue
        selected.append(info)
        total += info.file_size
    return selected, total


def plan_partial(url: str, dest_dir: str, extensions=None, name_glob=None,
                  max_files=None, max_total_gb=None):
    """
    Читает оглавление удалённого архива и возвращает список файлов,
    которые реально нужно докачать (то есть ещё не лежат на диске
    в правильном размере), и сумму их байт.
    """
    try:
        with RemoteZip(url) as zf:
            selected, _ = select_files(zf.infolist(), extensions, name_glob, max_files, max_total_gb)
    except Exception as e:
        print(f"  [ошибка] Не удалось прочитать оглавление {url}: {e}")
        return [], 0

    to_download, remaining = [], 0
    for info in selected:
        out_path = os.path.join(dest_dir, info.filename)
        if os.path.exists(out_path) and os.path.getsize(out_path) == info.file_size:
            continue
        to_download.append(info)
        remaining += info.file_size
    return to_download, remaining


def download_partial_selected(url: str, dest_dir: str, selected, overall: OverallTracker = None):
    """Скачивает уже отобранный список файлов"""
    if not selected:
        print("Нет файлов для скачивания")
        return

    os.makedirs(dest_dir, exist_ok=True)
    try:
        with RemoteZip(url) as zf:
            for info in selected:
                out_path = os.path.join(dest_dir, info.filename)
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

                with zf.open(info.filename) as src, open(out_path, "wb") as dst, tqdm(
                    total=info.file_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"  {os.path.basename(info.filename)}",
                    position=1,
                    leave=False,
                ) as pbar:
                    while True:
                        buf = src.read(CHUNK_SIZE)
                        if not buf:
                            break
                        dst.write(buf)
                        pbar.update(len(buf))
                        if overall:
                            overall.update(len(buf))

        print(f"  Готово. Файлы сохранены в: {os.path.abspath(dest_dir)}")

    except Exception as e:
        print(f"  [ошибка] Не удалось обработать архив: {e}")


# ---------------------------------------------------------------------------
# 6. Основной цикл: планирование + скачивание с общим прогресс-баром
# ---------------------------------------------------------------------------

def main():
    os.makedirs(DEST_DIR, exist_ok=True)

    unknown = [k for k in DOWNLOAD_PLAN if k not in URLS]
    if unknown:
        raise ValueError(f"Неизвестные ключи в DOWNLOAD_PLAN: {unknown}. "
                          f"Доступные ключи: {list(URLS.keys())}")

    print(f"Папка назначения: {os.path.abspath(DEST_DIR)}")
    print(f"План скачивания: {list(DOWNLOAD_PLAN.keys())}\n")
    print("Планирование\n")

    plan = {}
    total_to_download = 0

    for key, cfg in DOWNLOAD_PLAN.items():
        url = URLS[key]
        mode = cfg.get("mode", "stream")

        if mode == "stream":
            limit_gb = cfg.get("limit_gb", 100)
            limit_bytes = int(limit_gb * GB)
            filename = os.path.basename(url)
            dest_path = os.path.join(DEST_DIR, filename)
            remaining = plan_stream(url, dest_path, limit_bytes)
            plan[key] = {"mode": "stream", "url": url, "dest_path": dest_path, "limit_bytes": limit_bytes}
            print(f"  [{key}] к скачиванию: {remaining / GB:.2f} ГБ")

        elif mode == "partial":
            sub_dir = os.path.join(DEST_DIR, key)
            to_download, remaining = plan_partial(
                url, sub_dir,
                extensions=cfg.get("extensions"),
                name_glob=cfg.get("name_glob"),
                max_files=cfg.get("max_files"),
                max_total_gb=cfg.get("max_total_gb"),
            )
            plan[key] = {"mode": "partial", "url": url, "dest_dir": sub_dir, "selected": to_download}
            print(f"  [{key}] файлов к скачиванию: {len(to_download)}, объём: {remaining / GB:.2f} ГБ")

        else:
            print(f"  [{key}] [ошибка] неизвестный режим: {mode}")
            remaining = 0

        total_to_download += remaining

    print(f"\nИтого к скачиванию: {total_to_download / GB:.2f} ГБ\n")

    overall = OverallTracker(total_to_download) if total_to_download > 0 else None

    for key, info in plan.items():
        print(f"[{key}] {info['url']}  (режим: {info['mode']})")

        if info["mode"] == "stream":
            download_stream_with_limit(info["url"], info["dest_path"], info["limit_bytes"], overall=overall)
        elif info["mode"] == "partial":
            download_partial_selected(info["url"], info["dest_dir"], info["selected"], overall=overall)

        print()

    if overall:
        overall.close()
        print("Всё скачивание завершено.")
    else:
        print("Нет файлов для скачивания")


if __name__ == "__main__":
    main()