"""
Скрипт для выборочного скачивания архивов BDD100K.

Поддерживает два режима на каждую ссылку:

1) mode="stream"  — качаем архив потоком и обрываем
   по достижении лимита в ГБ. при обрыве архив невалиден
   (не откроется unzip'ом), т.к. обрезан не по границе файлов.

2) mode="partial" — через remotezip: смотрим оглавление архива,
   выбираем файлы по расширению/маске и скачиваем
   только их, целиком и корректно

Кеширование:
    Оглавление удалённого zip-архива (central directory) кешируется
    локально в JSON (./bdd100k_downloads/.cache/<hash>.json), чтобы
    повторные запуски не тратили сетевой запрос только на то, чтобы
    узнать список файлов внутри архива. Удалите файл кеша (или папку
    .cache целиком), если содержимое архива на сервере изменилось.

Установка зависимостей:
    pip install requests remotezip tqdm
"""

import fnmatch
import hashlib
import json
import os
import time

import requests
from remotezip import RemoteZip
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

GB = 1024**3

# Сколько раз пытаться перекачать один файл внутри архива, если
# соединение обрывается (IncompleteRead / ConnectionError / таймаут)
FILE_RETRY_ATTEMPTS = 5
FILE_RETRY_BACKOFF_SEC = 2  # 2s, 4s, 6s, 8s...


def make_retry_session() -> requests.Session:
    """Сессия с автоматическими повторами на уровне TCP/HTTP-соединения.

    Передаётся в RemoteZip(url, session=...), чтобы обрывы соединения
    при чтении Range-запросов (частая проблема с этим сервером)
    переподключались сами, без падения на уровне питоновского кода.
    """
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# ---------------------------------------------------------------------------
# 1. Список всех доступных ссылок
# ---------------------------------------------------------------------------
URLS = {
    "VIDEOS_TRAIN": "http://128.32.162.150/bdd100k/bdd100k_videos.zip",
    "VIDEOS_TEST": "http://128.32.162.150/bdd100k/bdd100k_videos.zip",
    "VIDEOS_VAL": "http://128.32.162.150/bdd100k/bdd100k_videos.zip",
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

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ---------------------------------------------------------------------------
# 2. НАСТРОЙКИ
# ---------------------------------------------------------------------------

DEST_DIR = "./bdd100k_downloads"
CHUNK_SIZE = 4 * 1024 * 1024  # для режима "stream" (обычный HTTP-докачка)

# Для чтения файлов ВНУТРИ zip через remotezip чанк должен быть заметно
# меньше: каждый read() — это отдельный Range-запрос к серверу, и чем
# он крупнее, тем выше риск, что нестабильный сервер оборвёт соединение
# посреди него (см. IncompleteRead). 8 КБ — дефолт самого zipfile
# (то, что использует zf.extract()), берём чуть крупнее ради скорости,
# но всё ещё маленький.
ZIP_READ_CHUNK_SIZE = 256 * 1024  # 256 КБ

# Папка для кеша оглавлений удалённых архивов
CACHE_DIR = os.path.join(DEST_DIR, ".cache")
USE_INDEX_CACHE = True  # поставьте False, чтобы всегда читать оглавление заново

DOWNLOAD_PLAN = {
    "VIDEOS_TRAIN": {
        "mode": "partial",
        "extensions": None,
        "name_glob": "*train/*",
        "max_files": None,
        "max_total_gb": 60,
    },
    "VIDEOS_TEST": {
        "mode": "partial",
        "extensions": None,
        "name_glob": "*test/*",
        "max_files": None,
        "max_total_gb": 20,
    },
    "VIDEOS_VAL": {
        "mode": "partial",
        "extensions": None,
        "name_glob": "*val/*",
        "max_files": None,
        "max_total_gb": 10,
    },
    # "LABELS": {
    #     "mode": "stream",
    #     "limit_gb": 100,
    # },
}

# ---------------------------------------------------------------------------
# 3. Кеш оглавления удалённого zip-архива
# ---------------------------------------------------------------------------


class CachedZipInfo:
    """Лёгкая замена zipfile.ZipInfo для данных, восстановленных из кеша.

    Хранит только то, что реально используется дальше по коду:
    имя файла внутри архива и его несжатый размер. Этого достаточно
    и для фильтрации (select_files), и для скачивания
    (download_partial_selected обращается к архиву по info.filename).
    """

    __slots__ = ("filename", "file_size", "_is_dir")

    def __init__(self, filename: str, file_size: int, is_dir: bool):
        self.filename = filename
        self.file_size = file_size
        self._is_dir = is_dir

    def is_dir(self) -> bool:
        return self._is_dir


def _cache_path_for_url(url: str) -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{h}.json")


def load_cached_index(url: str):
    """Возвращает список CachedZipInfo из кеша или None, если кеша нет."""
    path = _cache_path_for_url(url)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [
            CachedZipInfo(item["filename"], item["file_size"], item["is_dir"])
            for item in data.get("files", [])
        ]
    except (json.JSONDecodeError, OSError, KeyError):
        return None


def save_cached_index(url: str, infolist) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path_for_url(url)
    payload = {
        "url": url,
        "files": [
            {
                "filename": info.filename,
                "file_size": info.file_size,
                "is_dir": info.is_dir(),
            }
            for info in infolist
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def get_remote_index(url: str, use_cache: bool = USE_INDEX_CACHE):
    """Отдаёт оглавление архива: из кеша, если он есть, иначе по сети
    (и сразу сохраняет в кеш)."""
    if use_cache:
        cached = load_cached_index(url)
        if cached is not None:
            return cached

    with RemoteZip(url, session=make_retry_session()) as zf:
        infolist = zf.infolist()

    if use_cache:
        save_cached_index(url, infolist)

    return infolist


# ---------------------------------------------------------------------------
# 4. Общий прогресс-трекер по всему плану
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
# 5. Режим "stream" — качаем весь файл потоком с обрывом по лимиту
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


def download_stream_with_limit(
    url: str,
    dest_path: str,
    limit_bytes: int,
    chunk_size: int = CHUNK_SIZE,
    overall: OverallTracker = None,
):
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
            server_total = (
                (already + int(total_size_header)) if total_size_header else None
            )
            bar_total = min(server_total, limit_bytes) if server_total else limit_bytes

            downloaded_this_run = 0
            with (
                open(dest_path, mode) as f,
                tqdm(
                    total=bar_total,
                    initial=already,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"  {os.path.basename(dest_path)}",
                    position=1,
                    leave=False,
                ) as pbar,
            ):
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
# 6. Режим "partial" — выборочная докачка файлов ВНУТРИ архива
# ---------------------------------------------------------------------------


def select_files(
    all_infos, extensions=None, name_glob=None, max_files=None, max_total_gb=None
):
    """Фильтрует записи архива и возвращает (selected_infos, total_bytes)."""
    candidates = []
    for info in all_infos:
        name = info.filename
        if name.endswith("/") or info.is_dir():
            continue
        if extensions and not name.lower().endswith(
            tuple(e.lower() for e in extensions)
        ):
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


def plan_partial(
    url: str,
    dest_dir: str,
    extensions=None,
    name_glob=None,
    max_files=None,
    max_total_gb=None,
):
    """
    Читает оглавление удалённого архива (из кеша, если он есть) и
    возвращает список файлов, которые реально нужно докачать (то есть
    ещё не лежат на диске в правильном размере), и сумму их байт.
    """
    try:
        all_infos = get_remote_index(url)
    except Exception as e:
        print(f"  [ошибка] Не удалось прочитать оглавление {url}: {e}")
        return [], 0

    selected, _ = select_files(
        all_infos, extensions, name_glob, max_files, max_total_gb
    )

    to_download, remaining = [], 0
    for info in selected:
        out_path = os.path.join(dest_dir, info.filename)
        if os.path.exists(out_path) and os.path.getsize(out_path) == info.file_size:
            continue
        to_download.append(info)
        remaining += info.file_size
    return to_download, remaining


def _download_one_file(zf, info, out_path: str, overall: OverallTracker = None) -> bool:
    """Скачивает один файл из уже открытого RemoteZip с повторами при
    обрыве соединения. При неудачной попытке частично записанный файл
    отбрасывается (сжатые файлы в zip нельзя докачать 'с середины')
    и запись начинается заново со следующей попытки.
    Возвращает True при успехе, False если все попытки исчерпаны.
    """
    for attempt in range(1, FILE_RETRY_ATTEMPTS + 1):
        bytes_written_this_attempt = 0
        try:
            with (
                zf.open(info.filename) as src,
                open(out_path, "wb") as dst,
                tqdm(
                    total=info.file_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"  {os.path.basename(info.filename)}",
                    position=1,
                    leave=False,
                ) as pbar,
            ):
                while True:
                    buf = src.read(ZIP_READ_CHUNK_SIZE)
                    if not buf:
                        break
                    dst.write(buf)
                    pbar.update(len(buf))
                    bytes_written_this_attempt += len(buf)
                    if overall:
                        overall.update(len(buf))

            if os.path.getsize(out_path) == info.file_size:
                return True

            tqdm.write(
                f"  [{info.filename}] размер не совпал после скачивания "
                f"({os.path.getsize(out_path)} != {info.file_size}), повтор..."
            )

        except Exception as e:
            # ловим в т.ч. IncompleteRead/ConnectionError/таймауты
            tqdm.write(
                f"  [{info.filename}] попытка {attempt}/{FILE_RETRY_ATTEMPTS} "
                f"прервана: {e}"
            )
            # откатываем уже засчитанный в общий прогресс кусок,
            # т.к. файл будет перекачан заново
            if overall and bytes_written_this_attempt:
                overall.update(-bytes_written_this_attempt)

        if os.path.exists(out_path):
            os.remove(out_path)

        if attempt < FILE_RETRY_ATTEMPTS:
            time.sleep(FILE_RETRY_BACKOFF_SEC * attempt)

    return False


def download_partial_selected(
    url: str, dest_dir: str, selected, overall: OverallTracker = None
):
    """Скачивает уже отобранный список файлов. Обрыв соединения на одном
    файле не прерывает скачивание остальных — файл перекачивается
    заново (до FILE_RETRY_ATTEMPTS раз), затем скрипт идёт дальше."""
    if not selected:
        print("Нет файлов для скачивания")
        return

    os.makedirs(dest_dir, exist_ok=True)
    failed = []

    try:
        with RemoteZip(url, session=make_retry_session()) as zf:
            for info in selected:
                out_path = os.path.join(dest_dir, info.filename)
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

                ok = _download_one_file(zf, info, out_path, overall=overall)
                if not ok:
                    failed.append(info.filename)
                    tqdm.write(
                        f"  [{info.filename}] не удалось скачать за "
                        f"{FILE_RETRY_ATTEMPTS} попыток, пропускаю"
                    )

    except Exception as e:
        print(f"  [ошибка] Не удалось обработать архив: {e}")
        return

    if failed:
        print(
            f"  Готово с ошибками. Не скачано файлов: {len(failed)} "
            f"из {len(selected)}. Запустите скрипт повторно — "
            f"недокачанные файлы попробуются снова."
        )
    else:
        print(f"  Готово. Файлы сохранены в: {os.path.abspath(dest_dir)}")


# ---------------------------------------------------------------------------
# 7. Основной цикл: планирование + скачивание с общим прогресс-баром
# ---------------------------------------------------------------------------


def main():
    os.makedirs(DEST_DIR, exist_ok=True)

    unknown = [k for k in DOWNLOAD_PLAN if k not in URLS]
    if unknown:
        raise ValueError(
            f"Неизвестные ключи в DOWNLOAD_PLAN: {unknown}. "
            f"Доступные ключи: {list(URLS.keys())}"
        )

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
            plan[key] = {
                "mode": "stream",
                "url": url,
                "dest_path": dest_path,
                "limit_bytes": limit_bytes,
            }
            print(f"  [{key}] к скачиванию: {remaining / GB:.2f} ГБ")

        elif mode == "partial":
            sub_dir = os.path.join(DEST_DIR, key)
            to_download, remaining = plan_partial(
                url,
                sub_dir,
                extensions=cfg.get("extensions"),
                name_glob=cfg.get("name_glob"),
                max_files=cfg.get("max_files"),
                max_total_gb=cfg.get("max_total_gb"),
            )
            plan[key] = {
                "mode": "partial",
                "url": url,
                "dest_dir": sub_dir,
                "selected": to_download,
            }
            print(
                f"  [{key}] файлов к скачиванию: {len(to_download)},"
                f" объём: {remaining / GB:.2f} ГБ"
            )

        else:
            print(f"  [{key}] [ошибка] неизвестный режим: {mode}")
            remaining = 0

        total_to_download += remaining

    print(f"\nИтого к скачиванию: {total_to_download / GB:.2f} ГБ\n")

    overall = OverallTracker(total_to_download) if total_to_download > 0 else None

    for key, info in plan.items():
        print(f"[{key}] {info['url']}  (режим: {info['mode']})")

        if info["mode"] == "stream":
            download_stream_with_limit(
                info["url"], info["dest_path"], info["limit_bytes"], overall=overall
            )
        elif info["mode"] == "partial":
            download_partial_selected(
                info["url"], info["dest_dir"], info["selected"], overall=overall
            )

        print()

    if overall:
        overall.close()
        print("Всё скачивание завершено.")
    else:
        print("Нет файлов для скачивания")


if __name__ == "__main__":
    main()
