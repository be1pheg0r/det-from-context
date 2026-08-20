import os


def count_matching_files(videos_dir, annotations_dir):
    """
    Подсчитывает количество файлов с одинаковыми именами (до точки)
    в двух директориях.
    """
    # Проверяем существование директорий
    if not os.path.exists(videos_dir):
        print(f"❌ Ошибка: Директория не найдена: {videos_dir}")
        return
    if not os.path.exists(annotations_dir):
        print(f"❌ Ошибка: Директория не найдена: {annotations_dir}")
        return

    # Получаем имена файлов без расширений
    video_names = set()
    annotation_names = set()

    # Собираем имена видеофайлов
    for file in os.listdir(videos_dir):
        if os.path.isfile(os.path.join(videos_dir, file)):
            name_without_ext = os.path.splitext(file)[0]
            video_names.add(name_without_ext)

    # Собираем имена файлов аннотаций
    for file in os.listdir(annotations_dir):
        if os.path.isfile(os.path.join(annotations_dir, file)):
            name_without_ext = os.path.splitext(file)[0]
            annotation_names.add(name_without_ext)

    # Находим общие имена
    common_names = video_names & annotation_names

    # Выводим результаты
    print("=" * 60)
    print("📊 ПОДСЧЕТ ФАЙЛОВ С ОБЩИМИ ИМЕНАМИ")
    print("=" * 60)
    print("\n📁 Директория с видео:")
    print(f"   {videos_dir}")
    print(f"   Всего видеофайлов: {len(video_names)}")

    print("\n📁 Директория с аннотациями:")
    print(f"   {annotations_dir}")
    print(f"   Всего файлов аннотаций: {len(annotation_names)}")

    print(f"\n✅ Найдено файлов с общими именами: {len(common_names)}")

    if len(common_names) > 0 and len(common_names) <= 20:
        print("\n📋 Примеры общих имен:")
        for i, name in enumerate(sorted(common_names)[:20], 1):
            print(f"   {i}. {name}")
    elif len(common_names) > 20:
        print(f"\n📋 Первые 20 из {len(common_names)} общих имен:")
        for i, name in enumerate(sorted(common_names)[:20], 1):
            print(f"   {i}. {name}")

    print("\n" + "=" * 60)

    return {
        "video_count": len(video_names),
        "annotation_count": len(annotation_names),
        "common_count": len(common_names),
        "common_names": common_names,
    }


if __name__ == "__main__":
    # Пути к директориям
    videos_dir = (
        "/job/s3/bt17f0k57rfieh6jugr8/bdd100k_downloads/VIDEOS_TRAIN/bd"
        "d100k/videos/100k/train"
    )
    annotations_dir = "/job/s3/bt17f0k57rfieh6jugr8/bdd100k_downloads/100k/train"

    # Запускаем подсчет
    result = count_matching_files(videos_dir, annotations_dir)

    # Если нужно сохранить результат для использования в коде
    if result:
        print("\n📊 Итог:")
        print(f"   - Видео: {result['video_count']}")
        print(f"   - Аннотации: {result['annotation_count']}")
        print(f"   - Совпадений: {result['common_count']}")
