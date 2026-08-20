# Video dataloader

Компонент сопоставляет видео и BDD100K-подобные JSON по имени файла и
возвращает `DetectionClipBatch`: временную последовательность обычных проектных
`DetectionBatch` и маску кадров с разметкой.

Поддерживаются два режима:

- `tracking`: в JSON есть несколько размеченных кадров и стабильные object ID;
  каждый кадр клипа участвует в detection и association loss;
- `reference_frame`: размечен только опорный кадр; предыдущие кадры извлекаются
  из видео с `target_fps`, прогревают память и не входят в supervised loss.

`annotation_mode: auto` выбирает режим для каждого видео. В tracking-режиме
нужны минимум два timestamp и ID у всех принятых объектов. Исходные ID
перенумеровываются локально для каждого видео с нуля.

Корни видео и JSON независимы. Каталог split может находиться на любой глубине:

```text
videos/site-a/train/camera/sequence.mov
labels/export/train/sequence.json
```

При `strict_pairs: true` пропущенная или лишняя пара останавливает запуск.
Аугментации RF-DETR применяются с одинаковыми случайными параметрами ко всем
кадрам одного клипа. Размер входа задаётся выбранным вариантом RF-DETR, а не
конфигом датасета.

Для DataSphere достаточно определить `VIDEO_DATASET_VIDEOS_DIR` и
`VIDEO_DATASET_ANNOTATIONS_DIR`. Длина клипа, batch size и число workers берутся
из общего experiment config (`data.clip_len`, `train`/`validation`).
