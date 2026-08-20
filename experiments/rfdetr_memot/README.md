# RF-DETR + external MeMOT

Эксперимент запускает официальный Lightning lifecycle RF-DETR, но обрабатывает
фиксированные видео-клипы. RF-DETR независимо формирует гипотезы текущего кадра;
Memory Encoder/Decoder MeMOT подключаются только после его forward.

## Граница совместимости с оригинальным MeMOT

Здесь реализована **paper-aligned внешняя реконструкция**, а не побитовая копия
официального MeMOT. RF-DETR заменяет исходный генератор гипотез и не получает
temporal queries внутрь backbone/encoder/decoder. Memory Encoder хранит
short/long-term представления треков, а Memory Decoder после RF-DETR уточняет
детекции и предсказывает их ассоциацию с памятью. Жёсткий runtime lifecycle
(создание, обновление и удаление track ID) использует эти logits вместе с
motion/IoU/cosine matching. Поэтому модуль корректно называть внешним MeMOT
adapter, но не полной репликой исходной end-to-end архитектуры.

Перед DataSphere-запуском проверьте корни и connector ID в `paths.yaml`, затем:

```bash
python experiments/rfdetr_memot/submit_datasphere.py
```

Локальный запуск использует те же компоненты:

```bash
VIDEO_DATASET_VIDEOS_DIR=/data/videos \
VIDEO_DATASET_ANNOTATIONS_DIR=/data/labels \
RFDETR_MEMOT_OUTPUT_ROOT=./runs/rfdetr-memot \
python experiments/rfdetr_memot/run.py
```

В `annotation_mode: auto` tracking-разметка включает association/uniqueness loss
и HOTA/AssA/IDF1. Если JSON содержит только опорный кадр, предыдущие кадры лишь
прогревают память, а tracking-метрики сохраняются как `available: false` с
причиной. Эксперимент публикует checkpoints, split manifest, detection-графики,
ID overlays, association heatmap, memory lifecycle и `tracking-metrics.json`.
После каждой validation-эпохи GIF с предсказанными track ID сохраняется локально
и публикуется в ClearML как media в `MeMOT prediction animations`.
