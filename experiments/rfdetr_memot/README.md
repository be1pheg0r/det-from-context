# RF-DETR + external MeMOT

Эксперимент запускает официальный Lightning lifecycle RF-DETR, но обрабатывает
фиксированные видео-клипы. RF-DETR независимо формирует гипотезы текущего кадра;
Memory Encoder/Decoder MeMOT подключаются только после его forward.

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
