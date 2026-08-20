# Нативное дообучение RF-DETR

Эксперимент дообучает component-модель на `image_dataloader` через официальный
PyTorch Lightning stack RF-DETR 1.9.0. Локальный код не дублирует matcher, criterion,
optimizer, scheduler, AMP, EMA, COCO-метрики или checkpoint callbacks. Адаптер меняет
только источник батчей: стандартные project contracts преобразуются в upstream
`NestedTensor + targets` в памяти, без временного COCO-датасета.

Запуск:

```powershell
.venv\Scripts\python.exe experiments/rfdetr_image/run.py
```

Режим split определяется `datasets/image_dataloader/config.yaml`. При
`mode: predefined` автоматически строятся train/validation/test; при `generated`
test добавляется, только если `test_fraction > 0`. Validation и test используют
детерминированные transforms без train-аугментаций.

Для generated-режима включена multilabel-стратификация. Train использует
class-aware `inverse_sqrt` sampling с ограничением веса 5.0; validation/test не
ресэмплируются. Это компенсирует редкие классы, сохраняя официальный RF-DETR
IoU-aware BCE criterion без локального fork loss-функции.

Каждый запуск сохраняет:

- полный console/file log и JSONL скаляров;
- loss, learning rate, mAP/F1 и per-class upstream-метрики;
- batch time, throughput, gradient norm и CUDA memory;
- примеры ground truth/predictions/errors, распределения датасета,
  confidence/IoU, confusion matrix, PR/F1 threshold sweep и историю метрик;
- точный split manifest с SHA-256;
- best/latest и ограниченное `output.keep_last_checkpoints` число периодических
  checkpoint’ов;
- resolved config и полный снимок кода model/dataset/worker компонентов.

Скаляры одновременно идут в консоль, `metrics.jsonl` и ClearML. Визуализации имеют
ограничения `max_visual_images`/`max_diagnostic_images`, поэтому объём памяти не
растёт вместе с датасетом.

## Datasphere

Пути Object Storage и ID коннектора задаются только в `paths.yaml`; ClearML и
Datasphere credentials читаются из `.env` и не попадают в архив VM:

```powershell
.venv\Scripts\python.exe experiments/rfdetr_image/submit_datasphere.py
```

Job использует V100, устанавливает CUDA-сборку PyTorch, ставит проект с
`rfdetr[train]==1.9.0`, запускает тот же `run.py`, пишет результаты на S3 mount и
закрывает ClearML Task только после публикации summary, checkpoints и log-файла.
