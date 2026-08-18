# Метрики для ablation-экспериментов

Модуль `context_detection.evaluation` следует протоколу экспериментов из
`develop`: он возвращает только конечные скаляры, которые worker может сразу
передать в `ExperimentRun.log_metrics`.

## Один validation-прогон

```python
from context_detection.evaluation import coco_ap

metrics = coco_ap(predictions, targets)
run.log_metrics(metrics, step=epoch, split="validation")
```

`predictions` — список словарей:

```text
image_id  scalar, опционально
boxes     [N,4] normalized cxcywh
scores    [N] в [0,1]
labels    [N] int64
```

`targets`:

```text
image_id    scalar, опционально
image_size  [H,W], рекомендуется
boxes       [M,4] normalized cxcywh
labels      [M] int64
iscrowd     [M], опционально
```

Если `image_size` отсутствует, общие mAP/AP50/AP75 остаются корректными, но
area-метрики small/medium/large не возвращаются.

Как и в `pycocotools`, area-метрика может быть `-1`, если в наборе нет ни
одного объекта соответствующего размера. Это означает «не определена», а не
нулевое качество.

## Окклюзия и первое появление

Для AP по объектам под окклюзией target дополнительно передаёт:

```text
occluded  [M] bool
```

```python
from context_detection.evaluation import occluded_coco_ap

occluded_metrics = occluded_coco_ap(predictions, targets)
```

Для AP на впервые появившихся объектах нужны tracking-аннотации:

```text
sequence_id   scalar str/int
frame_id      scalar int
instance_ids  [M] str/int, постоянные между кадрами одной сцены
```

```python
from context_detection.evaluation import newly_appeared_coco_ap

new_object_metrics = newly_appeared_coco_ap(predictions, targets)
```

Записи можно передать не по порядку: функция сортирует кадры внутри сцены.
Однако набор должен начинаться с начала validation-сцены, иначе уже видимый до
начала обрезанного клипа объект будет ошибочно считаться новым.

Для BDD100K data-компонент должен переносить `label.id` в `instance_ids`, а
`label.attributes.occluded` — в `occluded`. Нужны box-tracking labels: ID из
обычной покадровой detection-разметки нельзя считать постоянным треком.

Объекты вне выбранного subset помечаются как ignore. Поэтому корректные
детекции обычных видимых или уже встречавшихся объектов не становятся false
positive в специализированной метрике.

## Из `DetectorOutput`

```python
from context_detection.evaluation import detector_output_to_predictions

batch_predictions = detector_output_to_predictions(
    output,
    image_ids=batch.frame_id,
    score_threshold=0.05,
    max_detections=100,
)
```

Этот helper предполагает sigmoid foreground logits. Детектор с отдельным
softmax no-object классом должен подготовить `scores` и `labels` в адаптере.

## Сравнение с baseline

Каждый вариант запускается experiment worker-ом отдельно. После этого готовые
метрики упаковываются в один плоский словарь:

```python
from context_detection.evaluation import failure_mode_grid

grid = failure_mode_grid(
    {
        "baseline": baseline_metrics,
        "memot": memot_metrics,
        "empty_context": empty_context_metrics,
        "shuffled_context": shuffled_context_metrics,
    }
)
run.log_metrics(grid, step=0, split="ablation")
```

Пример результата:

```text
baseline.map=0.401
memot.map=0.421
memot.delta_map=0.020
empty_context.map=0.397
empty_context.delta_map=-0.004
```

Hydra overrides, инференс, длина истории, empty/shuffled context, latency и
peak VRAM относятся к worker-у: модуль метрик не запускает модель повторно.

## Диагностика памяти

```python
from context_detection.evaluation import memory_diagnostics_summary

diagnostics = memory_diagnostics_summary(diagnostics_log)
run.log_metrics(diagnostics, step=epoch, split="memory")
```

Attention tensors пропускаются. `evicted` суммируется, остальные scalar
diagnostics усредняются.
