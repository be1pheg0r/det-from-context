# RF-DETR component

RF-DETR оформлен как стандартный model component проекта:

```text
models/rfdetr/
├── provider.py
├── config.yaml
├── README.md
└── artifacts/
```

`provider.py` экспортирует `PROTOCOL` с двумя endpoints:

- `build(config) -> nn.Module` возвращает `ContextDetector` для обычного
  experiment pipeline;
- `build_detector(config) -> DetectorAdapter` оставляет detector-only API для
  тестов и специализированных запусков.

## Upstream dependency

Проект фиксирует `rfdetr==1.9.0`. Адаптер не содержит собственной реализации
backbone, projector, deformable decoder, matcher или loss-функций. Он создаёт
публичный upstream-класс `RFDETRNano`, `RFDETRSmall`, `RFDETRMedium` либо
`RFDETRLarge`, а затем переводит результат его `nn.Module` в `DetectorOutput`.
Точная фиксация версии нужна потому, что adapter использует внутренние точки
подключения backbone и decoder.

Поддерживаемые значения `variant` и официальный model config задаются явно:

```yaml
name: rfdetr
variant: small  # nano | small | medium | large
model:
  encoder: dinov2_windowed_small
  out_feature_indexes: [3, 6, 9, 12]
  projector_scale: [P4]
  hidden_dim: 256
  dec_layers: 3
  sa_nheads: 8
  ca_nheads: 16
  dec_n_points: 2
  num_queries: 300
  num_classes: 31
  resolution: 512
  pretrain_weights: rf-detr-small.pth
```

`model` не является сокращённой локальной схемой: весь mapping передаётся в
конструктор выбранного upstream-класса. Поэтому доступны все поля официального
`ModelConfig`, включая `two_stage`, `group_detr`, `bbox_reparam`,
`lite_refpoint_refine`, `gradient_checkpointing`, `backbone_lora`, decoder
registers, segmentation/keypoint и dual-projector switches. Полный
воспроизводимый набор для `small` находится в `models/rfdetr/config.yaml`.
Опечатки и несовместимые сочетания проверяет upstream Pydantic-модель.
System-managed поля `device`, `license` и checkpoint metadata `model_name`
намеренно не фиксируются в component YAML и остаются upstream defaults/runtime
metadata; остальные рабочие поля `RFDETRSmallConfig` перечислены явно.

Устаревший upstream-вариант `base` намеренно не поддерживается. При
bare filename в `model.pretrain_weights` использует официальный checkpoint
варианта, а `null` явно отключает pretrained checkpoint. Пользовательский
checkpoint размещается в `models/rfdetr/artifacts/`, а в конфиге задаётся
относительно директории компонента:

```yaml
model:
  pretrain_weights: artifacts/my-checkpoint.pth
```

## Experiment config

Стандартные конфиги уже наследуют подключение компонента из `_base_.yaml`:

```yaml
detector:
  name: rfdetr
  component_path: models/rfdetr
  config_path: models/rfdetr/config.yaml
  dim: 256
  num_queries: 300
  freeze_backbone: true
  freeze_decoder: false
```

`detector.dim` обязан совпадать с шириной decoder выбранного варианта: это
проверяется до начала обучения. Параметры freeze остаются настройками
эксперимента, а вариант и источник весов принадлежат model component.

## Вход и выход

`DetectionBatch.images` должен уже содержать float-тензоры, подготовленные так
же, как вход upstream RF-DETR: нужный размер варианта и нормализация изображения.
Для `small` базовый конфиг использует `512 × 512`. Provider не дублирует
upstream preprocessing внутри модели, чтобы training transforms оставались
едиными для target и context frames.

Без контекста adapter вызывает исходный forward без изменения queries, поэтому
logits и boxes совпадают с upstream-моделью. Для context-веток временный PyTorch
hook подменяет только вход decoder на batch-specific `query_init`; backbone,
projector, decoder и heads продолжают исполняться кодом RF-DETR. Hooks снимаются
после каждого forward, в том числе при исключении.

`DetectorOutput` содержит:

- финальные `pred_logits` и `pred_boxes`;
- query states и auxiliary predictions по слоям decoder;
- multi-scale backbone features;
- остальные upstream outputs в `aux`.

Лёгкие contract-тесты используют структурный stand-in upstream-модуля и не
скачивают веса. Они проверяют baseline-эквивалентность, подмену queries,
кодирование context frames, freeze и загрузку directory provider.
