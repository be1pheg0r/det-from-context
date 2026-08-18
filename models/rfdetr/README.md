# RF-DETR model component

Компонент подключает официальный пакет `rfdetr==1.9.0` к модельному протоколу
проекта. Архитектура RF-DETR не скопирована: `provider.py` строит локальный
`RFDetrAdapter`, который использует upstream backbone, projector, decoder,
prediction heads и веса.

- `config.yaml` явно задаёт официальный `ModelConfig`: backbone/projector,
  decoder, queries/classes, resolution, refinement, precision, checkpointing,
  segmentation/keypoint switches и веса.
- Блок `model` передаётся публичному upstream-классу без локальной копии схемы;
  неизвестные и несовместимые поля отклоняет Pydantic-валидация RF-DETR.
- `model.pretrain_weights` с bare filename использует официальный cache и при
  необходимости скачивает checkpoint; `null` означает обучение с нуля.
- Относительный путь с директорией разрешается от этой component directory;
  пользовательские checkpoints следует хранить в `artifacts/`.
- `artifacts/` предназначен для checkpoints, созданных экспериментами, и не
  коммитит бинарные веса.

Эксперимент подключает компонент через `detector.component_path: models/rfdetr`
и `detector.config_path: models/rfdetr/config.yaml`. Полная схема и требования к
входным тензорам описаны в `docs/rfdetr.md`.
