# Протоколы датасетов и моделей

Основная единица расширения — self-contained component directory. Датасет и
модель живут каждый в своей папке, как отдельный эксперимент: рядом находятся
код provider, его конфиг и принадлежащие компоненту артефакты.

```text
datasets/<dataset_name>/
├── provider.py
├── config.yaml
└── artifacts/

models/<model_name>/
├── provider.py
├── config.yaml
└── artifacts/
```

`ComponentDirectory` проверяет layout до запуска обучения. `config.yaml` обязан
содержать строковый `name`, совпадающий с именем в experiment config;
`provider.py` обязан экспортировать объект `PROTOCOL`. Отсутствующий файл,
папка artifacts, несовпадающее имя или несовместимый provider завершают запуск
сразу и с указанием конкретного component path.

## Experiment config

Directory-backed components подключаются декларативно:

```yaml
data:
  name: synthetic_regression
  component_path: datasets/synthetic_regression
  config_path: datasets/synthetic_regression/config.yaml

detector:
  name: linear_regression
  component_path: models/linear_regression
  config_path: models/linear_regression/config.yaml
```

Пути разрешаются относительно project root. Во время запуска experiment
protocol подставляет абсолютные проверенные пути в runtime-копию конфига,
динамически регистрирует оба provider и сохраняет их код/конфиги в source
snapshot. Исходный portable YAML при этом остаётся относительным.

## Dataset endpoint

Dataset provider структурно реализует:

```python
class MyDatasetProtocol:
    def build(self, config, split) -> DataLoader:
        ...


PROTOCOL = MyDatasetProtocol()
```

Реализация свободна выбирать `Dataset`, sampler и collator, но конечный объект
всегда проверяется как `torch.utils.data.DataLoader`. Для detection datasets
доступен `DetectionCollator`, создающий существующие `DetectionBatch` и
`ContextBatch` и дополняющий переменное число context slots.

## Model endpoint

Model provider структурно реализует:

```python
class MyModelProtocol:
    def build(self, config) -> nn.Module:
        ...


PROTOCOL = MyModelProtocol()
```

Конечный объект всегда проверяется как `torch.nn.Module`. Detection provider
может дополнительно реализовать `build_detector(config) -> DetectorAdapter`,
что сохраняет совместимость старого `build_detector`. Встроенные `dummy` и
совместимый RF-DETR fallback продолжают работать через тот же registry, однако
стандартные experiment configs подключают RF-DETR как directory component из
`models/rfdetr`. Детали адаптера, весов и входного контракта описаны в
[`docs/rfdetr.md`](rfdetr.md).

## Experiment endpoint и ClearML

Для directory components используется `execute_components`:

```python
def train(run, config, components):
    train_loader = components.loader("train")
    model = components.model
    model_artifacts = components.artifacts("model")
    # train loop; checkpoint сохраняется в model_artifacts
    run.log_metrics({"loss": 0.1}, step=1, split="train")
    return {"status": "ok"}


ExperimentProtocol().execute_components("experiment.yaml", train)
```

Порядок гарантирован: component layout и provider проверяются до запуска,
затем создаётся tracking backend/ClearML Task, а `DataLoader` и `nn.Module`
строятся уже после него. По завершении protocol копирует runtime-файлы из
component `artifacts/` в изолированную run directory и загружает их в ClearML.
В `metadata.json` сохраняются component paths, endpoint types, artifact names,
ClearML task id и dashboard URL. Секреты из `.env` туда не попадают.

## Regression reference

Рабочий пример состоит из трёх независимых папок:

- `datasets/synthetic_regression` — генератор, dataset config и tensor artifacts;
- `models/linear_regression` — `nn.Linear`, model config и checkpoint artifacts;
- `experiments/regression_synthetic` — train loop и experiment config.

Dataset provider создаёт `train.pt` и `validation.pt`; worker сохраняет
`model.pt` в папку модели. Experiment protocol публикует все три файла локально
и в ClearML. Это сквозная проверка совместимости всех трёх протоколов.

## Совместимость и прямой registry API

Существующие `build_dataset`, `build_model`, `build_detector` и старый
двухаргументный `ExperimentProtocol.execute` не изменили контракт. Для
встроенных или создаваемых в Python компонентов остаются доступны
`register_dataset_protocol` и `register_model_protocol`; folder layout является
стандартным способом оформить самостоятельный воспроизводимый компонент.
