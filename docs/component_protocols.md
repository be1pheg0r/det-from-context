# Протоколы датасетов и моделей

Компоненты расширяются через структурные Python-протоколы и runtime-реестры.
Наследование от проектного базового класса не требуется. При этом внешние
границы фиксированы:

- dataset protocol всегда возвращает `torch.utils.data.DataLoader`;
- model protocol всегда возвращает `torch.nn.Module`;
- experiment protocol создаёт запуск до сборки компонентов и принимает worker,
  который использует эти стандартные PyTorch endpoints.

Существующие `build_dataset`, `build_model` и `build_detector` сохранены.
Встроенные имена `dummy`, `imagenet_vid`, `ovis` и `rfdetr` проходят через те же
реестры, поэтому отдельного legacy-пути нет.

## Dataset protocol

Минимальный provider реализует один метод:

```python
from torch.utils.data import DataLoader

from context_detection import DatasetSplit, register_dataset_protocol


class MyDataset:
    def build(self, config, split: DatasetSplit) -> DataLoader:
        dataset = create_dataset(config, split)
        return DataLoader(dataset, batch_size=config.train.batch_size)


register_dataset_protocol("my_dataset", MyDataset())
```

Регистрация должна произойти до загрузки Hydra/Pydantic-конфига. После неё
`data.name: my_dataset` считается валидным. Пользовательский provider может
работать без `data.root`; root обязателен только для встроенных
`imagenet_vid` и `ovis`.

`DatasetSplit` принимает `train`, `validation`, `test`; алиас `val`
нормализуется в `validation`. Реестр проверяет итоговый объект и сразу выдаёт
понятную ошибку, если provider вернул не `DataLoader`.

Для detection pipeline доступен `DetectionCollator`. Он собирает sample-словари
в существующие `DetectionBatch` и `ContextBatch`, дополняет разное число
контекстных слотов невалидными значениями и сохраняет extras. Изображения разных
размеров требуют пользовательского collator с resize/padding.

## Model protocol

Минимальный provider также структурный:

```python
from torch import nn

from context_detection import register_model_protocol


class MyModel:
    def build(self, config) -> nn.Module:
        return nn.Linear(config.detector.dim, config.detector.num_classes)


register_model_protocol("my_model", MyModel())
```

После регистрации имя разрешено в `detector.name`. Реестр гарантирует только
универсальную границу `nn.Module`; конкретная сигнатура `forward` определяется
pipeline. Для совместимости со старым `build_detector` provider дополнительно
реализует `build_detector(config) -> DetectorAdapter`. Встроенные `dummy` и
`rfdetr` реализуют оба endpoint и собирают прежний `ContextDetector` без
изменения его `forward(batch, context, state)`.

## Совместимость с experiment protocol и ClearML

Компоненты следует строить внутри worker, потому что `ExperimentProtocol`
сначала создаёт ClearML Task, подключает resolved config и только затем вызывает
worker:

```python
from context_detection.build import build_dataset, build_model
from context_detection.experiment import ExperimentProtocol


def train(run, config):
    loader = build_dataset(config, "train")
    model = build_model(config)
    # optimizer / train loop
    run.log_metrics({"loss": 0.1}, step=0, split="train")
    run.save_artifact("model", checkpoint_path)
    return {"status": "ok"}


ExperimentProtocol().execute("configs/experiment.yaml", train)
```

Так ClearML автоматически видит создаваемые PyTorch-модели, а метрики,
артефакты, resolved config и итоговый status проходят через единый API
`ExperimentRun`. При `clearml.enabled: false` тот же код полностью сохраняет
локальный результат, поэтому pipeline не ветвится по backend логирования.

## Гарантии и ограничения

- Дубликат имени запрещён, если явно не передан `replace=True`.
- Неизвестное имя отклоняется при валидации конфига.
- Неверный конечный тип отклоняется при сборке компонента.
- Реестры не навязывают тип исходного Dataset, sampler, архитектуру модели или
  training loop.
- `imagenet_vid` и `ovis` подключены к protocol endpoint, но их прежние
  незавершённые index builders остаются честными `NotImplementedError` до
  реализации чтения конкретного формата данных.
