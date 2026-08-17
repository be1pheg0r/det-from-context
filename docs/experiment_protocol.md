# Единый протокол экспериментов

Каждый эксперимент хранится в отдельной директории с полным Hydra-конфигом,
скриптом запуска и исходниками worker-а. Данные передаются полем
`data.config_path`: этот файл является ссылкой на отдельный конфиг датасета и
не подменяет машинный `data.root`.

Запуск создаёт директорию `output.root/<experiment>/<run-id>/` со структурой:

```text
artifacts/       произвольные результаты
checkpoints/     контрольные точки
logs/            текстовый журнал
sources/         снимок конфига, launch script и переданных исходников
config.yaml      полностью скомпонованный Hydra-конфиг
metadata.json    статус, время, overrides и ссылка на датасет
metrics.jsonl    метрики по одной записи на вызов
summary.json     итоговое резюме
```

Секреты не входят в Hydra-конфиг и снимок исходников. `ExperimentProtocol`
загружает `.env`; ClearML SDK получает из него `CLEARML_API_ACCESS_KEY`,
`CLEARML_API_SECRET_KEY` и адреса серверов. Интеграция включается через
`clearml.enabled=true` или переменную `CLEARML_ENABLED=true` в эталонном
эксперименте.

Worker принимает `ExperimentRun` и `ExperimentConfig`, пишет скаляры через
`log_metrics`, сохраняет файлы через `save_artifact` и возвращает словарь для
`summary.json`. Context manager гарантирует статусы `completed`/`failed` и
синхронное завершение ClearML task.
