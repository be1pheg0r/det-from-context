# Dummy protocol experiment

Эталонная директория одного эксперимента содержит:

- `config.yaml` — полный Hydra-конфиг;
- `run.py` — короткий скрипт запуска;
- `worker.py` — исходный код конкретного эксперимента.

Запуск без ClearML:

```powershell
pip install -e .
python experiments/dummy_protocol/run.py
```

Запуск с ClearML после заполнения `.env`:

```powershell
$env:CLEARML_ENABLED="true"
python experiments/dummy_protocol/run.py --set train.seed=123
```
