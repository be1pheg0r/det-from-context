# Synthetic linear regression

Сквозной тест трёх протоколов на реальном обучении PyTorch-модели:

- `datasets/synthetic_regression` владеет генератором, конфигом и tensor artifacts;
- `models/linear_regression` владеет `nn.Linear`, конфигом и checkpoint artifacts;
- эта папка владеет train loop и experiment config.

`ExperimentProtocol.execute_components` строит `DataLoader` и `nn.Module` после
инициализации ClearML Task. Worker пишет train/validation MSE, а component
artifacts автоматически копируются в run directory и загружаются в ClearML.

```powershell
pip install -e .
python experiments/regression_synthetic/run.py
```

Итоговый путь печатается в stdout. В `metadata.json` находятся ClearML task id,
dashboard URL, component directories и опубликованные artifact names.
