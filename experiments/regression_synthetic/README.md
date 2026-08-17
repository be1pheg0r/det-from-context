# Synthetic linear regression

Сквозной тест единого протокола на реальном обучении PyTorch-модели. Worker
генерирует данные `y = 3x + 2 + noise`, обучает `nn.Linear`, пишет train/validation
MSE в локальный JSONL и ClearML, сохраняет checkpoint и возвращает найденные
коэффициенты в `summary.json`.

```powershell
pip install -e .
python experiments/regression_synthetic/run.py
```
