# RF-DETR image class adaptation

Этот smoke-эксперимент обучает ровно два epoch на `image_dataloader` и
замораживает весь RF-DETR, кроме `class_embed`. Метрики loss/mAP, checkpoint,
кривые обучения и сетки предсказаний сохраняются как ClearML artifacts.

Datasphere-запуск:

```powershell
.venv\Scripts\python.exe experiments/rfdetr_image/submit_datasphere.py
```

`DATASPHERE_PROJECT_ID` остаётся в `.env`; `.env` не передаётся на VM. Все
пути S3 и ID S3 connector находятся в [`paths.yaml`](paths.yaml). В секции
`data` задаются независимые `images_dir` и `annotations_dir`; измените только
этот файл, если изменилась структура Object Storage. Эти значения передаются
dataset-component через его конфиг.
