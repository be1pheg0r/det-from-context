# RF-DETR artifacts

Сюда можно положить пользовательский checkpoint и указать его относительным
путём в `../config.yaml`, например
`model.pretrain_weights: artifacts/model.pth`. Бинарные веса
игнорируются Git; experiment protocol копирует созданные runtime-артефакты в
изолированную директорию запуска.
