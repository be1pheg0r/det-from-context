# Универсальный image dataloader

Компонент читает детекционный датасет из двух независимых корней — изображений и
аннотаций — и возвращает стандартный `DetectionBatch`. COCO-конвертация на диске не
нужна: изображения и targets проходят официальные transforms RF-DETR синхронно.

## Разбиения

Для детерминированного разбиения одного набора файлов:

```yaml
splits:
  mode: generated
  train_fraction: 0.8
  validation_fraction: 0.1
  test_fraction: 0.1
```

Один seed эксперимента формирует общую перестановку, после чего она разрезается на
непересекающиеся train/validation/test. Для заранее зафиксированных разбиений:

```text
images/
├── train/
├── val/
└── test/
annotations/
├── train/
├── val/
└── test/
```

```yaml
splits:
  mode: predefined
  train_dir: train
  validation_dir: val
  test_dir: test
```

Имена относительных файлов изображения и аннотации должны совпадать, расширения
задаются отдельно. При `strict_pairs: true` компонент останавливает запуск как при
пропущенной аннотации, так и при orphan-аннотации. Точный manifest каждого split и
его SHA-256 автоматически входят в артефакты RF-DETR-эксперимента.

### Стратификация и class balance

Для long-tail detection включается отдельный блок:

```yaml
class_balance:
  stratify_generated: true
  sampling: inverse_sqrt       # none | inverse_sqrt | inverse_frequency
  max_sample_weight: 5.0
```

Generated split использует детерминированную multilabel-стратификацию по наличию
класса в изображении. Сначала распределяются изображения с самым редким классом;
при назначении одновременно обновляются квоты всех классов этого изображения.
Размеры split'ов остаются точными, а validation/test никогда не ресэмплируются.
`predefined` директории не переставляются, поскольку их фиксированный состав имеет
приоритет.

Train sampler считает число изображений `n_c`, содержащих каждый класс. Для
`inverse_sqrt` вес класса равен
`min(max_sample_weight, sqrt(max(n_c) / n_c))`, для `inverse_frequency` — то же
отношение без квадратного корня. Вес multilabel-изображения равен максимальному
весу его классов; пустое изображение получает вес 1. Epoch сохраняет исходное
число samples, но `WeightedRandomSampler(replacement=True)` чаще показывает редкие
классы. Квадратный корень выбран по умолчанию как менее агрессивный режим, который
снижает риск переобучения на нескольких редких кадрах.

RF-DETR не предоставляет per-class weights: его `focal_alpha` разделяет positive и
negative targets, а `ia_bce_loss` использует IoU, но оба механизма не учитывают
частоту конкретного класса. Поэтому балансировка реализована вокруг официального
trainer и не изменяет upstream matcher/criterion. Полные image/object counts,
class/sample weights и признак replacement сохраняются в `dataset-splits.json` и
metadata ClearML.

## Форматы аннотаций

Встроенный `bdd100k_json` читает отдельный JSON на изображение и возвращает
абсолютные `xyxy`. Новый формат добавляется реализацией протокола
`AnnotationReader.read(path, classes, image_size)` и регистрацией через
`register_annotation_reader`. Dataset при этом менять не требуется: он отвечает за
pairing, валидацию targets и официальные RF-DETR transforms, а reader — только за
разбор исходного файла.

Классы в `config.yaml` обязаны иметь непрерывные ID от нуля. Некорректные,
нечисловые и вырожденные boxes фильтруются reader-ом; после transforms дополнительно
проверяются форма, конечность и диапазон нормализованных координат.
