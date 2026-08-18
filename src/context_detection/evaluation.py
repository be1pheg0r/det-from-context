"""Метрики детекции и сборка результатов абляций.

Функции в этом модуле не управляют моделью и не запускают Hydra sweep. Worker
эксперимента собирает predictions/targets для каждого варианта, передаёт их в
``coco_ap`` и пишет полученный плоский словарь через
``ExperimentRun.log_metrics``.

Внутренний формат боксов проекта — normalized ``cxcywh``. Конвертация в
абсолютный COCO ``xywh`` выполняется только на границе этого модуля.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import redirect_stdout
from io import StringIO
from typing import Any

import torch
from torch import Tensor

from .contracts import DetectorOutput

MetricMap = Mapping[str, float | int]
DetectionRecord = Mapping[str, Any]

_COCO_STAT_NAMES: tuple[str, ...] = (
    "map",
    "map_50",
    "map_75",
    "map_small",
    "map_medium",
    "map_large",
    "mar_1",
    "mar_10",
    "mar_100",
    "mar_small",
    "mar_medium",
    "mar_large",
)
_DEFAULT_DELTA_METRICS: tuple[str, ...] = ("map", "map_50")


def detector_output_to_predictions(
    output: DetectorOutput,
    image_ids: Sequence[int] | Tensor | None = None,
    *,
    score_threshold: float = 0.05,
    max_detections: int = 100,
) -> list[dict[str, Tensor | int]]:
    """Преобразовать финальный выход детектора в записи для ``coco_ap``.

    Текущий RF-DETR использует независимые foreground logits, поэтому score —
    максимальная sigmoid-вероятность по классам. Адаптер с softmax/no-object
    семантикой должен подготовить predictions самостоятельно.
    """
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold должен быть в [0, 1]")
    if max_detections <= 0:
        raise ValueError("max_detections должен быть положительным")

    batch_size: int = output.logits.shape[0]
    normalized_ids: list[int] = _normalize_image_ids(image_ids, batch_size)
    probabilities: Tensor = output.logits.detach().sigmoid()
    scores, labels = probabilities.max(dim=-1)
    boxes: Tensor = output.boxes.detach()

    predictions: list[dict[str, Tensor | int]] = []
    for batch_index, image_id in enumerate(normalized_ids):
        keep: Tensor = scores[batch_index] >= score_threshold
        selected_scores: Tensor = scores[batch_index][keep]
        selected_labels: Tensor = labels[batch_index][keep]
        selected_boxes: Tensor = boxes[batch_index][keep]
        if selected_scores.numel() > max_detections:
            order: Tensor = selected_scores.argsort(descending=True)[:max_detections]
            selected_scores = selected_scores[order]
            selected_labels = selected_labels[order]
            selected_boxes = selected_boxes[order]
        predictions.append(
            {
                "image_id": image_id,
                "boxes": selected_boxes.cpu(),
                "scores": selected_scores.cpu(),
                "labels": selected_labels.cpu(),
            }
        )
    return predictions


def coco_ap(
    predictions: Sequence[DetectionRecord],
    targets: Sequence[DetectionRecord],
) -> dict[str, float]:
    """Посчитать COCO bbox AP/AR для normalized ``cxcywh`` записей.

    Каждая prediction-запись содержит ``boxes [N,4]``, ``scores [N]`` и
    ``labels [N]``. Target содержит ``boxes [M,4]`` и ``labels [M]``.
    ``image_id`` опционален с обеих сторон; без него используется позиция в
    последовательности. Target также может содержать ``ignore [M]``: такие
    объекты не участвуют в AP, а совпавшие с ними detections не становятся
    false positive. Это используется subset-метриками ниже.

    Для стандартных area-метрик target должен нести ``image_size=[H,W]``
    (также принимаются ``orig_size`` или ``size``). Если размеров нет, IoU и
    общие AP/AR остаются корректными на единичном холсте, но small/medium/large
    метрики намеренно не возвращаются.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions ({len(predictions)}) и targets ({len(targets)}) "
            "должны описывать одинаковое число изображений"
        )
    if not targets:
        raise ValueError("нельзя посчитать AP для пустого набора изображений")

    images: list[dict[str, int]] = []
    ground_truth: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    labels_seen: set[int] = set()
    image_ids_seen: set[int] = set()
    all_sizes_known: bool = True
    annotation_id: int = 1
    detection_id: int = 1

    prepared: list[
        tuple[int, Tensor, Tensor, Tensor, Tensor, Tensor, tuple[int, int], bool]
    ] = []
    for index, (prediction, target) in enumerate(
        zip(predictions, targets, strict=True)
    ):
        image_id: int = _paired_image_id(prediction, target, index)
        if image_id in image_ids_seen:
            raise ValueError(f"повторяющийся image_id={image_id}")
        image_ids_seen.add(image_id)

        target_boxes, target_labels = _target_tensors(target, index)
        prediction_boxes, prediction_labels, prediction_scores = _prediction_tensors(
            prediction, index
        )
        image_size, size_known = _record_image_size(target)
        all_sizes_known &= size_known
        labels_seen.update(int(value) for value in target_labels.tolist())
        labels_seen.update(int(value) for value in prediction_labels.tolist())
        prepared.append(
            (
                image_id,
                target_boxes,
                target_labels,
                prediction_boxes,
                prediction_labels,
                prediction_scores,
                image_size,
                size_known,
            )
        )
        images.append({"id": image_id, "height": image_size[0], "width": image_size[1]})

    target_count: int = sum(item[1].shape[0] for item in prepared)
    if not target_count:
        raise ValueError("AP не определён: во всех targets отсутствуют объекты")
    if not labels_seen:
        raise ValueError("AP не определён: отсутствуют категории")

    category_map: dict[int, int] = {
        label: index for index, label in enumerate(sorted(labels_seen), start=1)
    }
    categories: list[dict[str, Any]] = [
        {"id": category_id, "name": str(label)}
        for label, category_id in category_map.items()
    ]

    for index, item in enumerate(prepared):
        (
            image_id,
            target_boxes,
            target_labels,
            prediction_boxes,
            prediction_labels,
            prediction_scores,
            image_size,
            _,
        ) = item
        target_xywh: Tensor = _normalized_cxcywh_to_xywh(target_boxes, image_size)
        prediction_xywh: Tensor = _normalized_cxcywh_to_xywh(
            prediction_boxes, image_size
        )
        iscrowd: Tensor = _optional_vector(
            targets[index], "iscrowd", target_boxes.shape[0], torch.int64
        )
        ignored: Tensor = _optional_vector(
            targets[index], "ignore", target_boxes.shape[0], torch.bool
        )
        # COCOeval для bbox реализует ignore через iscrowd. Так detections на
        # объектах вне выбранного subset не штрафуют метрику как false positive.
        evaluation_crowd: Tensor = iscrowd.bool() | ignored
        for box, label, crowd in zip(
            target_xywh,
            target_labels,
            evaluation_crowd,
            strict=True,
        ):
            xywh: list[float] = [float(value) for value in box.tolist()]
            ground_truth.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_map[int(label)],
                    "bbox": xywh,
                    "area": xywh[2] * xywh[3],
                    "iscrowd": int(crowd),
                }
            )
            annotation_id += 1

        for box, label, score in zip(
            prediction_xywh,
            prediction_labels,
            prediction_scores,
            strict=True,
        ):
            xywh = [float(value) for value in box.tolist()]
            detections.append(
                {
                    "id": detection_id,
                    "image_id": image_id,
                    "category_id": category_map[int(label)],
                    "bbox": xywh,
                    "area": xywh[2] * xywh[3],
                    "iscrowd": 0,
                    "score": float(score),
                }
            )
            detection_id += 1

    stats: list[float] = _run_coco_eval(
        images,
        categories,
        ground_truth,
        detections,
    )
    metrics: dict[str, float] = {
        name: _finite_float(stats[index], name)
        for index, name in enumerate(_COCO_STAT_NAMES)
    }
    if not all_sizes_known:
        for name in (
            "map_small",
            "map_medium",
            "map_large",
            "mar_small",
            "mar_medium",
            "mar_large",
        ):
            metrics.pop(name)
    return metrics


def subset_coco_ap(
    predictions: Sequence[DetectionRecord],
    targets: Sequence[DetectionRecord],
    mask_key: str,
) -> dict[str, float]:
    """Посчитать COCO AP только по объектам, отмеченным boolean-маской.

    В каждом target поле ``mask_key`` должно иметь форму ``[M]`` и совпадать
    с количеством target boxes. Не выбранные объекты остаются ignore-регионами:
    корректные detections на них не ухудшают AP выбранного subset.

    Типичные поля: ``occluded`` и ``newly_appeared``.
    """
    if not mask_key:
        raise ValueError("mask_key не должен быть пустым")

    subset_targets: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        boxes, _ = _target_tensors(target, index)
        if mask_key not in target:
            raise ValueError(f"targets[{index}] не содержит поле {mask_key!r}")
        selected: Tensor = _boolean_vector(
            target[mask_key],
            boxes.shape[0],
            f"targets[{index}].{mask_key}",
        )
        existing_ignore: Tensor = _optional_vector(
            target,
            "ignore",
            boxes.shape[0],
            torch.bool,
        ).bool()
        prepared_target: dict[str, Any] = dict(target)
        prepared_target["ignore"] = existing_ignore | ~selected
        subset_targets.append(prepared_target)

    return coco_ap(predictions, subset_targets)


def add_newly_appeared_flags(
    targets: Sequence[DetectionRecord],
) -> list[dict[str, Any]]:
    """Пометить первое появление каждого instance в каждой видеосцене.

    Требуемые target-поля: scalar ``sequence_id``, scalar ``frame_id`` и
    ``instance_ids [M]``. Записи можно передавать в любом порядке: внутри
    каждой сцены они сортируются по ``frame_id``, а результат возвращается в
    исходном порядке. Повторное появление после пропуска кадров новым объектом
    не считается.

    Для корректной семантики передавайте полный validation-фрагмент сцены, а
    не клип, вырезанный после того, как объект уже появился.
    """
    prepared: list[dict[str, Any]] = [dict(target) for target in targets]
    frames_by_sequence: defaultdict[Any, list[tuple[int, int, list[Any]]]] = (
        defaultdict(list)
    )

    for index, target in enumerate(targets):
        if "sequence_id" not in target:
            raise ValueError(f"targets[{index}] не содержит поле 'sequence_id'")
        if "frame_id" not in target:
            raise ValueError(f"targets[{index}] не содержит поле 'frame_id'")
        boxes, _ = _target_tensors(target, index)
        instance_ids: list[Any] = _instance_ids(
            target.get("instance_ids"),
            boxes.shape[0],
            f"targets[{index}].instance_ids",
        )
        sequence_id: Any = _hashable_scalar(
            target["sequence_id"], f"targets[{index}].sequence_id"
        )
        frame_id: int = _integer_scalar(
            target["frame_id"], f"targets[{index}].frame_id"
        )
        frames_by_sequence[sequence_id].append((frame_id, index, instance_ids))

    for sequence_id, frames in frames_by_sequence.items():
        seen: set[Any] = set()
        previous_frame_id: int | None = None
        for frame_id, index, instance_ids in sorted(frames):
            if frame_id == previous_frame_id:
                raise ValueError(
                    f"повторяющийся frame_id={frame_id} в sequence_id={sequence_id!r}"
                )
            flags: list[bool] = []
            for instance_id in instance_ids:
                flags.append(instance_id not in seen)
                seen.add(instance_id)
            prepared[index]["newly_appeared"] = torch.tensor(flags, dtype=torch.bool)
            previous_frame_id = frame_id
    return prepared


def occluded_coco_ap(
    predictions: Sequence[DetectionRecord],
    targets: Sequence[DetectionRecord],
) -> dict[str, float]:
    """COCO AP только для targets с ``occluded=True``."""
    return subset_coco_ap(predictions, targets, "occluded")


def newly_appeared_coco_ap(
    predictions: Sequence[DetectionRecord],
    targets: Sequence[DetectionRecord],
) -> dict[str, float]:
    """COCO AP на первом размеченном появлении каждого instance."""
    return subset_coco_ap(
        predictions, add_newly_appeared_flags(targets), "newly_appeared"
    )


def metric_deltas(
    metrics: MetricMap,
    baseline_metrics: MetricMap,
    *,
    names: Iterable[str] | None = None,
) -> dict[str, float]:
    """Вернуть ``delta_<name> = metric - baseline`` для общих метрик."""
    selected: list[str] = (
        list(names)
        if names is not None
        else sorted(set(metrics).intersection(baseline_metrics))
    )
    deltas: dict[str, float] = {}
    for name in selected:
        if name not in metrics or name not in baseline_metrics:
            raise KeyError(f"метрика {name!r} отсутствует в одном из запусков")
        current: float = _finite_float(metrics[name], name)
        baseline: float = _finite_float(baseline_metrics[name], name)
        deltas[f"delta_{name}"] = current - baseline
    return deltas


def failure_mode_grid(
    runs: Mapping[str, MetricMap],
    baseline: str = "baseline",
    *,
    delta_metrics: Iterable[str] = _DEFAULT_DELTA_METRICS,
) -> dict[str, float]:
    """Собрать плоский словарь результатов нескольких ablation-прогонов.

    Hydra sweep и инференс остаются в experiment worker. Здесь только упаковка
    уже посчитанных результатов в формат, совместимый с ``run.log_metrics``.

    Пример ключей: ``baseline.map``, ``memot.map``, ``memot.delta_map``.
    """
    if baseline not in runs:
        raise KeyError(f"нет baseline-запуска {baseline!r}")
    baseline_metrics: MetricMap = runs[baseline]
    flattened: dict[str, float] = {}
    selected_deltas: tuple[str, ...] = tuple(delta_metrics)

    for run_name, metrics in runs.items():
        if not run_name or "." in run_name:
            raise ValueError("имя запуска должно быть непустым и не содержать '.'")
        for metric_name, value in metrics.items():
            flattened[f"{run_name}.{metric_name}"] = _finite_float(
                value, f"{run_name}.{metric_name}"
            )
        if run_name == baseline:
            continue
        deltas: dict[str, float] = metric_deltas(
            metrics,
            baseline_metrics,
            names=selected_deltas,
        )
        for delta_name, value in deltas.items():
            flattened[f"{run_name}.{delta_name}"] = value
    return flattened


def memory_diagnostics_summary(
    diagnostics_log: Iterable[Mapping[str, Any] | Any],
) -> dict[str, float]:
    """Агрегировать scalar diagnostics памяти в experiment-friendly dict.

    Принимаются как сами словари, так и объекты с атрибутом ``diagnostics``
    (например, ``ContextOutput``). Многомерные attention weights пропускаются.
    ``evicted`` суммируется, остальные известные показатели усредняются.
    """
    values: defaultdict[str, list[float]] = defaultdict(list)
    for entry in diagnostics_log:
        diagnostics: Any = (
            entry if isinstance(entry, Mapping) else getattr(entry, "diagnostics", None)
        )
        if not isinstance(diagnostics, Mapping):
            raise TypeError("элемент diagnostics_log не содержит mapping diagnostics")
        for name, value in diagnostics.items():
            scalar: float | None = _optional_scalar(value)
            if scalar is not None:
                values[name].append(scalar)

    summary: dict[str, float] = {}
    for name in ("active_slots", "mean_age", "write_rate", "mean_missed"):
        if values[name]:
            output_name: str = (
                f"memory_{name}" if name.startswith("mean_") else f"memory_{name}_mean"
            )
            summary[output_name] = sum(values[name]) / len(values[name])
    if values["evicted"]:
        summary["memory_evicted_total"] = sum(values["evicted"])
    return summary


def _run_coco_eval(
    images: list[dict[str, int]],
    categories: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    detections: list[dict[str, Any]],
) -> list[float]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise RuntimeError(
            "для coco_ap нужен pycocotools; установите зависимости проекта"
        ) from error

    with redirect_stdout(StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = {
            "info": {},
            "licenses": [],
            "images": images,
            "categories": categories,
            "annotations": ground_truth,
        }
        coco_gt.createIndex()

        coco_dt = COCO()
        coco_dt.dataset = {
            "info": {},
            "licenses": [],
            "images": images,
            "categories": categories,
            "annotations": detections,
        }
        coco_dt.createIndex()

        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = [image["id"] for image in images]
        evaluator.params.catIds = [category["id"] for category in categories]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return [float(value) for value in evaluator.stats]


def _normalize_image_ids(
    image_ids: Sequence[int] | Tensor | None,
    batch_size: int,
) -> list[int]:
    if image_ids is None:
        return list(range(batch_size))
    if isinstance(image_ids, Tensor):
        normalized: list[int] = [int(value) for value in image_ids.flatten().tolist()]
    else:
        normalized = [int(value) for value in image_ids]
    if len(normalized) != batch_size:
        raise ValueError(
            f"image_ids содержит {len(normalized)} значений, ожидалось {batch_size}"
        )
    return normalized


def _paired_image_id(
    prediction: DetectionRecord,
    target: DetectionRecord,
    fallback: int,
) -> int:
    prediction_id: int | None = _optional_image_id(prediction)
    target_id: int | None = _optional_image_id(target)
    if prediction_id is not None and target_id is not None:
        if prediction_id != target_id:
            raise ValueError(
                f"image_id prediction={prediction_id} не совпадает с "
                f"target={target_id} на позиции {fallback}"
            )
        return target_id
    if target_id is not None:
        return target_id
    if prediction_id is not None:
        return prediction_id
    return fallback


def _optional_image_id(record: DetectionRecord) -> int | None:
    if "image_id" not in record:
        return None
    value: Any = record["image_id"]
    tensor: Tensor = torch.as_tensor(value)
    if tensor.numel() != 1:
        raise ValueError("image_id должен быть скаляром")
    return int(tensor.item())


def _record_image_size(record: DetectionRecord) -> tuple[tuple[int, int], bool]:
    for key in ("image_size", "orig_size", "size"):
        if key not in record:
            continue
        size: Tensor = torch.as_tensor(record[key]).flatten()
        if size.numel() != 2:
            raise ValueError(f"{key} должен содержать [H, W]")
        height, width = (int(size[0]), int(size[1]))
        if height <= 0 or width <= 0:
            raise ValueError(f"{key} должен содержать положительные H и W")
        return (height, width), True
    return (1, 1), False


def _target_tensors(record: DetectionRecord, index: int) -> tuple[Tensor, Tensor]:
    boxes: Tensor = _boxes(record.get("boxes"), f"targets[{index}].boxes")
    labels: Tensor = _labels(record.get("labels"), f"targets[{index}].labels")
    if boxes.shape[0] != labels.shape[0]:
        raise ValueError(f"targets[{index}]: число boxes и labels не совпадает")
    return boxes, labels


def _prediction_tensors(
    record: DetectionRecord,
    index: int,
) -> tuple[Tensor, Tensor, Tensor]:
    boxes: Tensor = _boxes(record.get("boxes"), f"predictions[{index}].boxes")
    labels: Tensor = _labels(record.get("labels"), f"predictions[{index}].labels")
    if "scores" not in record:
        raise ValueError(f"не задано поле predictions[{index}].scores")
    scores: Tensor = (
        torch.as_tensor(record["scores"], dtype=torch.float32).detach().cpu()
    )
    if scores.ndim != 1:
        raise ValueError(f"predictions[{index}].scores должен иметь форму [N]")
    if boxes.shape[0] != labels.shape[0] or boxes.shape[0] != scores.shape[0]:
        raise ValueError(
            f"predictions[{index}]: число boxes, labels и scores не совпадает"
        )
    if scores.numel() and (
        not torch.isfinite(scores).all() or scores.min() < 0 or scores.max() > 1
    ):
        raise ValueError(f"predictions[{index}].scores должен быть конечным в [0,1]")
    return boxes, labels, scores


def _boxes(value: Any, name: str) -> Tensor:
    if value is None:
        raise ValueError(f"не задано поле {name}")
    boxes: Tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"{name} должен иметь форму [N,4]")
    if boxes.numel() and (
        not torch.isfinite(boxes).all() or boxes.min() < 0 or boxes.max() > 1
    ):
        raise ValueError(f"{name} должен быть normalized cxcywh в [0,1]")
    return boxes


def _labels(value: Any, name: str) -> Tensor:
    if value is None:
        raise ValueError(f"не задано поле {name}")
    labels: Tensor = torch.as_tensor(value, dtype=torch.int64).detach().cpu()
    if labels.ndim != 1:
        raise ValueError(f"{name} должен иметь форму [N]")
    if labels.numel() and labels.min() < 0:
        raise ValueError(f"{name} содержит отрицательную категорию")
    return labels


def _optional_vector(
    record: DetectionRecord,
    name: str,
    length: int,
    dtype: torch.dtype,
) -> Tensor:
    if name not in record:
        return torch.zeros(length, dtype=dtype)
    value: Tensor = torch.as_tensor(record[name], dtype=dtype).detach().cpu()
    if value.shape != (length,):
        raise ValueError(f"{name} должен иметь форму [{length}]")
    return value


def _boolean_vector(value: Any, length: int, name: str) -> Tensor:
    vector: Tensor = torch.as_tensor(value).detach().cpu()
    if vector.shape != (length,):
        raise ValueError(f"{name} должен иметь форму [{length}]")
    if vector.dtype != torch.bool:
        if vector.numel() and not bool(((vector == 0) | (vector == 1)).all()):
            raise ValueError(f"{name} должен содержать только bool/0/1")
        vector = vector.bool()
    return vector


def _instance_ids(value: Any, length: int, name: str) -> list[Any]:
    if value is None:
        raise ValueError(f"не задано поле {name}")
    if isinstance(value, Tensor):
        if value.ndim != 1:
            raise ValueError(f"{name} должен иметь форму [M]")
        normalized: list[Any] = value.detach().cpu().tolist()
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        normalized = list(value)
    else:
        raise ValueError(f"{name} должен быть одномерной последовательностью")
    if len(normalized) != length:
        raise ValueError(
            f"{name} содержит {len(normalized)} значений, ожидалось {length}"
        )
    for instance_id in normalized:
        if instance_id is None:
            raise ValueError(f"{name} содержит пустой instance id")
        try:
            hash(instance_id)
        except TypeError as error:
            raise ValueError(f"{name} содержит нехешируемый instance id") from error
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} содержит повторяющийся instance id в одном кадре")
    return normalized


def _hashable_scalar(value: Any, name: str) -> Any:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} должен быть скаляром")
        value = value.item()
    if value is None:
        raise ValueError(f"{name} не должен быть пустым")
    try:
        hash(value)
    except TypeError as error:
        raise ValueError(f"{name} должен быть хешируемым скаляром") from error
    return value


def _integer_scalar(value: Any, name: str) -> int:
    tensor: Tensor = torch.as_tensor(value)
    if tensor.numel() != 1:
        raise ValueError(f"{name} должен быть скаляром")
    scalar: float = float(tensor.item())
    if not math.isfinite(scalar) or not scalar.is_integer():
        raise ValueError(f"{name} должен быть конечным целым числом")
    return int(scalar)


def _normalized_cxcywh_to_xywh(
    boxes: Tensor,
    image_size: tuple[int, int],
) -> Tensor:
    if not boxes.numel():
        return boxes.clone()
    height, width = image_size
    center_x, center_y, box_width, box_height = boxes.unbind(-1)
    x1: Tensor = ((center_x - box_width / 2) * width).clamp(0, width)
    y1: Tensor = ((center_y - box_height / 2) * height).clamp(0, height)
    x2: Tensor = ((center_x + box_width / 2) * width).clamp(0, width)
    y2: Tensor = ((center_y + box_height / 2) * height).clamp(0, height)
    return torch.stack((x1, y1, (x2 - x1).clamp_min(0), (y2 - y1).clamp_min(0)), -1)


def _finite_float(value: float | int, name: str) -> float:
    result: float = float(value)
    if not math.isfinite(result):
        raise ValueError(f"метрика {name!r} должна быть конечной")
    return result


def _optional_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return _finite_float(value, "diagnostics")
    if isinstance(value, Tensor) and value.numel() == 1:
        return _finite_float(float(value.detach().cpu()), "diagnostics")
    return None
