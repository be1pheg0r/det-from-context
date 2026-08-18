"""Адаптер официальной реализации RF-DETR к модельному контракту проекта."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from importlib import import_module
from types import ModuleType
from typing import Any, cast

from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from ..contracts import ContextBatch, DetectionBatch, DetectorOutput
from .detector import (
    DetectorAdapter,
    QueryTransform,
    apply_query_transform,
)


class RFDetrVariant(StrEnum):
    """Поддерживаемые upstream-варианты детектора."""

    NANO = "nano"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

    @property
    def upstream_class_name(self) -> str:
        """Имя публичного класса варианта в пакете :mod:`rfdetr`."""
        return f"RFDETR{self.value.title()}"


class _RFDetrForwardCapture:
    """Подключить адаптер к upstream backbone и decoder на один forward."""

    def __init__(
        self,
        model: nn.Module,
        query_init: Tensor | None,
        query_transform: QueryTransform | None,
        *,
        bbox_reparam: bool,
    ) -> None:
        self.model: nn.Module = model
        self.query_init: Tensor | None = query_init
        self.query_transform: QueryTransform | None = query_transform
        self.bbox_reparam: bool = bbox_reparam
        self.features: list[Tensor] = []
        self.decoder_queries: Tensor | None = None
        self.decoder_reference_points: Tensor | None = None
        self._handles: list[RemovableHandle] = []

    def install(self) -> None:
        """Установить временные hooks без изменения исходного RF-DETR."""
        backbone: nn.Module = _child_module(self.model, "backbone")
        transformer: nn.Module = _child_module(self.model, "transformer")
        decoder: nn.Module = _child_module(transformer, "decoder")
        self._handles = [
            backbone.register_forward_hook(self._capture_backbone),
            decoder.register_forward_pre_hook(self._inject_queries, with_kwargs=True),
            decoder.register_forward_hook(self._capture_decoder),
        ]

    def remove(self) -> None:
        """Всегда снять hooks, включая неуспешный upstream forward."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def require_decoder_queries(self) -> Tensor:
        """Вернуть decoder states или сообщить о несовместимом upstream API."""
        if self.decoder_queries is None:
            raise RuntimeError("RF-DETR decoder не вернул query states")
        return self.decoder_queries

    def require_decoder_reference_points(self) -> Tensor:
        """Вернуть реальные decoder anchors в нормализованных координатах."""
        if self.decoder_reference_points is None:
            raise RuntimeError("RF-DETR decoder не вернул reference points")
        return self.decoder_reference_points

    def _capture_backbone(
        self,
        module: nn.Module,
        inputs: tuple[object, ...],
        output: object,
    ) -> None:
        del module, inputs
        self.features = _backbone_feature_tensors(output)

    def _inject_queries(
        self,
        module: nn.Module,
        inputs: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[tuple[object, ...], dict[str, object]] | None:
        del module
        if self.query_init is None and self.query_transform is None:
            return None
        if not inputs or not isinstance(inputs[0], Tensor):
            raise RuntimeError("неожиданная сигнатура RF-DETR decoder")

        original: Tensor = inputs[0]
        raw_references: object = kwargs.get("refpoints_unsigmoid")
        if not isinstance(raw_references, Tensor):
            raise RuntimeError("RF-DETR decoder не получил Tensor refpoints_unsigmoid")
        references: Tensor = self._normalize_references(raw_references)
        if references.shape != (*original.shape[:2], 4):
            raise RuntimeError(
                "RF-DETR decoder reference points имеют форму "
                f"{tuple(references.shape)}, ожидалось {(*original.shape[:2], 4)}"
            )
        replacement: Tensor
        if self.query_init is not None:
            replacement = self.query_init
            replacement = apply_query_transform(
                original, references, lambda *_: replacement
            )
        else:
            replacement = apply_query_transform(
                original,
                references,
                self.query_transform,
            )
        return ((replacement.contiguous(), *inputs[1:]), kwargs)

    def _capture_decoder(
        self,
        module: nn.Module,
        inputs: tuple[object, ...],
        output: object,
    ) -> None:
        del module, inputs
        if not isinstance(output, tuple) or not output:
            raise RuntimeError("неожиданный выход RF-DETR decoder")
        queries: object = output[0]
        references: object = output[1] if len(output) > 1 else None
        if not isinstance(queries, Tensor) or not isinstance(references, Tensor):
            raise RuntimeError("RF-DETR decoder не вернул Tensor queries/references")
        self.decoder_queries = queries.unsqueeze(0) if queries.ndim == 3 else queries
        normalized: Tensor = self._normalize_references(references)
        self.decoder_reference_points = (
            normalized.unsqueeze(0) if normalized.ndim == 3 else normalized
        )

    def _normalize_references(self, references: Tensor) -> Tensor:
        return references if self.bbox_reparam else references.sigmoid()


class RFDetrAdapter(DetectorAdapter):
    """Тонкая голова над официальным пакетом RF-DETR.

    Архитектура, веса, backbone, projector, decoder и prediction heads остаются
    upstream-кодом. Адаптер только применяет transform к содержимому queries
    на входе decoder и преобразует результат в :class:`DetectorOutput`.
    """

    def __init__(
        self,
        variant: str,
        weights: str | None = None,
        *,
        model_options: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.variant: RFDetrVariant = _parse_variant(variant)
        upstream: object = self._build_upstream(weights, model_options)
        model_context: object = getattr(upstream, "model", None)
        model: object = getattr(model_context, "model", None)
        model_config: object = getattr(upstream, "model_config", None)
        if not isinstance(model, nn.Module):
            raise TypeError("публичный RF-DETR provider не содержит nn.Module")

        hidden_dim: object = getattr(model_config, "hidden_dim", None)
        num_queries: object = getattr(model_config, "num_queries", None)
        if not isinstance(hidden_dim, int) or not isinstance(num_queries, int):
            raise TypeError("RF-DETR model_config не содержит hidden_dim/num_queries")
        self.model: nn.Module = model
        self._dim: int = hidden_dim
        self.num_queries: int = num_queries
        self.bbox_reparam: bool = bool(getattr(model, "bbox_reparam", False))

    @property
    def dim(self) -> int:
        """Размерность upstream decoder queries."""
        return self._dim

    def initial_queries(self, batch: DetectionBatch) -> Tensor:
        """Повторить обучаемые RF-DETR queries для каждого элемента батча."""
        query_embedding: nn.Module = _child_module(self.model, "query_feat")
        if not isinstance(query_embedding, nn.Embedding):
            raise TypeError("RF-DETR query_feat должен быть nn.Embedding")
        query_weights: Tensor = query_embedding.weight
        count: int = query_weights.shape[0] if self.training else self.num_queries
        return query_weights[:count].unsqueeze(0).expand(batch.batch_size, -1, -1)

    def forward(
        self,
        batch: DetectionBatch,
        query_init: Tensor | None = None,
        *,
        query_transform: QueryTransform | None = None,
    ) -> DetectorOutput:
        """Запустить оригинальный forward и привести его к локальному контракту."""
        if query_init is not None and query_transform is not None:
            raise ValueError("query_init and query_transform are mutually exclusive")
        capture = _RFDetrForwardCapture(
            self.model,
            query_init,
            query_transform,
            bbox_reparam=self.bbox_reparam,
        )
        capture.install()
        try:
            raw_output: object = self.model(batch.images)
        finally:
            capture.remove()

        if not isinstance(raw_output, Mapping):
            raise TypeError("RF-DETR forward обязан вернуть mapping")
        predictions: Mapping[str, Any] = cast("Mapping[str, Any]", raw_output)
        logits: Tensor = _prediction_tensor(predictions, "pred_logits")
        boxes: Tensor = _prediction_tensor(predictions, "pred_boxes")
        query_layers: Tensor = capture.require_decoder_queries()
        reference_layers: Tensor = capture.require_decoder_reference_points()
        decoder_layers: list[dict[str, Tensor]] = _decoder_layers(
            predictions,
            query_layers,
            reference_layers,
        )
        queries: Tensor = decoder_layers[-1]["queries"]
        aux: dict[str, Any] = {
            key: value
            for key, value in predictions.items()
            if key not in {"pred_logits", "pred_boxes", "aux_outputs"}
        }
        return DetectorOutput(
            logits=logits,
            boxes=boxes,
            queries=queries,
            reference_points=decoder_layers[-1]["reference_points"],
            features=capture.features,
            decoder_layers=decoder_layers,
            aux=aux,
        )

    def encode_context_frames(self, context: ContextBatch) -> list[Tensor] | None:
        """Переиспользовать upstream backbone для пиксельного контекста."""
        if context.images is None:
            return None
        batch_size, num_frames = context.images.shape[:2]
        if num_frames == 0:
            return []
        flat_images: Tensor = context.images.flatten(0, 1)
        backbone: nn.Module = _child_module(self.model, "backbone")
        tensor_utilities: ModuleType = import_module("rfdetr.utilities.tensors")
        nested_factory: object = getattr(
            tensor_utilities,
            "nested_tensor_from_tensor_list",
            None,
        )
        if not callable(nested_factory):
            raise ImportError(
                "rfdetr.utilities.tensors не экспортирует "
                "nested_tensor_from_tensor_list"
            )
        nested_context: object = nested_factory(list(flat_images))
        raw_features: object = backbone(nested_context)
        return [
            feature.unflatten(0, (batch_size, num_frames))
            for feature in _backbone_feature_tensors(raw_features)
        ]

    def freeze(self, backbone: bool = True, decoder: bool = False) -> None:
        """Заморозить выбранные upstream-части детектора."""
        _set_trainable(_child_module(self.model, "backbone"), not backbone)
        transformer: nn.Module = _child_module(self.model, "transformer")
        _set_trainable(_child_module(transformer, "decoder"), not decoder)

    def _build_upstream(
        self,
        weights: str | None,
        model_options: Mapping[str, Any] | None,
    ) -> object:
        """Создать официальный публичный RF-DETR variant."""
        package: ModuleType = import_module("rfdetr")
        provider: object = getattr(
            package,
            self.variant.upstream_class_name,
            None,
        )
        if not callable(provider):
            raise ImportError(
                f"rfdetr не экспортирует {self.variant.upstream_class_name}"
            )
        factory: Callable[..., object] = cast("Callable[..., object]", provider)
        options: dict[str, Any] = dict(model_options or {})
        if weights is not None:
            configured_weights: object = options.get("pretrain_weights")
            if configured_weights is not None and configured_weights != weights:
                raise ValueError(
                    "weights и model_options['pretrain_weights'] задают разные файлы"
                )
            options["pretrain_weights"] = weights
        return factory(**options)


def _parse_variant(variant: str) -> RFDetrVariant:
    normalized: str = variant.strip().lower().removeprefix("rfdetr-")
    try:
        return RFDetrVariant(normalized)
    except ValueError as error:
        supported: str = ", ".join(item.value for item in RFDetrVariant)
        raise ValueError(
            f"неизвестный RF-DETR variant {variant!r}; доступно: {supported}"
        ) from error


def _child_module(owner: nn.Module, name: str) -> nn.Module:
    child: object = getattr(owner, name, None)
    if not isinstance(child, nn.Module):
        raise TypeError(f"RF-DETR {type(owner).__name__}.{name} должен быть nn.Module")
    return child


def _backbone_feature_tensors(output: object) -> list[Tensor]:
    if not isinstance(output, Sequence) or not output:
        raise RuntimeError("неожиданный выход RF-DETR backbone")
    features: object = output[0]
    if not isinstance(features, Sequence):
        raise RuntimeError("RF-DETR backbone не вернул multi-scale features")

    tensors: list[Tensor] = []
    for feature in features:
        tensor: object = getattr(feature, "tensors", feature)
        if not isinstance(tensor, Tensor):
            raise RuntimeError("RF-DETR backbone feature не является Tensor")
        tensors.append(tensor)
    return tensors


def _prediction_tensor(predictions: Mapping[str, Any], name: str) -> Tensor:
    value: object = predictions.get(name)
    if not isinstance(value, Tensor):
        raise RuntimeError(f"RF-DETR output не содержит Tensor {name!r}")
    return value


def _decoder_layers(
    predictions: Mapping[str, Any],
    queries: Tensor,
    references: Tensor,
) -> list[dict[str, Tensor]]:
    raw_aux: object = predictions.get("aux_outputs", [])
    if not isinstance(raw_aux, Sequence):
        raise RuntimeError("RF-DETR aux_outputs должен быть последовательностью")
    layer_predictions: list[Mapping[str, Any]] = []
    for layer in raw_aux:
        if not isinstance(layer, Mapping):
            raise RuntimeError("RF-DETR aux layer должен быть mapping")
        layer_predictions.append(cast("Mapping[str, Any]", layer))
    layer_predictions.append(predictions)

    if queries.ndim != 4 or queries.shape[0] < len(layer_predictions):
        raise RuntimeError(
            "число captured RF-DETR queries не совпадает с prediction layers"
        )
    selected_queries: Tensor = queries[-len(layer_predictions) :]
    if references.ndim != 4:
        raise RuntimeError("captured RF-DETR reference points должны иметь 4 измерения")
    if references.shape[0] == 1:
        selected_references: Tensor = references.expand(
            len(layer_predictions),
            -1,
            -1,
            -1,
        )
    elif references.shape[0] >= len(layer_predictions):
        selected_references = references[-len(layer_predictions) :]
    else:
        raise RuntimeError(
            "число captured RF-DETR reference points не совпадает с prediction layers"
        )
    return [
        {
            "queries": layer_queries,
            "logits": _prediction_tensor(layer, "pred_logits"),
            "boxes": _prediction_tensor(layer, "pred_boxes"),
            "reference_points": layer_references,
        }
        for layer, layer_queries, layer_references in zip(
            layer_predictions,
            selected_queries,
            selected_references,
            strict=True,
        )
    ]


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)
