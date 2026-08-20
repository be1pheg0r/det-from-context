"""Адаптер официальной реализации RF-DETR к модельному контракту проекта."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from importlib import import_module
from types import ModuleType
from typing import Any, Final, cast

from rfdetr.utilities import box_ops
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from ..contracts import ContextBatch, DetectionBatch, DetectorOutput
from .detector import DetectorAdapter


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


# Official input sizes of the released RF-DETR detection checkpoints.
RFDETR_PRETRAINED_RESOLUTIONS: Final[dict[RFDetrVariant, int]] = {
    RFDetrVariant.NANO: 384,
    RFDetrVariant.SMALL: 512,
    RFDetrVariant.MEDIUM: 576,
    RFDetrVariant.LARGE: 704,
}


def rfdetr_pretrained_resolution(variant: str | RFDetrVariant) -> int:
    """Return the fixed input resolution for an official pretrained variant."""
    parsed = variant if isinstance(variant, RFDetrVariant) else _parse_variant(variant)
    return RFDETR_PRETRAINED_RESOLUTIONS[parsed]


class _RFDetrForwardCapture:
    """Подключить адаптер к upstream backbone и decoder на один forward."""

    def __init__(self, model: nn.Module, query_init: Tensor | None) -> None:
        self.model: nn.Module = model
        self.query_init: Tensor | None = query_init
        self.features: list[Tensor] = []
        self.decoder_queries: Tensor | None = None
        self._handles: list[RemovableHandle] = []

    def install(self) -> None:
        """Установить временные hooks без изменения исходного RF-DETR."""
        backbone: nn.Module = _child_module(self.model, "backbone")
        transformer: nn.Module = _child_module(self.model, "transformer")
        decoder: nn.Module = _child_module(transformer, "decoder")
        self._handles = [
            backbone.register_forward_hook(self._capture_backbone),
            decoder.register_forward_pre_hook(self._inject_queries),
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
    ) -> tuple[object, ...] | None:
        del module
        if self.query_init is None:
            return None
        if not inputs or not isinstance(inputs[0], Tensor):
            raise RuntimeError("неожиданная сигнатура RF-DETR decoder")

        original: Tensor = inputs[0]
        query_init: Tensor = self.query_init
        if query_init.shape != original.shape:
            raise ValueError(
                "query_init имеет форму "
                f"{tuple(query_init.shape)}, ожидалось {tuple(original.shape)}"
            )
        if query_init.device != original.device or query_init.dtype != original.dtype:
            raise ValueError(
                "query_init должен совпадать с RF-DETR queries по device и dtype"
            )
        return (query_init.contiguous(), *inputs[1:])

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
        if not isinstance(queries, Tensor):
            raise RuntimeError("RF-DETR decoder не вернул Tensor queries")
        self.decoder_queries = queries.unsqueeze(0) if queries.ndim == 3 else queries


class RFDetrAdapter(DetectorAdapter):
    """Тонкая голова над официальным пакетом RF-DETR.

    Архитектура, веса, backbone, projector, decoder и prediction heads остаются
    upstream-кодом. Адаптер только подменяет вход decoder при переданном
    ``query_init`` и преобразует результат в :class:`DetectorOutput`.
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
        self.input_resolution: int = rfdetr_pretrained_resolution(self.variant)
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
        self.model_config: Any = model_config
        self._dim: int = hidden_dim
        self.num_queries: int = num_queries

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
        self, batch: DetectionBatch, query_init: Tensor | None = None
    ) -> DetectorOutput:
        """Запустить оригинальный forward и привести его к локальному контракту."""
        capture = _RFDetrForwardCapture(self.model, query_init)
        capture.install()
        try:
            block_size = int(getattr(self.model_config, "patch_size", 1)) * int(
                getattr(self.model_config, "num_windows", 1)
            )
            samples = nested_tensor_from_tensor_list(
                list(batch.images.unbind(0)), block_size=block_size
            )
            targets = batch.targets if self.model.training else None
            raw_output: object = self.model(samples, targets)
        finally:
            capture.remove()

        if not isinstance(raw_output, Mapping):
            raise TypeError("RF-DETR forward обязан вернуть mapping")
        predictions: Mapping[str, Any] = cast("Mapping[str, Any]", raw_output)
        logits: Tensor = _prediction_tensor(predictions, "pred_logits")
        boxes: Tensor = _normalized_prediction_boxes(
            predictions,
            bbox_reparam=bool(getattr(self.model_config, "bbox_reparam", False)),
        )
        query_layers: Tensor = capture.require_decoder_queries()
        decoder_layers: list[dict[str, Tensor]] = _decoder_layers(
            predictions,
            query_layers,
        )
        queries: Tensor = decoder_layers[-1]["queries"]
        aux: dict[str, Any] = {
            key: value
            for key, value in predictions.items()
            if key not in {"pred_logits", "pred_boxes", "aux_outputs"}
        }
        aux["upstream_outputs"] = predictions
        proposal_count = min(self.num_queries, logits.shape[1])
        return DetectorOutput(
            logits=logits[:, :proposal_count],
            boxes=boxes[:, :proposal_count],
            queries=queries[:, :proposal_count],
            reference_points=boxes[:, :proposal_count],
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
        raw_features: object = backbone(flat_images)
        return [
            feature.unflatten(0, (batch_size, num_frames))
            for feature in _backbone_feature_tensors(raw_features)
        ]

    def freeze(
        self,
        backbone: bool | None = None,
        decoder: bool = False,
        *,
        encoder: bool | None = None,
        bbox_embed: bool = False,
        cls_embed: bool = False,
    ) -> None:
        """Заморозить/разморозить выбранные upstream-блоки детектора."""
        freeze_backbone = bool(backbone)
        freeze_encoder = encoder if encoder is not None else freeze_backbone
        backbone_module = _find_module(self.model, "backbone")
        if backbone_module is not None:
            _set_trainable(backbone_module, not freeze_backbone)
        encoder_module = _find_module(self.model, "transformer.encoder")
        if encoder_module is not None:
            _set_trainable(encoder_module, not freeze_encoder)
        _set_trainable(
            _resolve_module(self.model, ("transformer.decoder", "decoder")), not decoder
        )
        _set_trainable(
            _resolve_module(self.model, ("bbox_embed", "box_embed")), not bbox_embed
        )
        _set_trainable(
            _resolve_module(self.model, ("cls_embed", "class_embed")), not cls_embed
        )

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
        if "resolution" in options:
            raise ValueError(
                "RF-DETR resolution is fixed by the pretrained variant and must not "
                "be set in model options"
            )
        options["resolution"] = self.input_resolution
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


def _child_module_path(owner: nn.Module, path: str) -> nn.Module:
    module: nn.Module = owner
    for name in path.split("."):
        module = _child_module(module, name)
    return module


def _resolve_module(owner: nn.Module, candidates: Sequence[str]) -> nn.Module:
    for path in candidates:
        try:
            return _child_module_path(owner, path)
        except TypeError:
            continue
    tried: str = ", ".join(candidates)
    raise TypeError(
        f"RF-DETR {type(owner).__name__} не содержит ожидаемый блок: {tried}"
    )


def _find_module(owner: nn.Module, path: str) -> nn.Module | None:
    try:
        return _child_module_path(owner, path)
    except TypeError:
        return None


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
    return [
        {
            "queries": layer_queries,
            "logits": _prediction_tensor(layer, "pred_logits"),
            "boxes": _normalized_prediction_boxes(layer),
            "reference_points": _normalized_prediction_boxes(layer),
        }
        for layer, layer_queries in zip(
            layer_predictions,
            selected_queries,
            strict=True,
        )
    ]


def _normalized_prediction_boxes(
    predictions: Mapping[str, Any], *, bbox_reparam: bool = False
) -> Tensor:
    """Return project-contract boxes using RF-DETR's own box semantics.

    RF-DETR with ``bbox_reparam=True`` can emit unbounded normalized ``cxcywh``
    values.  Its official postprocessor converts them to ``xyxy`` and clamps to
    image bounds.  Mirror that operation on the unit square for the local
    ``DetectorOutput`` contract while keeping the untouched upstream mapping in
    ``aux["upstream_outputs"]`` for the native criterion and postprocessor.
    """
    boxes = _prediction_tensor(predictions, "pred_boxes")
    if not boxes.numel() or (boxes.detach().amin() >= 0 and boxes.detach().amax() <= 1):
        return boxes
    if bbox_reparam:
        xyxy = box_ops.box_cxcywh_to_xyxy(boxes).clamp(0.0, 1.0)
        return box_ops.box_xyxy_to_cxcywh(xyxy)
    return boxes.sigmoid()


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)
