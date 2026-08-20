"""Расширяемый model protocol с endpoint в виде ``torch.nn.Module``."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from torch import nn

from ..config import ExperimentConfig
from ..registry import register_detector_name
from .detector import DetectorAdapter, DummyDetector
from .memory import (
    BridgeADMemory,
    ContextCrossAttention,
    ContextModule,
    EMASlot,
    MeMOTMemory,
    NoContext,
    StreamQueue,
)
from .memot import MeMOTMemoryDecoder, MeMOTMemoryEncoder, MeMOTTracker
from .wrapper import ContextDetector


@runtime_checkable
class ModelProtocol(Protocol):
    """Структурный интерфейс провайдера модели.

    Реализация свободна строить любую архитектуру, но endpoint всегда является
    обычным ``nn.Module`` и поэтому совместим с optimizer, DDP, AMP и ClearML.
    """

    def build(self, config: ExperimentConfig) -> nn.Module:
        """Построить модель из полностью скомпонованного конфига."""


@runtime_checkable
class DetectionModelProtocol(ModelProtocol, Protocol):
    """Дополнительный legacy endpoint для detector adapters."""

    def build_detector(self, config: ExperimentConfig) -> DetectorAdapter:
        """Построить адаптер детектора без context wrapper."""


class ModelProtocolRegistry:
    """Реестр model protocols с проверкой конечного интерфейса."""

    def __init__(self) -> None:
        self._protocols: dict[str, ModelProtocol] = {}

    @property
    def names(self) -> frozenset[str]:
        """Зарегистрированные имена моделей."""
        return frozenset(self._protocols)

    def register(
        self,
        name: str,
        protocol: ModelProtocol,
        *,
        replace: bool = False,
    ) -> None:
        """Зарегистрировать структурно совместимый provider."""
        if not name.strip():
            raise ValueError("имя model protocol не может быть пустым")
        if not isinstance(protocol, ModelProtocol):
            raise TypeError("model protocol обязан реализовать build(config)")
        if name in self._protocols and not replace:
            raise ValueError(f"model protocol {name!r} уже зарегистрирован")
        self._protocols[name] = protocol
        register_detector_name(name)

    def get(self, name: str) -> ModelProtocol:
        """Получить provider с информативной ошибкой для неизвестного имени."""
        try:
            return self._protocols[name]
        except KeyError as error:
            raise ValueError(
                f"нет model protocol {name!r}, доступно: {sorted(self.names)}"
            ) from error

    def build(self, config: ExperimentConfig) -> nn.Module:
        """Построить и проверить публичный endpoint."""
        model: nn.Module = self.get(config.detector.name).build(config)
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"model protocol {config.detector.name!r} вернул "
                f"{type(model).__name__}, ожидался nn.Module"
            )
        return model

    def build_detector(self, config: ExperimentConfig) -> DetectorAdapter:
        """Сохранить существующий detector-only API для detection providers."""
        protocol: ModelProtocol = self.get(config.detector.name)
        if not isinstance(protocol, DetectionModelProtocol):
            raise TypeError(
                f"model protocol {config.detector.name!r} не предоставляет "
                "DetectorAdapter"
            )
        return protocol.build_detector(config)


DetectorFactory = Callable[[ExperimentConfig], DetectorAdapter]


class ContextDetectorModelProtocol:
    """Адаптирует detector factory к существующему ``ContextDetector``."""

    def __init__(self, detector_factory: DetectorFactory) -> None:
        self.detector_factory: DetectorFactory = detector_factory

    def build_detector(self, config: ExperimentConfig) -> DetectorAdapter:
        """Построить и применить режимы заморозки к detector adapter."""
        detector: DetectorAdapter = self.detector_factory(config)
        detector_config = config.detector
        freeze_backbone: bool = bool(detector_config.freeze_backbone)
        freeze_decoder: bool = bool(detector_config.freeze_decoder)
        freeze_encoder: bool = bool(detector_config.freeze_encoder)
        freeze_bbox_embed: bool = bool(detector_config.freeze_bbox_embed)
        freeze_cls_embed: bool = bool(detector_config.freeze_cls_embed)
        if (
            freeze_backbone
            or freeze_decoder
            or freeze_encoder
            or freeze_bbox_embed
            or freeze_cls_embed
        ):
            detector.freeze(
                backbone=freeze_backbone,
                decoder=freeze_decoder,
                encoder=freeze_encoder,
                bbox_embed=freeze_bbox_embed,
                cls_embed=freeze_cls_embed,
            )
        return detector

    def build(self, config: ExperimentConfig) -> nn.Module:
        """Собрать текущий detector + context pipeline без изменения forward."""
        detector: DetectorAdapter = self.build_detector(config)
        return build_context_model(detector, config)


_CONTEXT_MODULES: dict[str, type[ContextModule]] = {
    "none": NoContext,
    "memot": MeMOTMemory,
    "ema_slot": EMASlot,
    "cross_attn": ContextCrossAttention,
    "stream_queue": StreamQueue,
    "bridge_ad": BridgeADMemory,
}


def build_context_module(config: ExperimentConfig) -> ContextModule:
    """Построить context module, сохранив прежние ошибки и параметры."""
    name: str = config.context.name
    context_type: type[ContextModule] = _CONTEXT_MODULES[name]
    if inspect.isabstract(context_type):
        raise NotImplementedError(
            f"ветка контекста {name!r} ещё не реализована (Человек 4): "
            f"{context_type.__name__} не переопределяет read/write"
        )
    if context_type is MeMOTMemory:
        return context_type(
            dim=config.detector.dim,
            num_heads=config.detector.num_heads,
            num_slots=config.context.num_slots,
            memory_length=config.context.memory_length,
            short_memory_length=config.context.short_memory_length,
            write_threshold=config.context.write_threshold,
            max_missed=config.context.max_missed,
            association_iou_threshold=config.context.association_iou_threshold,
            association_cosine_threshold=config.context.association_cosine_threshold,
            association_appearance_weight=(
                config.context.association_appearance_weight
            ),
            motion_momentum=config.context.motion_momentum,
        )
    return context_type()


def build_context_model(
    detector: DetectorAdapter, config: ExperimentConfig
) -> nn.Module:
    """Wrap a detector while keeping MeMOT strictly outside its forward path."""
    context_module = build_context_module(config)
    if isinstance(context_module, MeMOTMemory):
        return MeMOTTracker(
            detector=detector,
            memory_encoder=MeMOTMemoryEncoder(context_module),
            memory_decoder=MeMOTMemoryDecoder(
                dim=config.detector.dim,
                num_heads=config.detector.num_heads,
                num_classes=config.detector.num_classes,
                num_slots=config.context.num_slots,
                num_layers=config.context.memory_decoder_layers,
            ),
            detach_state=config.train.detach_state,
        )
    return ContextDetector(
        detector=detector,
        context_module=context_module,
        fusion=config.context.fusion,
        detach_state=config.train.detach_state,
    )


def _build_dummy_detector(config: ExperimentConfig) -> DetectorAdapter:
    detector = config.detector
    return DummyDetector(
        num_queries=detector.num_queries,
        dim=detector.dim,
        num_classes=detector.num_classes,
        num_decoder_layers=detector.num_decoder_layers,
        num_heads=detector.num_heads,
    )


def _build_rfdetr_detector(config: ExperimentConfig) -> DetectorAdapter:
    from .rfdetr import RFDetrAdapter

    variant: str | None = config.detector.variant
    if variant is None:
        raise ValueError("для rfdetr нужен variant")
    return RFDetrAdapter(
        variant=variant,
        weights=config.detector.weights,
    )


MODEL_PROTOCOLS: ModelProtocolRegistry = ModelProtocolRegistry()
MODEL_PROTOCOLS.register(
    "dummy",
    ContextDetectorModelProtocol(_build_dummy_detector),
)
MODEL_PROTOCOLS.register(
    "rfdetr",
    ContextDetectorModelProtocol(_build_rfdetr_detector),
)


def register_model_protocol(
    name: str,
    protocol: ModelProtocol,
    *,
    replace: bool = False,
) -> None:
    """Публичная точка расширения реестра моделей."""
    MODEL_PROTOCOLS.register(name, protocol, replace=replace)


def build_registered_model(config: ExperimentConfig) -> nn.Module:
    """Построить nn.Module через зарегистрированный protocol."""
    return MODEL_PROTOCOLS.build(_prepare_directory_model_config(config))


def build_registered_detector(config: ExperimentConfig) -> DetectorAdapter:
    """Построить DetectorAdapter через совместимый встроенный protocol."""
    return MODEL_PROTOCOLS.build_detector(_prepare_directory_model_config(config))


def _prepare_directory_model_config(config: ExperimentConfig) -> ExperimentConfig:
    if config.detector.component_path is None:
        return config
    from ..components import ComponentDirectory, ComponentKind

    component: ComponentDirectory = ComponentDirectory.load(
        config.detector.component_path,
        project_root=Path.cwd(),
        kind=ComponentKind.MODEL,
        expected_name=config.detector.name,
    )
    runtime_config: ExperimentConfig = config.model_copy(deep=True)
    runtime_config.detector.component_path = str(component.root)
    runtime_config.detector.config_path = str(component.config_path)
    return runtime_config
