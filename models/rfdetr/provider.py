"""Directory model provider для официальной реализации RF-DETR."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from context_detection.config import ExperimentConfig
from context_detection.models.detector import DetectorAdapter
from context_detection.models.protocols import build_context_model
from context_detection.models.rfdetr import RFDetrAdapter, RFDetrVariant


class RFDetrSettings(BaseModel):
    """Воспроизводимая конфигурация RF-DETR component directory."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["rfdetr"]
    variant: RFDetrVariant = RFDetrVariant.SMALL
    model: dict[str, Any] = Field(min_length=1)
    freeze: RFDetrFreezeSettings = Field(default_factory=lambda: RFDetrFreezeSettings())


class RFDetrFreezeSettings(BaseModel):
    """Дефолтные режимы заморозки блоков RF-DETR для component directory."""

    model_config = ConfigDict(extra="forbid")

    encoder: bool = False
    decoder: bool = False
    bbox_embed: bool = False
    cls_embed: bool = False


class RFDetrProtocol:
    """Собрать detector-only или detector+context endpoint из component config."""

    def build_detector(self, config: ExperimentConfig) -> DetectorAdapter:
        """Создать upstream adapter и применить режим заморозки эксперимента."""
        settings: RFDetrSettings = self._load_settings(config)
        model_options: dict[str, Any] = self._resolve_model_options(
            config,
            settings.model,
        )
        detector = RFDetrAdapter(
            settings.variant,
            model_options=model_options,
        )
        if detector.dim != config.detector.dim:
            raise ValueError(
                f"RF-DETR {settings.variant.value} использует dim={detector.dim}, "
                f"но experiment config задаёт detector.dim={config.detector.dim}"
            )
        freeze_encoder: bool = self._resolve_freeze_option(
            config.detector.freeze_encoder,
            config.detector.freeze_backbone,
            settings.freeze.encoder,
        )
        freeze_decoder: bool = self._resolve_freeze_option(
            config.detector.freeze_decoder,
            None,
            settings.freeze.decoder,
        )
        freeze_bbox_embed: bool = self._resolve_freeze_option(
            config.detector.freeze_bbox_embed,
            None,
            settings.freeze.bbox_embed,
        )
        freeze_cls_embed: bool = self._resolve_freeze_option(
            config.detector.freeze_cls_embed,
            None,
            settings.freeze.cls_embed,
        )
        if freeze_encoder or freeze_decoder or freeze_bbox_embed or freeze_cls_embed:
            detector.freeze(
                backbone=freeze_encoder,
                encoder=freeze_encoder,
                decoder=freeze_decoder,
                bbox_embed=freeze_bbox_embed,
                cls_embed=freeze_cls_embed,
            )
        return detector

    def build(self, config: ExperimentConfig) -> nn.Module:
        """Собрать публичный ContextDetector endpoint."""
        return build_context_model(self.build_detector(config), config)

    @staticmethod
    def _load_settings(config: ExperimentConfig) -> RFDetrSettings:
        config_path: str | None = config.detector.config_path
        if config_path is None:
            raise ValueError("rfdetr component требует detector.config_path")
        raw: Any = OmegaConf.to_container(
            OmegaConf.load(Path(config_path)),
            resolve=True,
        )
        return RFDetrSettings.model_validate(raw)

    @staticmethod
    def _resolve_model_options(
        config: ExperimentConfig,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        resolved: dict[str, Any] = dict(options)
        # The dataset component is the source of truth for the class mapping.
        # ExperimentProtocol synchronizes this value before building the model.
        resolved["num_classes"] = config.detector.num_classes
        weights: object = resolved.get("pretrain_weights")
        if weights is None:
            return resolved
        if not isinstance(weights, str):
            raise TypeError("model.pretrain_weights должен быть строкой или null")
        config_path: str | None = config.detector.config_path
        if config_path is None:
            raise ValueError("rfdetr component требует detector.config_path")
        weights_path: Path = Path(weights)
        if not weights_path.is_absolute() and weights_path.parent == Path():
            return resolved
        if not weights_path.is_absolute():
            weights_path = Path(config_path).parent / weights_path
        weights_path = weights_path.resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(weights_path)
        resolved["pretrain_weights"] = str(weights_path)
        return resolved

    @staticmethod
    def _resolve_freeze_option(
        explicit: bool | None,
        legacy: bool | None,
        component_default: bool,
    ) -> bool:
        if explicit is not None:
            return explicit
        if legacy is not None:
            return legacy
        return component_default


PROTOCOL: RFDetrProtocol = RFDetrProtocol()
