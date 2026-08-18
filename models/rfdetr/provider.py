"""Directory model provider для официальной реализации RF-DETR."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from context_detection.config import ExperimentConfig
from context_detection.models.detector import DetectorAdapter
from context_detection.models.protocols import build_context_module
from context_detection.models.rfdetr import RFDetrAdapter, RFDetrVariant
from context_detection.models.wrapper import ContextDetector


class RFDetrSettings(BaseModel):
    """Воспроизводимая конфигурация RF-DETR component directory."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["rfdetr"]
    variant: RFDetrVariant = RFDetrVariant.SMALL
    model: dict[str, Any] = Field(min_length=1)


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
        if config.detector.freeze_backbone or config.detector.freeze_decoder:
            detector.freeze(
                backbone=config.detector.freeze_backbone,
                decoder=config.detector.freeze_decoder,
            )
        return detector

    def build(self, config: ExperimentConfig) -> nn.Module:
        """Собрать публичный ContextDetector endpoint."""
        return ContextDetector(
            detector=self.build_detector(config),
            context_module=build_context_module(config),
            fusion=config.context.fusion,
            detach_state=config.train.detach_state,
        )

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
        if config.detector.group_detr is not None:
            resolved["group_detr"] = config.detector.group_detr
        if config.detector.num_decoder_registers is not None:
            resolved["num_decoder_registers"] = config.detector.num_decoder_registers
        if config.context.name == "memot":
            if resolved.get("group_detr") != 1:
                raise ValueError(
                    "MeMOT требует RF-DETR group_detr=1: temporal memory "
                    "сопоставляет один стабильный набор queries между кадрами"
                )
            if resolved.get("num_decoder_registers", 0) != 0:
                raise ValueError(
                    "MeMOT пока не поддерживает RF-DETR decoder register tokens"
                )
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


PROTOCOL: RFDetrProtocol = RFDetrProtocol()
