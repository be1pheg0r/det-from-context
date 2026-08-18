"""Модели и публичный model protocol."""

from .protocols import (
    MODEL_PROTOCOLS,
    DetectionModelProtocol,
    ModelProtocol,
    ModelProtocolRegistry,
    build_registered_model,
    register_model_protocol,
)

__all__ = [
    "MODEL_PROTOCOLS",
    "DetectionModelProtocol",
    "ModelProtocol",
    "ModelProtocolRegistry",
    "build_registered_model",
    "register_model_protocol",
]
