"""Публичный API протоколов моделей и датасетов."""

from .data.protocols import (
    DATASET_PROTOCOLS,
    DatasetProtocol,
    DatasetProtocolRegistry,
    DatasetSplit,
    build_dataloader,
    register_dataset_protocol,
)
from .models.protocols import (
    MODEL_PROTOCOLS,
    ModelProtocol,
    ModelProtocolRegistry,
    build_registered_model,
    register_model_protocol,
)

__all__ = [
    "DATASET_PROTOCOLS",
    "MODEL_PROTOCOLS",
    "DatasetProtocol",
    "DatasetProtocolRegistry",
    "DatasetSplit",
    "ModelProtocol",
    "ModelProtocolRegistry",
    "build_dataloader",
    "build_registered_model",
    "register_dataset_protocol",
    "register_model_protocol",
]
