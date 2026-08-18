"""Детекция с переиспользованием временного контекста."""

from typing import Any

from .components import ComponentDirectory, ComponentKind
from .experiment import (
    ExperimentComponents,
    ExperimentProtocol,
    ExperimentRun,
    RunStatus,
)

_LAZY_PROTOCOL_EXPORTS = frozenset(
    {
        "DatasetProtocol",
        "DatasetSplit",
        "ModelProtocol",
        "build_dataloader",
        "build_registered_model",
        "register_dataset_protocol",
        "register_model_protocol",
    }
)

__all__ = [
    "DatasetProtocol",
    "DatasetSplit",
    "ComponentDirectory",
    "ComponentKind",
    "ExperimentComponents",
    "ExperimentProtocol",
    "ExperimentRun",
    "ModelProtocol",
    "RunStatus",
    "build_dataloader",
    "build_registered_model",
    "register_dataset_protocol",
    "register_model_protocol",
]


def __getattr__(name: str) -> Any:
    """Не импортировать PyTorch, пока protocol API действительно не нужен."""
    if name in _LAZY_PROTOCOL_EXPORTS:
        from . import protocols

        return getattr(protocols, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_PROTOCOL_EXPORTS})
