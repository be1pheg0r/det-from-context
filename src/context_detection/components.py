"""Загрузка self-contained директорий моделей и датасетов."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

from omegaconf import OmegaConf

from .config import ExperimentConfig


class ComponentKind(StrEnum):
    """Поддерживаемые типы directory components."""

    DATASET = "dataset"
    MODEL = "model"


@dataclass(frozen=True)
class ComponentDirectory:
    """Проверенная папка с provider-кодом, конфигом и артефактами."""

    root: Path
    kind: ComponentKind
    name: str
    provider_path: Path
    config_path: Path
    artifacts_path: Path

    PROVIDER_FILE: ClassVar[str] = "provider.py"
    CONFIG_FILE: ClassVar[str] = "config.yaml"
    ARTIFACTS_DIR: ClassVar[str] = "artifacts"

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        project_root: Path,
        kind: ComponentKind,
        expected_name: str,
    ) -> ComponentDirectory:
        """Проверить layout, импортировать ``PROTOCOL`` и зарегистрировать его."""
        root: Path = Path(path)
        if not root.is_absolute():
            root = project_root / root
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)

        provider_path: Path = root / cls.PROVIDER_FILE
        config_path: Path = root / cls.CONFIG_FILE
        artifacts_path: Path = root / cls.ARTIFACTS_DIR
        for required_file in (provider_path, config_path):
            if not required_file.is_file():
                raise FileNotFoundError(required_file)
        if not artifacts_path.is_dir():
            raise FileNotFoundError(artifacts_path)

        raw: Any = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
            raise ValueError(f"{config_path}: обязателен строковый ключ name")
        name: str = raw["name"]
        if name != expected_name:
            raise ValueError(
                f"имя {kind} в experiment config ({expected_name!r}) "
                f"не совпадает с {config_path} ({name!r})"
            )

        module: ModuleType = cls._load_provider(provider_path, kind)
        try:
            protocol: Any = module.PROTOCOL
        except AttributeError as error:
            raise ValueError(f"{provider_path}: не экспортирован PROTOCOL") from error

        if kind is ComponentKind.DATASET:
            from .data.protocols import register_dataset_protocol

            register_dataset_protocol(name, protocol, replace=True)
        else:
            from .models.protocols import register_model_protocol

            register_model_protocol(name, protocol, replace=True)

        return cls(
            root=root,
            kind=kind,
            name=name,
            provider_path=provider_path,
            config_path=config_path,
            artifacts_path=artifacts_path,
        )

    @staticmethod
    def _load_provider(path: Path, kind: ComponentKind) -> ModuleType:
        digest: str = hashlib.sha256(str(path).encode()).hexdigest()[:12]
        module_name: str = f"context_detection_{kind}_component_{digest}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"не удалось создать import spec для {path}")
        module: ModuleType = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module


def load_component_directories(
    config: ExperimentConfig,
    project_root: Path,
) -> dict[ComponentKind, ComponentDirectory]:
    """Загрузить directory-backed components, указанные экспериментом."""
    components: dict[ComponentKind, ComponentDirectory] = {}
    if config.data.component_path is not None:
        components[ComponentKind.DATASET] = ComponentDirectory.load(
            config.data.component_path,
            project_root=project_root,
            kind=ComponentKind.DATASET,
            expected_name=config.data.name,
        )
    if config.detector.component_path is not None:
        components[ComponentKind.MODEL] = ComponentDirectory.load(
            config.detector.component_path,
            project_root=project_root,
            kind=ComponentKind.MODEL,
            expected_name=config.detector.name,
        )
    return components
