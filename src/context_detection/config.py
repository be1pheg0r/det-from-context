"""Конфиги на Pydantic. Человек 1.

Валидация имён веток — через Literal из registry, поэтому опечатка в конфиге
падает при загрузке, а не через полчаса обучения. Модуль не импортирует torch:
конфиг можно проверить без окружения для обучения.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .registry import (
    DATASETS,
    DATASETS_REQUIRING_ROOT,
    DETECTORS,
    NEEDS_CONTEXT_FRAMES,
    ContextName,
    ContextStrategy,
    FusionMode,
)


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OptimizerName(StrEnum):
    """Поддерживаемые оптимизаторы эксперимента."""

    ADAMW = "adamw"
    SGD = "sgd"


class LogLevel(StrEnum):
    """Поддерживаемые уровни журналирования."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class DataConfig(_Section):
    name: str = "dummy"
    component_path: str | None = None
    config_path: str = Field("dataset_configs/dummy.yaml", min_length=1)
    root: str | None = None
    context_k: int = Field(4, ge=0)
    context_strategy: ContextStrategy = "prev_k"
    clip_len: int = Field(4, ge=1)
    image_size: int = Field(224, gt=0)
    augmentations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> DataConfig:
        if self.name not in DATASETS and self.component_path is None:
            raise ValueError(
                f"неизвестный датасет {self.name!r}, доступно: {sorted(DATASETS)}"
            )
        if self.name in DATASETS_REQUIRING_ROOT and not self.root:
            raise ValueError(f"датасету {self.name!r} нужен root")
        if self.context_strategy == "empty" and self.context_k:
            raise ValueError(
                "strategy='empty' с context_k > 0: слоты будут всегда невалидны. "
                "Для baseline без контекста ставь context_k=0"
            )
        return self


class DetectorConfig(_Section):
    name: str = "dummy"
    component_path: str | None = None
    config_path: str | None = None
    variant: str | None = None
    weights: str | None = None
    freeze_backbone: bool = False
    freeze_decoder: bool = False
    num_queries: int = Field(100, gt=0)
    dim: int = Field(64, gt=0)
    num_classes: int = Field(31, gt=0)
    num_heads: int = Field(4, gt=0)
    num_decoder_layers: int = Field(2, gt=0)

    @model_validator(mode="after")
    def _check(self) -> DetectorConfig:
        if self.name not in DETECTORS and self.component_path is None:
            raise ValueError(
                f"неизвестная модель {self.name!r}, доступно: {sorted(DETECTORS)}"
            )
        if self.dim % self.num_heads:
            raise ValueError(f"dim={self.dim} не делится на num_heads={self.num_heads}")
        if self.name == "rfdetr" and self.component_path is None and not self.variant:
            raise ValueError("для rfdetr нужен variant")
        return self


class ContextConfig(_Section):
    name: ContextName = "none"
    fusion: FusionMode = "residual"
    num_slots: int = Field(64, ge=0)
    memory_length: int = Field(24, gt=0)
    short_memory_length: int = Field(3, gt=0)
    write_threshold: float = Field(0.5, ge=0.0, le=1.0)
    max_missed: int = Field(24, ge=0)
    association_iou_threshold: float = Field(0.1, ge=0.0, le=1.0)
    association_cosine_threshold: float = Field(0.5, ge=-1.0, le=1.0)
    association_appearance_weight: float = Field(0.25, ge=0.0)
    motion_momentum: float = Field(0.8, ge=0.0, lt=1.0)
    write_gate: bool = False
    motion: str | None = None
    horizon: int = Field(0, ge=0)

    @model_validator(mode="after")
    def _check(self) -> ContextConfig:
        if self.name == "none":
            return self
        if not self.num_slots and self.name != "cross_attn":
            raise ValueError(f"ветке {self.name!r} нужен num_slots > 0")
        if self.name == "memot" and self.short_memory_length > self.memory_length:
            raise ValueError("memot требует short_memory_length <= memory_length")
        if self.name == "bridge_ad" and not self.horizon:
            raise ValueError("bridge_ad адресуется по горизонту — нужен horizon > 0")
        return self


class TrainConfig(_Section):
    epochs: int = Field(12, gt=0)
    lr: float = Field(1e-4, gt=0)
    batch_size: int = Field(2, gt=0)
    grad_accum: int = Field(1, ge=1)
    optimizer: OptimizerName = OptimizerName.ADAMW
    weight_decay: float = Field(1e-4, ge=0.0)
    num_workers: int = Field(4, ge=0)
    seed: int = Field(42, ge=0)
    amp: bool = True
    denoising: bool = True
    #: False = полный BPTT по клипу. Не запрещаем — иногда нужно для проверки
    #: гипотезы, — но по умолчанию это прямой путь в OOM.
    detach_state: bool = True


class ValidationConfig(_Section):
    """Параметры периодической оценки модели."""

    metrics: list[str] = Field(
        default_factory=lambda: ["map", "map_50"],
        min_length=1,
    )
    every_n_epochs: int = Field(1, gt=0)
    batch_size: int = Field(2, gt=0)


class LoggingConfig(_Section):
    """Параметры локального журналирования."""

    level: LogLevel = LogLevel.INFO
    every_n_steps: int = Field(20, gt=0)


class OutputConfig(_Section):
    """Единая схема сохранения результатов эксперимента."""

    root: str = Field("outputs", min_length=1)
    checkpoint_every_n_epochs: int = Field(1, gt=0)
    keep_last_checkpoints: int = Field(3, gt=0)
    save_best: bool = True
    monitor: str = Field("map", min_length=1)


class ClearMLConfig(_Section):
    """Настройки ClearML; секреты читаются только из ``.env``."""

    enabled: bool = False
    project_name: str = Field("context-detection", min_length=1)
    tags: list[str] = Field(default_factory=list)


class ExperimentConfig(_Section):
    name: str = Field("unnamed", min_length=1)
    data: DataConfig = Field(default_factory=DataConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    clearml: ClearMLConfig = Field(default_factory=ClearMLConfig)

    @model_validator(mode="after")
    def _check(self) -> ExperimentConfig:
        if self.output.monitor not in self.validation.metrics:
            raise ValueError(
                f"output.monitor={self.output.monitor!r} отсутствует "
                "в validation.metrics"
            )
        name = self.context.name
        if name in NEEDS_CONTEXT_FRAMES and not self.data.context_k:
            raise ValueError(f"{name} читает пиксели контекстных кадров, а context_k=0")
        # clip_len нужен только тем, кто переносит состояние с кадра на кадр.
        # Feature-level ветки читают K кадров из ContextBatch и на одиночном
        # кадре осмысленны — запрещать им clip_len=1 значит блокировать
        # легитимный ablation-контроль сообщением, которое к ним не относится.
        recurrent = name != "none" and name not in NEEDS_CONTEXT_FRAMES
        if recurrent and self.data.clip_len < 2:
            raise ValueError(
                f"ветка {name!r} переносит состояние между кадрами, "
                "а clip_len=1: память никогда не будет прочитана"
            )
        return self


class ConfigFileExtension(StrEnum):
    """Поддерживаемые расширения конфигурационных файлов."""

    YAML = ".yaml"


class HydraConfigLoader:
    """Компонует YAML-конфигурацию Hydra и проверяет доменную схему."""

    _JOB_NAME: ClassVar[str] = "context_detection_config"

    def load(
        self, path: str | Path, overrides: list[str] | None = None
    ) -> ExperimentConfig:
        """Загрузить и провалидировать конфигурацию эксперимента.

        Args:
            path: Путь к корневому YAML-конфигу Hydra.
            overrides: Выражения Hydra Override Grammar.

        Returns:
            Скомпонованная и провалидированная конфигурация эксперимента.

        Raises:
            FileNotFoundError: Файл конфигурации не существует.
            ValueError: Передан конфиг в неподдерживаемом формате.
            TypeError: Результат композиции не является отображением.
            ConfigCompositionException: Ошибка композиции Hydra (например,
                неизвестный override).
        """
        config_path: Path = Path(path)
        self._validate_path(config_path)
        config_dir: Path = config_path.parent.resolve()
        override_values: list[str] = list(overrides or ())

        with initialize_config_dir(
            version_base="1.3",
            config_dir=str(config_dir),
            job_name=self._JOB_NAME,
        ):
            composed: DictConfig = compose(
                config_name=config_path.stem,
                overrides=override_values,
            )

        raw: Any = OmegaConf.to_container(
            composed,
            resolve=True,
            throw_on_missing=True,
        )
        if not isinstance(raw, dict):
            raise TypeError("корневой узел Hydra-конфига должен быть отображением")
        return ExperimentConfig.model_validate(raw)

    @staticmethod
    def _validate_path(path: Path) -> None:
        if path.suffix != ConfigFileExtension.YAML:
            raise ValueError(
                f"ожидался Hydra YAML-конфиг с расширением "
                f"{ConfigFileExtension.YAML}, получено {path}"
            )
        if not path.is_file():
            raise FileNotFoundError(path)


_CONFIG_LOADER: HydraConfigLoader = HydraConfigLoader()


def load_config(
    path: str | Path, overrides: list[str] | None = None
) -> ExperimentConfig:
    """Загрузить Hydra-конфиг с defaults-композицией и override'ами.

    Args:
        path: Путь к корневому ``.yaml``-конфигу.
        overrides: Выражения Hydra Override Grammar вида ``a.b=value``.

    Returns:
        Строго провалидированная конфигурация эксперимента.
    """
    return _CONFIG_LOADER.load(path, overrides)
