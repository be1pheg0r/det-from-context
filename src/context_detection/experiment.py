"""Единый lifecycle воспроизводимых экспериментов.

Модуль не содержит кода обучения: конкретный эксперимент передаёт worker,
а протокол отвечает за конфигурацию, структуру результатов и трекинг.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar

from omegaconf import OmegaConf

from .config import ExperimentConfig, load_config


class RunStatus(StrEnum):
    """Состояния локального запуска эксперимента."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ResultEntry(StrEnum):
    """Имена обязательных элементов директории результата."""

    ARTIFACTS = "artifacts"
    CHECKPOINTS = "checkpoints"
    CONFIG = "config.yaml"
    LOGS = "logs"
    METADATA = "metadata.json"
    METRICS = "metrics.jsonl"
    SOURCES = "sources"
    SUMMARY = "summary.json"


class ClearMLEnv(StrEnum):
    """Переменные окружения, необходимые ClearML SDK."""

    ACCESS_KEY = "CLEARML_API_ACCESS_KEY"
    SECRET_KEY = "CLEARML_API_SECRET_KEY"


class ExperimentTracker(ABC):
    """Backend отслеживания, не связанный с кодом конкретного эксперимента."""

    @abstractmethod
    def log_metrics(self, metrics: Mapping[str, float], step: int, split: str) -> None:
        """Записать набор скалярных метрик."""

    @abstractmethod
    def upload_artifact(self, name: str, path: Path) -> None:
        """Загрузить артефакт запуска."""

    @abstractmethod
    def complete(self) -> None:
        """Отметить запуск успешно завершённым."""

    @abstractmethod
    def fail(self, reason: str) -> None:
        """Отметить запуск завершённым с ошибкой."""


class LocalTracker(ExperimentTracker):
    """Локальный backend; постоянные данные пишет :class:`ExperimentRun`."""

    def log_metrics(self, metrics: Mapping[str, float], step: int, split: str) -> None:
        """Не дублировать локальную запись метрик."""

    def upload_artifact(self, name: str, path: Path) -> None:
        """Не дублировать локальный артефакт."""

    def complete(self) -> None:
        """Локальный статус обновляет сам запуск."""

    def fail(self, reason: str) -> None:
        """Локальный статус обновляет сам запуск."""


class ClearMLTracker(ExperimentTracker):
    """Адаптер ClearML с отложенным импортом SDK."""

    def __init__(self, config: ExperimentConfig) -> None:
        missing: list[str] = [key for key in ClearMLEnv if not os.getenv(key)]
        if missing:
            names: str = ", ".join(missing)
            raise RuntimeError(f"в .env не заданы обязательные переменные: {names}")

        try:
            from clearml import Task
        except ImportError as error:
            raise RuntimeError(
                "для включённого ClearML установите зависимости проекта"
            ) from error

        self._task: Any = Task.init(
            project_name=config.clearml.project_name,
            task_name=config.name,
            tags=config.clearml.tags,
            reuse_last_task_id=False,
        )
        self._task.connect(
            config.model_dump(mode="json"),
            name="resolved_config",
        )
        self._logger: Any = self._task.get_logger()

    def log_metrics(self, metrics: Mapping[str, float], step: int, split: str) -> None:
        """Передать скаляры в ClearML."""
        for name, value in metrics.items():
            self._logger.report_scalar(
                title=split,
                series=name,
                value=value,
                iteration=step,
            )

    def upload_artifact(self, name: str, path: Path) -> None:
        """Передать локальный артефакт в ClearML."""
        self._task.upload_artifact(
            name=name,
            artifact_object=str(path),
            wait_on_upload=True,
        )

    def complete(self) -> None:
        """Закрыть ClearML task как успешный."""
        self._task.close()

    def fail(self, reason: str) -> None:
        """Закрыть ClearML task как неуспешный."""
        self._task.mark_failed(status_reason=reason)
        self._task.flush(wait_for_uploads=True)


class ExperimentRun:
    """Один активный запуск и его единая директория результатов."""

    _LOG_FORMAT: ClassVar[str] = "%(asctime)s %(levelname)s %(message)s"

    def __init__(
        self,
        config: ExperimentConfig,
        root: Path,
        tracker: ExperimentTracker,
        metadata: Mapping[str, Any],
    ) -> None:
        self.config: ExperimentConfig = config
        self.root: Path = root
        self.tracker: ExperimentTracker = tracker
        self.status: RunStatus = RunStatus.RUNNING
        self._metadata: dict[str, Any] = dict(metadata)
        self._logger: logging.Logger = self._make_logger()
        self._write_metadata()

    @property
    def checkpoints_dir(self) -> Path:
        """Директория контрольных точек."""
        return self.root / ResultEntry.CHECKPOINTS

    @property
    def artifacts_dir(self) -> Path:
        """Директория произвольных артефактов."""
        return self.root / ResultEntry.ARTIFACTS

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: int,
        split: str = "train",
    ) -> None:
        """Добавить метрики в JSONL и настроенный backend.

        Args:
            metrics: Имена метрик и конечные числовые значения.
            step: Номер шага, эпохи или итерации.
            split: Логическая группа метрик, например ``train`` или ``val``.
        """
        if self.status is not RunStatus.RUNNING:
            raise RuntimeError("метрики можно писать только в активный запуск")
        if step < 0:
            raise ValueError("номер шага не может быть отрицательным")
        normalized: dict[str, float] = {
            name: float(value) for name, value in metrics.items()
        }
        if not all(
            value == value and abs(value) != float("inf")
            for value in normalized.values()
        ):
            raise ValueError("метрики должны быть конечными числами")

        record: dict[str, Any] = {
            "step": step,
            "split": split,
            "metrics": normalized,
            "timestamp": _utc_now(),
        }
        metrics_path: Path = self.root / ResultEntry.METRICS
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.tracker.log_metrics(normalized, step, split)
        self._logger.info("metrics split=%s step=%d values=%s", split, step, normalized)

    def save_artifact(self, name: str, source: str | Path) -> Path:
        """Скопировать файл в запуск и передать его backend-у.

        Args:
            name: Безопасное имя артефакта без компонентов пути.
            source: Существующий файл.

        Returns:
            Путь к локальной копии артефакта.
        """
        if not name or Path(name).name != name:
            raise ValueError("имя артефакта не должно содержать путь")
        source_path: Path = Path(source).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        destination: Path = self.artifacts_dir / name
        shutil.copy2(source_path, destination)
        self.tracker.upload_artifact(name, destination)
        return destination

    def complete(self, summary: Mapping[str, Any] | None = None) -> None:
        """Зафиксировать итог и успешно закрыть запуск."""
        if self.status is not RunStatus.RUNNING:
            return
        summary_path: Path = self.root / ResultEntry.SUMMARY
        _write_json(summary_path, dict(summary or {}))
        self.tracker.upload_artifact("summary", summary_path)
        self.tracker.complete()
        self.status = RunStatus.COMPLETED
        self._metadata["finished_at"] = _utc_now()
        self._write_metadata()
        self._logger.info("experiment completed")
        self._close_logger()

    def fail(self, error: BaseException) -> None:
        """Зафиксировать ошибку и закрыть запуск."""
        if self.status is not RunStatus.RUNNING:
            return
        reason: str = f"{type(error).__name__}: {error}"
        self.tracker.fail(reason)
        self.status = RunStatus.FAILED
        self._metadata["finished_at"] = _utc_now()
        self._metadata["error"] = reason
        self._write_metadata()
        self._logger.error(
            "experiment failed",
            exc_info=(type(error), error, error.__traceback__),
        )
        self._close_logger()

    def __enter__(self) -> ExperimentRun:
        """Вернуть активный запуск."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Всегда завершить tracker и не подавлять исключения worker-а."""
        if exc_value is None:
            self.complete()
        else:
            self.fail(exc_value)
        return False

    def _make_logger(self) -> logging.Logger:
        logger: logging.Logger = logging.getLogger(f"experiment.{self.root.name}")
        logger.setLevel(self.config.logging.level.value)
        logger.propagate = False
        handler: logging.FileHandler = logging.FileHandler(
            self.root / ResultEntry.LOGS / "experiment.log",
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(self._LOG_FORMAT))
        logger.addHandler(handler)
        return logger

    def _close_logger(self) -> None:
        for handler in tuple(self._logger.handlers):
            handler.close()
            self._logger.removeHandler(handler)

    def _write_metadata(self) -> None:
        self._metadata["status"] = self.status
        _write_json(self.root / ResultEntry.METADATA, self._metadata)


ExperimentWorker = Callable[[ExperimentRun, ExperimentConfig], Mapping[str, Any] | None]


class ExperimentProtocol:
    """Создаёт и исполняет эксперименты по единому протоколу."""

    _ENV_FILE: ClassVar[str] = ".env"
    _SLUG_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"[^a-zA-Z0-9_.-]+")

    def __init__(self, project_root: str | Path | None = None) -> None:
        self.project_root: Path = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[2]
        )

    def start(
        self,
        config_path: str | Path,
        overrides: Sequence[str] | None = None,
        source_paths: Sequence[str | Path] | None = None,
        launch_script: str | Path | None = None,
    ) -> ExperimentRun:
        """Подготовить один изолированный запуск.

        Args:
            config_path: Корневой Hydra YAML эксперимента.
            overrides: Hydra overrides для конкретного запуска.
            source_paths: Дополнительные исходники для снимка.
            launch_script: Скрипт запуска, который также войдёт в снимок.

        Returns:
            Активный запуск, пригодный как context manager.
        """
        self._load_environment()
        resolved_config_path: Path = self._resolve_path(config_path)
        override_values: list[str] = list(overrides or ())
        config: ExperimentConfig = load_config(
            resolved_config_path,
            override_values,
        )
        dataset_config_path: Path = self._resolve_path(config.data.config_path)
        if not dataset_config_path.is_file():
            raise FileNotFoundError(dataset_config_path)
        root: Path = self._create_result_root(config)
        self._create_layout(root)
        OmegaConf.save(
            OmegaConf.create(config.model_dump(mode="json")),
            root / ResultEntry.CONFIG,
        )
        self._snapshot_sources(
            root,
            resolved_config_path,
            [dataset_config_path, *(source_paths or ())],
            launch_script,
        )
        tracker: ExperimentTracker = self._make_tracker(config)
        metadata: dict[str, Any] = {
            "name": config.name,
            "run_id": root.name,
            "started_at": _utc_now(),
            "config_source": str(resolved_config_path),
            "dataset_config": str(dataset_config_path),
            "overrides": override_values,
            "command": sys.argv,
        }
        return ExperimentRun(config, root, tracker, metadata)

    def execute(
        self,
        config_path: str | Path,
        worker: ExperimentWorker,
        overrides: Sequence[str] | None = None,
        source_paths: Sequence[str | Path] | None = None,
        launch_script: str | Path | None = None,
    ) -> Path:
        """Выполнить worker и вернуть директорию результата."""
        experiment: ExperimentRun = self.start(
            config_path=config_path,
            overrides=overrides,
            source_paths=source_paths,
            launch_script=launch_script,
        )
        with experiment:
            summary: Mapping[str, Any] | None = worker(
                experiment,
                experiment.config,
            )
            if summary is not None:
                experiment.complete(summary)
        return experiment.root

    def _load_environment(self) -> None:
        env_path: Path = self.project_root / self._ENV_FILE
        try:
            from dotenv import load_dotenv
        except ImportError:
            return
        load_dotenv(env_path, override=False)

    def _resolve_path(self, path: str | Path) -> Path:
        candidate: Path = Path(path)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        return candidate.resolve()

    def _create_result_root(self, config: ExperimentConfig) -> Path:
        output_root: Path = self._resolve_path(config.output.root)
        experiment_name: str = self._SLUG_PATTERN.sub("-", config.name).strip("-")
        if not experiment_name:
            raise ValueError("имя эксперимента не содержит допустимых символов")
        timestamp: str = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        result_root: Path = output_root / experiment_name / timestamp
        result_root.mkdir(parents=True, exist_ok=False)
        return result_root

    @staticmethod
    def _create_layout(root: Path) -> None:
        for entry in (
            ResultEntry.ARTIFACTS,
            ResultEntry.CHECKPOINTS,
            ResultEntry.LOGS,
            ResultEntry.SOURCES,
        ):
            (root / entry).mkdir()

    def _snapshot_sources(
        self,
        root: Path,
        config_path: Path,
        source_paths: Sequence[str | Path],
        launch_script: str | Path | None,
    ) -> None:
        sources: list[Path] = [config_path]
        sources.extend(self._resolve_path(path) for path in source_paths)
        if launch_script is not None:
            sources.append(self._resolve_path(launch_script))

        target_dir: Path = root / ResultEntry.SOURCES
        seen_targets: set[Path] = set()
        for source in sources:
            if source.name == self._ENV_FILE:
                raise ValueError(".env с секретами запрещено добавлять в снимок")
            if not source.is_file():
                raise FileNotFoundError(source)
            try:
                relative_source: Path = source.relative_to(self.project_root)
            except ValueError:
                relative_source = Path(source.name)
            destination: Path = target_dir / relative_source
            if destination in seen_targets:
                continue
            seen_targets.add(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    @staticmethod
    def _make_tracker(config: ExperimentConfig) -> ExperimentTracker:
        if config.clearml.enabled:
            return ClearMLTracker(config)
        return LocalTracker()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary: Path = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
        stream.write("\n")
    temporary.replace(path)
