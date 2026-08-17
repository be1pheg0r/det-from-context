"""Проверки единого lifecycle экспериментов."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import OmegaConf

from context_detection.config import ExperimentConfig
from context_detection.experiment import (
    ClearMLTracker,
    ExperimentProtocol,
    ExperimentRun,
)


def _write_config(root: Path, *, clearml: bool = False) -> Path:
    dataset_path: Path = root / "dataset.yaml"
    OmegaConf.save({"name": "dummy", "root": None}, dataset_path)
    config_path: Path = root / "experiment.yaml"
    OmegaConf.save(
        {
            "defaults": ["_self_"],
            "name": "protocol-test",
            "data": {"config_path": "dataset.yaml"},
            "output": {"root": "results"},
            "clearml": {"enabled": clearml},
        },
        config_path,
    )
    return config_path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value: Any = json.load(stream)
    assert isinstance(value, dict)
    return value


def test_config_covers_entire_experiment_protocol() -> None:
    config: ExperimentConfig = ExperimentConfig()
    assert config.data.config_path
    assert config.train.optimizer
    assert config.validation.metrics
    assert config.logging.every_n_steps > 0
    assert config.output.checkpoint_every_n_epochs > 0
    assert config.clearml.project_name


def test_execute_creates_uniform_result_directory(tmp_path: Path) -> None:
    config_path: Path = _write_config(tmp_path)
    source_path: Path = tmp_path / "worker.py"
    source_path.write_text("# worker\n", encoding="utf-8")

    def worker(
        experiment: ExperimentRun,
        config: ExperimentConfig,
    ) -> dict[str, Any]:
        experiment.log_metrics({"loss": 0.25}, step=1, split="train")
        return {"seed": config.train.seed}

    protocol: ExperimentProtocol = ExperimentProtocol(tmp_path)
    result: Path = protocol.execute(
        config_path,
        worker,
        source_paths=[source_path],
    )

    assert (result / "artifacts").is_dir()
    assert (result / "checkpoints").is_dir()
    assert (result / "logs" / "experiment.log").is_file()
    assert (result / "sources" / "experiment.yaml").is_file()
    assert (result / "sources" / "dataset.yaml").is_file()
    assert (result / "sources" / "worker.py").is_file()
    assert _read_json(result / "metadata.json")["status"] == "completed"
    assert _read_json(result / "summary.json")["seed"] == 42
    assert '"loss": 0.25' in (result / "metrics.jsonl").read_text(encoding="utf-8")


def test_failed_worker_is_recorded_and_reraised(tmp_path: Path) -> None:
    config_path: Path = _write_config(tmp_path)

    def worker(
        experiment: ExperimentRun,
        config: ExperimentConfig,
    ) -> None:
        raise LookupError("broken worker")

    protocol: ExperimentProtocol = ExperimentProtocol(tmp_path)
    with pytest.raises(LookupError, match="broken worker"):
        protocol.execute(config_path, worker)

    result_roots: list[Path] = list((tmp_path / "results" / "protocol-test").iterdir())
    assert len(result_roots) == 1
    metadata: dict[str, Any] = _read_json(result_roots[0] / "metadata.json")
    assert metadata["status"] == "failed"
    assert "broken worker" in metadata["error"]


def test_dataset_config_reference_must_exist(tmp_path: Path) -> None:
    config_path: Path = _write_config(tmp_path)
    (tmp_path / "dataset.yaml").unlink()

    with pytest.raises(FileNotFoundError, match="dataset.yaml"):
        ExperimentProtocol(tmp_path).start(config_path)


def test_monitor_must_be_a_validation_metric() -> None:
    with pytest.raises(ValueError, match="validation.metrics"):
        ExperimentConfig.model_validate(
            {
                "validation": {"metrics": ["map"]},
                "output": {"monitor": "loss"},
            }
        )


def test_clearml_tracker_closes_without_terminating_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []

    class FakeTask:
        id = "clearml-task-123"

        @classmethod
        def init(cls, **kwargs: Any) -> FakeTask:
            calls.append(("init", kwargs))
            return cls()

        def connect(self, value: Any, name: str) -> None:
            calls.append(("connect", name))

        def get_logger(self) -> SimpleNamespace:
            return SimpleNamespace(report_scalar=lambda **kwargs: None)

        def get_output_log_web_page(self) -> str:
            return "https://clearml.example/projects/demo/experiments/clearml-task-123"

        def upload_artifact(self, **kwargs: Any) -> None:
            calls.append(("upload", kwargs))

        def close(self) -> None:
            calls.append(("close", None))

        def mark_failed(self, **kwargs: Any) -> None:
            calls.append(("failed", kwargs))

        def flush(self, **kwargs: Any) -> None:
            calls.append(("flush", kwargs))

    monkeypatch.setenv("CLEARML_API_ACCESS_KEY", "access")
    monkeypatch.setenv("CLEARML_API_SECRET_KEY", "secret")
    monkeypatch.setitem(sys.modules, "clearml", SimpleNamespace(Task=FakeTask))

    tracker: ClearMLTracker = ClearMLTracker(ExperimentConfig())
    assert tracker.describe() == {
        "backend": "clearml",
        "task_id": "clearml-task-123",
        "task_url": (
            "https://clearml.example/projects/demo/experiments/clearml-task-123"
        ),
    }
    tracker.complete()

    assert ("close", None) in calls
    assert all(call[0] != "mark_completed" for call in calls)
