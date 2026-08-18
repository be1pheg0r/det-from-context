"""Контрактные тесты dataset/model protocols и их experiment-интеграции."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from context_detection.build import build_dataset, build_model
from context_detection.components import ComponentDirectory, ComponentKind
from context_detection.config import ExperimentConfig
from context_detection.contracts import ContextBatch, DetectionBatch
from context_detection.data import DetectionCollator
from context_detection.data.protocols import (
    DatasetSplit,
    register_dataset_protocol,
)
from context_detection.experiment import (
    ExperimentComponents,
    ExperimentProtocol,
    ExperimentRun,
)
from context_detection.models.protocols import register_model_protocol


def _dummy_config(**overrides: Any) -> ExperimentConfig:
    payload: dict[str, Any] = {
        "data": {
            "name": "dummy",
            "context_k": 1,
            "context_strategy": "prev_k",
            "clip_len": 2,
            "image_size": 16,
        },
        "detector": {
            "name": "dummy",
            "dim": 8,
            "num_heads": 2,
            "num_queries": 4,
            "num_classes": 3,
        },
        "train": {"batch_size": 2, "num_workers": 0},
        "validation": {"batch_size": 2},
    }
    payload.update(overrides)
    return ExperimentConfig.model_validate(payload)


def test_builtin_protocols_reach_torch_endpoints_and_forward() -> None:
    config: ExperimentConfig = _dummy_config()

    loader: DataLoader[Any] = build_dataset(config, "train")
    model: nn.Module = build_model(config)
    batch, context = next(iter(loader))

    assert isinstance(loader, DataLoader)
    assert isinstance(model, nn.Module)
    assert isinstance(batch, DetectionBatch)
    assert isinstance(context, ContextBatch)
    output, state = model(batch, context)
    assert output.boxes.shape == (2, 4, 4)
    assert state is None


class _TensorDatasetProtocol:
    def build(
        self,
        config: ExperimentConfig,
        split: DatasetSplit,
    ) -> DataLoader[Any]:
        del split
        values: torch.Tensor = torch.arange(6, dtype=torch.float32).unsqueeze(1)
        return DataLoader(TensorDataset(values), batch_size=config.train.batch_size)


class _LinearModelProtocol:
    def build(self, config: ExperimentConfig) -> nn.Module:
        return nn.Linear(1, config.detector.num_classes)


def test_structural_protocols_are_extensible_without_inheritance_or_data_root() -> None:
    dataset_name = "test_tensor_dataset"
    model_name = "test_linear_model"
    register_dataset_protocol(dataset_name, _TensorDatasetProtocol())
    register_model_protocol(model_name, _LinearModelProtocol())
    config: ExperimentConfig = _dummy_config(
        data={
            "name": dataset_name,
            "context_k": 0,
            "context_strategy": "empty",
            "clip_len": 1,
            "image_size": 1,
        },
        detector={
            "name": model_name,
            "dim": 1,
            "num_heads": 1,
            "num_classes": 2,
        },
    )

    loader: DataLoader[Any] = build_dataset(config, "validation")
    model: nn.Module = build_model(config)
    (features,) = next(iter(loader))

    assert config.data.root is None
    assert model(features).shape == (2, 2)


class _BadDatasetEndpoint:
    def build(self, config: ExperimentConfig, split: DatasetSplit) -> object:
        del config, split
        return object()


class _BadModelEndpoint:
    def build(self, config: ExperimentConfig) -> object:
        del config
        return object()


def test_registries_reject_invalid_torch_endpoints() -> None:
    dataset_name = "test_bad_dataset_endpoint"
    model_name = "test_bad_model_endpoint"
    register_dataset_protocol(dataset_name, _BadDatasetEndpoint())  # type: ignore[arg-type]
    register_model_protocol(model_name, _BadModelEndpoint())  # type: ignore[arg-type]

    data_config: ExperimentConfig = _dummy_config(
        data={
            "name": dataset_name,
            "context_k": 0,
            "context_strategy": "empty",
            "clip_len": 1,
        }
    )
    model_config: ExperimentConfig = _dummy_config(
        detector={"name": model_name, "dim": 1, "num_heads": 1}
    )

    with pytest.raises(TypeError, match="ожидался DataLoader"):
        build_dataset(data_config, "test")
    with pytest.raises(TypeError, match="ожидался nn.Module"):
        build_model(model_config)


def test_collator_pads_variable_context_and_reports_missing_fields() -> None:
    def sample(frame_id: int, slots: int) -> dict[str, Any]:
        return {
            "image": torch.zeros(3, 8, 8),
            "target": {
                "boxes": torch.zeros(1, 4),
                "labels": torch.zeros(1, dtype=torch.long),
            },
            "sequence_id": "sequence",
            "frame_id": frame_id,
            "timestamp": float(frame_id),
            "is_sequence_start": frame_id == 0,
            "context_valid_mask": torch.ones(slots, dtype=torch.bool),
            "context_time_offsets": torch.arange(1, slots + 1),
            "context_images": torch.zeros(slots, 3, 8, 8),
        }

    batch, context = DetectionCollator()([sample(0, 1), sample(1, 3)])

    assert batch.images.shape == (2, 3, 8, 8)
    assert context.valid_mask.tolist() == [[True, False, False], [True, True, True]]
    assert context.images is not None
    assert context.images.shape == (2, 3, 3, 8, 8)
    with pytest.raises(ValueError, match="не содержит ключи"):
        DetectionCollator()([{"image": torch.zeros(3, 8, 8)}])


def test_experiment_dataset_and_model_protocols_work_together(tmp_path: Path) -> None:
    OmegaConf.save({"name": "dummy", "root": None}, tmp_path / "dataset.yaml")
    config_path: Path = tmp_path / "experiment.yaml"
    OmegaConf.save(
        {
            "defaults": ["_self_"],
            "name": "component-protocol-test",
            "data": {
                "name": "dummy",
                "config_path": "dataset.yaml",
                "context_k": 1,
                "clip_len": 2,
                "image_size": 16,
            },
            "detector": {
                "name": "dummy",
                "dim": 8,
                "num_heads": 2,
                "num_queries": 4,
                "num_classes": 3,
            },
            "train": {"batch_size": 2, "num_workers": 0},
            "output": {"root": "results"},
            "clearml": {"enabled": False},
        },
        config_path,
    )

    def worker(
        run: ExperimentRun,
        config: ExperimentConfig,
        components: ExperimentComponents,
    ) -> dict[str, Any]:
        del config
        loader: DataLoader[Any] = components.loader("train")
        model: nn.Module = components.model
        batch, context = next(iter(loader))
        output, _ = model(batch, context)
        run.log_metrics({"mean_box": float(output.boxes.detach().mean())}, step=0)
        return {
            "dataset_endpoint": type(loader).__name__,
            "model_endpoint": type(model).__name__,
        }

    result_root: Path = ExperimentProtocol(tmp_path).execute_components(
        config_path,
        worker,
    )
    with (result_root / "metadata.json").open(encoding="utf-8") as stream:
        metadata: dict[str, Any] = json.load(stream)
    with (result_root / "summary.json").open(encoding="utf-8") as stream:
        summary: dict[str, Any] = json.load(stream)

    assert metadata["status"] == "completed"
    assert summary == {
        "dataset_endpoint": "DataLoader",
        "model_endpoint": "ContextDetector",
    }
    assert (result_root / "metrics.jsonl").read_text(encoding="utf-8")


def test_directory_components_own_code_config_and_artifacts(tmp_path: Path) -> None:
    project_root: Path = Path(__file__).resolve().parents[1]
    shutil.copytree(
        project_root / "datasets" / "synthetic_regression",
        tmp_path / "datasets" / "synthetic_regression",
    )
    shutil.copytree(
        project_root / "models" / "linear_regression",
        tmp_path / "models" / "linear_regression",
    )
    config_path: Path = tmp_path / "regression.yaml"
    OmegaConf.save(
        {
            "defaults": ["_self_"],
            "name": "directory-component-test",
            "data": {
                "name": "synthetic_regression",
                "component_path": "datasets/synthetic_regression",
                "config_path": "datasets/synthetic_regression/config.yaml",
                "context_k": 0,
                "context_strategy": "empty",
                "clip_len": 1,
                "image_size": 1,
            },
            "detector": {
                "name": "linear_regression",
                "component_path": "models/linear_regression",
                "config_path": "models/linear_regression/config.yaml",
                "dim": 1,
                "num_heads": 1,
                "num_classes": 1,
            },
            "train": {"batch_size": 32, "num_workers": 0},
            "validation": {"metrics": ["mse"], "batch_size": 64},
            "output": {"root": "results", "monitor": "mse"},
            "clearml": {"enabled": False},
        },
        config_path,
    )

    def worker(
        run: ExperimentRun,
        config: ExperimentConfig,
        components: ExperimentComponents,
    ) -> dict[str, Any]:
        del run, config
        inputs, targets = next(iter(components.loader("train")))
        predictions: torch.Tensor = components.model(inputs)
        torch.save(
            components.model.state_dict(),
            components.artifacts(ComponentKind.MODEL) / "model.pt",
        )
        return {
            "batch": inputs.shape[0],
            "target_shape": list(targets.shape),
            "prediction_shape": list(predictions.shape),
        }

    result_root: Path = ExperimentProtocol(tmp_path).execute_components(
        config_path,
        worker,
    )

    assert (tmp_path / "datasets/synthetic_regression/artifacts/train.pt").is_file()
    assert (
        tmp_path / "datasets/synthetic_regression/artifacts/validation.pt"
    ).is_file()
    assert (tmp_path / "models/linear_regression/artifacts/model.pt").is_file()
    assert (result_root / "artifacts/dataset__train.pt").is_file()
    assert (result_root / "artifacts/model__model.pt").is_file()
    with (result_root / "summary.json").open(encoding="utf-8") as stream:
        summary: dict[str, Any] = json.load(stream)
    assert summary["prediction_shape"] == summary["target_shape"]


def test_component_directory_rejects_incomplete_layout_and_name_mismatch(
    tmp_path: Path,
) -> None:
    component_root: Path = tmp_path / "datasets" / "broken"
    component_root.mkdir(parents=True)
    (component_root / "provider.py").write_text(
        "PROTOCOL = object()\n",
        encoding="utf-8",
    )
    OmegaConf.save({"name": "actual_name"}, component_root / "config.yaml")

    with pytest.raises(FileNotFoundError, match="artifacts"):
        ComponentDirectory.load(
            component_root,
            project_root=tmp_path,
            kind=ComponentKind.DATASET,
            expected_name="actual_name",
        )

    (component_root / "artifacts").mkdir()
    with pytest.raises(ValueError, match="не совпадает"):
        ComponentDirectory.load(
            component_root,
            project_root=tmp_path,
            kind=ComponentKind.DATASET,
            expected_name="different_name",
        )
