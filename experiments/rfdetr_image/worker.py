"""RF-DETR training worker backed by the official PyTorch Lightning stack."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from rfdetr.training import build_trainer
from torch.utils.data import Subset

from context_detection.config import ExperimentConfig
from context_detection.experiment import ExperimentComponents, ExperimentRun
from context_detection.models.rfdetr import RFDetrAdapter
from context_detection.models.rfdetr_training import (
    ComponentRFDetrModule,
    ProjectRFDetrDataModule,
    build_rfdetr_train_config,
)
from context_detection.monitoring import (
    ExperimentLightningLogger,
    MetricHistory,
    RFDetrMonitoringCallback,
)


class RFDetrImageExperiment:
    """Fine-tune the component RF-DETR without converting data to COCO on disk."""

    def __init__(
        self,
        experiment: ExperimentRun,
        config: ExperimentConfig,
        components: ExperimentComponents,
    ) -> None:
        self.experiment = experiment
        self.config = config
        self.components = components
        detector = getattr(components.model, "detector", None)
        if not isinstance(detector, RFDetrAdapter):
            raise TypeError("RF-DETR experiment requires RFDetrAdapter")
        self.detector = detector
        self.class_names = _class_names(experiment.dataset_config_path)
        if len(self.class_names) != config.detector.num_classes:
            raise ValueError(
                "dataset class mapping and component model disagree: "
                f"{len(self.class_names)} != {config.detector.num_classes}"
            )
        model_resolution = int(getattr(detector.model_config, "resolution", 0))
        if model_resolution != config.data.image_size:
            raise ValueError(
                "RF-DETR model and dataset resolutions disagree: "
                f"{model_resolution} != {config.data.image_size}"
            )
        detector.model_config.amp = config.train.amp

    def run(self) -> dict[str, Any]:
        """Build upstream module/trainer, execute fit, and return scalar summary."""
        has_test = "test" in self.components.dataloaders
        train_config = build_rfdetr_train_config(
            self.config,
            self.experiment.checkpoints_dir,
            class_names=self.class_names,
            has_test_split=has_test,
        )
        module = ComponentRFDetrModule(self.detector, train_config)
        block_size = int(module.model_config.patch_size) * int(
            module.model_config.num_windows
        )
        datamodule = ProjectRFDetrDataModule(
            self.components.dataloaders,
            block_size=block_size,
            class_names=self.class_names,
        )
        history = MetricHistory()
        logger = ExperimentLightningLogger(self.experiment, history)
        trainer: Trainer = build_trainer(
            train_config,
            module.model_config,
            accelerator=train_config.accelerator,
            logger=logger,
        )
        trainer.callbacks.append(
            RFDetrMonitoringCallback(
                self.experiment,
                logger,
                class_names=self.class_names,
                every_n_steps=self.config.logging.every_n_steps,
                visualize_every_n_epochs=(self.config.logging.visualize_every_n_epochs),
                max_visual_images=self.config.logging.max_visual_images,
                max_diagnostic_images=self.config.logging.max_diagnostic_images,
                score_threshold=(self.config.logging.prediction_score_threshold),
            )
        )
        self._record_runtime_metadata(module, trainer, has_test)
        self._publish_split_manifest()
        trainer.fit(module, datamodule=datamodule, ckpt_path=train_config.resume)
        self._publish_checkpoints()
        return self._summary(trainer, has_test)

    def _publish_split_manifest(self) -> None:
        """Save exact split membership and a stable digest for reproducibility."""
        splits = {
            name: _loader_manifest(loader)
            for name, loader in sorted(self.components.dataloaders.items())
        }
        canonical = json.dumps(splits, ensure_ascii=False, sort_keys=True)
        payload = {
            "seed": self.config.train.seed,
            "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "counts": {name: len(samples) for name, samples in splits.items()},
            "splits": splits,
        }
        manifest_path = self.experiment.root / "logs" / "dataset-splits.json"
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.experiment.save_artifact("dataset-splits", manifest_path)
        self.experiment.record_metadata(
            "dataset_splits",
            {"sha256": payload["sha256"], "counts": payload["counts"]},
        )

    def _publish_checkpoints(self) -> None:
        """Apply retention to periodic checkpoints and publish retained files."""
        periodic = sorted(
            self.experiment.checkpoints_dir.glob("checkpoint_*.ckpt"),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        stale = periodic[: -self.config.output.keep_last_checkpoints]
        for path in stale:
            path.unlink()
        retained = sorted(
            path
            for path in self.experiment.checkpoints_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".ckpt", ".pth"}
        )
        published: list[str] = []
        for checkpoint in retained:
            artifact_name = f"checkpoint__{checkpoint.name}"
            self.experiment.save_artifact(artifact_name, checkpoint)
            published.append(artifact_name)
        self.experiment.record_metadata("checkpoints", published)

    def _record_runtime_metadata(
        self,
        module: ComponentRFDetrModule,
        trainer: Trainer,
        has_test: bool,
    ) -> None:
        """Persist non-secret runtime and upstream integration details."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        gpu_memory = (
            round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
            if torch.cuda.is_available()
            else 0.0
        )
        self.experiment.record_metadata(
            "rfdetr_training",
            {
                "framework": "rfdetr.training.RFDETRModelModule",
                "trainer": type(trainer).__name__,
                "device": device,
                "gpu_name": gpu_name,
                "gpu_memory_gb": gpu_memory,
                "precision": trainer.precision,
                "test_split": has_test,
                "class_names": self.class_names,
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in module.model.parameters()
                    if parameter.requires_grad
                ),
                "total_parameters": sum(
                    parameter.numel() for parameter in module.model.parameters()
                ),
            },
        )

    def _summary(self, trainer: Trainer, has_test: bool) -> dict[str, Any]:
        """Convert final Lightning callback metrics into JSON-compatible scalars."""
        metrics: dict[str, float] = {}
        for name, value in trainer.callback_metrics.items():
            tensor = torch.as_tensor(value).detach().cpu()
            if tensor.numel() == 1 and torch.isfinite(tensor):
                metrics[name] = float(tensor)
        return {
            "epochs": self.config.train.epochs,
            "train_samples": len(self.components.loader("train").dataset),
            "validation_samples": len(self.components.loader("validation").dataset),
            "test_samples": (
                len(self.components.loader("test").dataset) if has_test else 0
            ),
            "native_rfdetr_training": True,
            "metrics": metrics,
        }


def _class_names(config_path: Path) -> list[str]:
    """Return class names ordered by their contiguous dataset IDs."""
    raw: Any = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, dict) or not isinstance(raw.get("classes"), dict):
        raise ValueError(f"{config_path}: dataset classes mapping is required")
    classes: dict[str, Any] = raw["classes"]
    ordered = sorted(classes.items(), key=lambda item: item[1])
    if [class_id for _, class_id in ordered] != list(range(len(ordered))):
        raise ValueError(f"{config_path}: class IDs must be contiguous from zero")
    return [name for name, _ in ordered]


def _loader_manifest(loader: Any) -> list[dict[str, str]]:
    """Extract selected image/annotation pairs from a component DataLoader."""
    dataset = loader.dataset
    indices: list[int] | None = None
    if isinstance(dataset, Subset):
        indices = [int(index) for index in dataset.indices]
        dataset = dataset.dataset
    source = getattr(dataset, "dataset", dataset)
    manifest_method = getattr(source, "manifest", None)
    if not callable(manifest_method):
        raise TypeError(
            f"dataset {type(source).__name__} does not expose manifest() provenance"
        )
    manifest = manifest_method()
    selected = manifest if indices is None else [manifest[index] for index in indices]
    return [dict(sample) for sample in selected]


def run_rfdetr_image(
    experiment: ExperimentRun,
    config: ExperimentConfig,
    components: ExperimentComponents,
) -> dict[str, Any]:
    """Execute the native RF-DETR image experiment."""
    return RFDetrImageExperiment(experiment, config, components).run()
