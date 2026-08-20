"""RF-DETR training worker backed by the official PyTorch Lightning stack."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from rfdetr.training import build_trainer

from context_detection.config import ExperimentConfig
from context_detection.experiment import ExperimentComponents, ExperimentRun
from context_detection.models.rfdetr import RFDetrAdapter
from context_detection.models.rfdetr_training import (
    ComponentRFDetrModule,
    ProjectRFDetrDataModule,
    build_rfdetr_train_config,
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
        trainer: Trainer = build_trainer(
            train_config,
            module.model_config,
            accelerator=train_config.accelerator,
            logger=False,
        )
        self._record_runtime_metadata(module, trainer, has_test)
        trainer.fit(module, datamodule=datamodule, ckpt_path=train_config.resume)
        return self._summary(trainer, has_test)

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


def run_rfdetr_image(
    experiment: ExperimentRun,
    config: ExperimentConfig,
    components: ExperimentComponents,
) -> dict[str, Any]:
    """Execute the native RF-DETR image experiment."""
    return RFDetrImageExperiment(experiment, config, components).run()
