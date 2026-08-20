"""RF-DETR + MeMOT worker backed by the official Lightning lifecycle."""

from __future__ import annotations

import json
from typing import Any

from experiments.rfdetr_image.worker import RFDetrImageExperiment
from pytorch_lightning import Trainer
from rfdetr.training import build_trainer

from context_detection.config import ExperimentConfig
from context_detection.experiment import ExperimentComponents, ExperimentRun
from context_detection.models.memot import MeMOTTracker
from context_detection.models.rfdetr_training import (
    ComponentRFDetrMeMOTModule,
    ProjectRFDetrDataModule,
    build_rfdetr_train_config,
)
from context_detection.monitoring import (
    ExperimentLightningLogger,
    MeMOTMonitoringCallback,
    MetricHistory,
)


class RFDetrMeMOTExperiment(RFDetrImageExperiment):
    """Fine-tune RF-DETR hypotheses and an external MeMOT adapter on clips."""

    def __init__(
        self,
        experiment: ExperimentRun,
        config: ExperimentConfig,
        components: ExperimentComponents,
    ) -> None:
        super().__init__(experiment, config, components)
        if not isinstance(components.model, MeMOTTracker):
            raise TypeError("video experiment requires external MeMOTTracker")
        self.tracker = components.model

    def run(self) -> dict[str, Any]:
        """Execute native fit and publish tracking availability explicitly."""
        has_test = "test" in self.components.dataloaders
        train_config = build_rfdetr_train_config(
            self.config,
            self.experiment.checkpoints_dir,
            class_names=self.class_names,
            has_test_split=has_test,
        )
        module = ComponentRFDetrMeMOTModule(
            self.tracker,
            train_config,
            self.config,
        )
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
            MeMOTMonitoringCallback(
                self.experiment,
                logger,
                class_names=self.class_names,
                every_n_steps=self.config.logging.every_n_steps,
                visualize_every_n_epochs=(self.config.logging.visualize_every_n_epochs),
                max_visual_images=self.config.logging.max_visual_images,
                max_diagnostic_images=self.config.logging.max_diagnostic_images,
                score_threshold=self.config.logging.prediction_score_threshold,
            )
        )
        self._record_runtime_metadata(module, trainer, has_test)
        self.experiment.record_metadata(
            "memot_architecture",
            {
                "boundary": "external-after-rfdetr-hypotheses",
                "num_slots": self.config.context.num_slots,
                "memory_length": self.config.context.memory_length,
                "short_memory_length": self.config.context.short_memory_length,
                "decoder_layers": self.config.context.memory_decoder_layers,
                "clip_len": self.config.data.clip_len,
                "detach_state": self.config.train.detach_state,
            },
        )
        self._publish_split_manifest()
        trainer.fit(module, datamodule=datamodule, ckpt_path=train_config.resume)
        self._publish_checkpoints()
        tracking_path = self.experiment.root / "logs" / "tracking-metrics.json"
        tracking_path.write_text(
            json.dumps(module.validation_tracking_result, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        self.experiment.save_artifact("tracking-metrics", tracking_path)
        self.experiment.record_metadata(
            "tracking_evaluation", module.validation_tracking_result
        )
        summary = self._summary(trainer, has_test)
        summary["external_memot"] = True
        summary["tracking"] = module.validation_tracking_result
        return summary


def run_rfdetr_memot(
    experiment: ExperimentRun,
    config: ExperimentConfig,
    components: ExperimentComponents,
) -> dict[str, Any]:
    """Execute the native RF-DETR + external MeMOT experiment."""
    return RFDetrMeMOTExperiment(experiment, config, components).run()
