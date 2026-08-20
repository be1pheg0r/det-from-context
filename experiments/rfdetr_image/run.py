"""Entrypoint for native RF-DETR fine-tuning on an image dataset component."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

try:
    from .worker import run_rfdetr_image
except ImportError:  # Direct ``python experiments/rfdetr_image/run.py`` execution.
    from worker import run_rfdetr_image

from context_detection.experiment import ExperimentProtocol


def main() -> None:
    experiment_dir = Path(__file__).resolve().parent
    project_root = experiment_dir.parents[1]
    protocol = ExperimentProtocol(experiment_dir.parents[1])
    result_dir = protocol.execute_components(
        config_path=experiment_dir / "config.yaml",
        worker=run_rfdetr_image,
        splits=_configured_splits(
            project_root / "datasets/image_dataloader/config.yaml"
        ),
        source_paths=[experiment_dir / "worker.py"],
        launch_script=__file__,
    )
    print(result_dir)


def _configured_splits(dataset_config_path: Path) -> tuple[str, ...]:
    """Request test only when the dataset configuration declares it."""
    raw: Any = OmegaConf.to_container(OmegaConf.load(dataset_config_path), resolve=True)
    if not isinstance(raw, dict) or not isinstance(raw.get("splits"), dict):
        raise ValueError(f"{dataset_config_path}: splits mapping is required")
    split_config: dict[str, Any] = raw["splits"]
    has_test = (
        split_config.get("mode") == "predefined"
        or float(split_config.get("test_fraction", 0.0)) > 0.0
    )
    return ("train", "validation", "test") if has_test else ("train", "validation")


if __name__ == "__main__":
    main()
