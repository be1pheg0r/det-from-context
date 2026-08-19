"""Entrypoint for the two-epoch RF-DETR Datasphere smoke experiment."""

from __future__ import annotations

from pathlib import Path

from worker import run_rfdetr_image

from context_detection.experiment import ExperimentProtocol


def main() -> None:
    experiment_dir = Path(__file__).resolve().parent
    protocol = ExperimentProtocol(experiment_dir.parents[1])
    result_dir = protocol.execute_components(
        config_path=experiment_dir / "config.yaml",
        worker=run_rfdetr_image,
        source_paths=[experiment_dir / "worker.py"],
        launch_script=__file__,
    )
    print(result_dir)


if __name__ == "__main__":
    main()
