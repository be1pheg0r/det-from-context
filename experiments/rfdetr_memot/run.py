"""Entrypoint for native RF-DETR + external MeMOT video fine-tuning."""

from __future__ import annotations

from pathlib import Path

try:
    from .worker import run_rfdetr_memot
except ImportError:  # Direct ``python experiments/rfdetr_memot/run.py`` execution.
    from worker import run_rfdetr_memot

from context_detection.config import load_config
from context_detection.experiment import ExperimentProtocol


def main() -> None:
    experiment_dir = Path(__file__).resolve().parent
    print(
        f"[rfdetr_memot] loading config from {experiment_dir / 'config.yaml'}",
        flush=True,
    )
    protocol = ExperimentProtocol(experiment_dir.parents[1])
    config = load_config(experiment_dir / "config.yaml")
    print("[rfdetr_memot] building train/validation/test components", flush=True)
    result_dir = protocol.execute_components(
        config_path=experiment_dir / "config.yaml",
        worker=run_rfdetr_memot,
        splits=tuple(config.data.splits),
        source_paths=[experiment_dir / "worker.py"],
        launch_script=__file__,
    )
    print(f"[rfdetr_memot] completed; result_dir={result_dir}", flush=True)


if __name__ == "__main__":
    main()
