"""Точка запуска эталонного эксперимента."""

from __future__ import annotations

import argparse
from pathlib import Path

from worker import run_dummy

from context_detection.experiment import ExperimentProtocol


def main() -> None:
    """Выполнить эксперимент с необязательными Hydra overrides."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--set", action="append", default=[])
    args: argparse.Namespace = parser.parse_args()

    experiment_dir: Path = Path(__file__).resolve().parent
    project_root: Path = experiment_dir.parents[1]
    protocol: ExperimentProtocol = ExperimentProtocol(project_root)
    result_dir: Path = protocol.execute(
        config_path=experiment_dir / "config.yaml",
        worker=run_dummy,
        overrides=args.set,
        source_paths=[experiment_dir / "worker.py"],
        launch_script=__file__,
    )
    print(result_dir)


if __name__ == "__main__":
    main()
