"""Точка запуска синтетической линейной регрессии."""

from __future__ import annotations

from pathlib import Path

from worker import run_regression

from context_detection.experiment import ExperimentProtocol


def main() -> None:
    """Запустить regression-worker через единый протокол."""
    experiment_dir: Path = Path(__file__).resolve().parent
    project_root: Path = experiment_dir.parents[1]
    protocol: ExperimentProtocol = ExperimentProtocol(project_root)
    result_dir: Path = protocol.execute_components(
        config_path=experiment_dir / "config.yaml",
        worker=run_regression,
        source_paths=[experiment_dir / "worker.py"],
        launch_script=__file__,
    )
    print(result_dir)


if __name__ == "__main__":
    main()
