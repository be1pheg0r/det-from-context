"""Prepare the editable project package before a DataSphere video job."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


def _stage(message: str) -> None:
    print(f"[bootstrap] {message}", flush=True)


def main() -> None:
    _stage("starting DataSphere bootstrap")
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    project_root = Path("/job/project")
    project_root.mkdir(parents=True, exist_ok=True)
    _stage(f"extracting project archive: {args.archive}")
    with ZipFile(args.archive) as archive:
        archive.extractall(project_root)
    _stage(f"archive extracted to {project_root}")
    experiment_dir = project_root / "experiments" / "rfdetr_memot"
    for path in (project_root, project_root / "src", experiment_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    _stage("installing pinned CUDA PyTorch")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-cache-dir",
            "torch==2.7.1",
            "torchvision==0.22.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu126",
        ],
        check=True,
    )
    _stage("PyTorch installation completed; installing project")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        check=True,
        cwd=project_root,
    )
    _stage("project installation completed; importing experiment entrypoint")
    from run import main as run_experiment

    _stage("starting RF-DETR + MeMOT experiment")
    run_experiment()
    _stage("experiment finished")


if __name__ == "__main__":
    main()
