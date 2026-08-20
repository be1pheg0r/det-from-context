"""Prepare the editable project package before running the DataSphere worker."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    project_root = Path("/job/project")
    project_root.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.archive) as archive:
        archive.extractall(project_root)
    experiment_dir = project_root / "experiments" / "rfdetr_image"
    for path in (project_root / "src", experiment_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    # ``python: auto`` may resolve the unpinned ``torch`` project dependency
    # to a CPU wheel even when DataSphere allocated a GPU instance.  Install a
    # CUDA-enabled wheel explicitly before installing the editable project.
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
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        check=True,
        cwd=project_root,
    )
    from run import main as run_experiment

    run_experiment()


if __name__ == "__main__":
    main()
