"""Render and submit the RF-DETR + MeMOT DataSphere job without secrets."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from dotenv import load_dotenv
from omegaconf import OmegaConf

_PROJECT_SOURCES: tuple[str, ...] = (
    "pyproject.toml",
    "src",
    "datasets/video_dataloader",
    "models/rfdetr",
    "experiments/rfdetr_image",
    "experiments/rfdetr_memot",
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    project_id = os.environ.get("DATASPHERE_PROJECT_ID")
    if not project_id:
        raise RuntimeError("DATASPHERE_PROJECT_ID is required in .env")
    paths = _load_paths(Path(__file__).with_name("paths.yaml"))

    template = Path(__file__).with_name("datasphere_job.template.yaml")
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        archive_path = temporary / "project.zip"
        _build_project_archive(project_root, archive_path)
        rendered = (
            template.read_text(encoding="utf-8")
            .replace("__S3_CONNECTOR_ID__", paths["connector_id"])
            .replace("__PROJECT_ARCHIVE__", archive_path.as_posix())
            .replace("__VIDEO_DATASET_VIDEOS_DIR__", paths["videos_dir"])
            .replace("__VIDEO_DATASET_ANNOTATIONS_DIR__", paths["annotations_dir"])
            .replace("__RFDETR_MEMOT_OUTPUT_ROOT__", paths["output_root"])
        )
        config_path = temporary / "datasphere-job.yaml"
        config_path.write_text(rendered, encoding="utf-8")
        subprocess.run(
            [
                str(_datasphere_executable()),
                "project",
                "job",
                "execute",
                "-p",
                project_id,
                "-c",
                str(config_path),
            ],
            check=True,
            cwd=project_root,
        )


def _datasphere_executable() -> Path:
    """Resolve the CLI from PATH or the active virtual environment."""
    discovered = shutil.which("datasphere")
    if discovered is not None:
        return Path(discovered)
    suffix = ".exe" if os.name == "nt" else ""
    sibling = Path(sys.executable).with_name(f"datasphere{suffix}")
    if sibling.is_file():
        return sibling
    raise FileNotFoundError(
        "datasphere CLI is not installed; install the project dependencies first"
    )


def _build_project_archive(project_root: Path, archive_path: Path) -> None:
    """Archive only runtime sources; credentials and agent_specs stay outside."""
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for relative in _PROJECT_SOURCES:
            source = project_root / relative
            if source.is_file():
                archive.write(source, source.relative_to(project_root))
                continue
            for file_path in source.rglob("*"):
                if file_path.is_file():
                    archive.write(file_path, file_path.relative_to(project_root))


def _load_paths(path: Path) -> dict[str, str]:
    raw: Any = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("s3"), dict)
        or not isinstance(raw.get("data"), dict)
    ):
        raise ValueError(f"{path} must define s3 and data mappings")
    s3: Any = raw["s3"]
    data: Any = raw["data"]
    values = {
        "connector_id": s3.get("connector_id"),
        "videos_dir": data.get("videos_dir"),
        "annotations_dir": data.get("annotations_dir"),
        "output_root": raw.get("output_root"),
    }
    missing = [name for name, value in values.items() if not isinstance(value, str)]
    if missing:
        raise ValueError(f"{path} is missing string values: {', '.join(missing)}")
    return values


if __name__ == "__main__":
    main()
