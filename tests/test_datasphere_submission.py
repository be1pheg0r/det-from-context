"""Tests for the secret-safe RF-DETR Datasphere submission helpers."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest
from experiments.rfdetr_image import submit_datasphere
from experiments.rfdetr_memot import submit_datasphere as memot_submission


def test_datasphere_cli_resolves_next_to_active_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suffix = ".exe" if submit_datasphere.os.name == "nt" else ""
    python = tmp_path / f"python{suffix}"
    cli = tmp_path / f"datasphere{suffix}"
    cli.touch()
    monkeypatch.setattr(submit_datasphere.shutil, "which", lambda _: None)
    monkeypatch.setattr(submit_datasphere.sys, "executable", str(python))

    assert submit_datasphere._datasphere_executable() == cli


def test_datasphere_cli_missing_error_is_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(submit_datasphere.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        submit_datasphere.sys,
        "executable",
        str(tmp_path / "python.exe"),
    )

    with pytest.raises(FileNotFoundError, match="project dependencies"):
        submit_datasphere._datasphere_executable()


def test_memot_datasphere_archive_excludes_agent_specs(tmp_path: Path) -> None:
    project = tmp_path / "project"
    for relative in memot_submission._PROJECT_SOURCES:
        path = project / relative
        if Path(relative).suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("source", encoding="utf-8")
        else:
            path.mkdir(parents=True, exist_ok=True)
            (path / "source.py").write_text("source", encoding="utf-8")
    ignored = project / "agent_specs" / "tasks" / "private.md"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("private", encoding="utf-8")
    archive = tmp_path / "project.zip"

    memot_submission._build_project_archive(project, archive)

    with ZipFile(archive) as stream:
        names = stream.namelist()
    assert names
    assert not any(name.startswith("agent_specs/") for name in names)
