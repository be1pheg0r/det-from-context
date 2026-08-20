"""Tests for the secret-safe RF-DETR Datasphere submission helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from experiments.rfdetr_image import submit_datasphere


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
