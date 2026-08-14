"""Конфиги и реестр имён. Без torch — этот файл должен проходить в пустом
окружении, чтобы опечатка в конфиге ловилась за секунды, а не за минуты.

Запуск: pytest -q   или   python tests/test_configs.py
"""

from __future__ import annotations

import json
import pathlib

import pytest
from pydantic import ValidationError

from context_detection import registry
from context_detection.config import ExperimentConfig, load_config

CONFIG_DIR = pathlib.Path(__file__).parent.parent / "configs"
CONFIGS = sorted(p for p in CONFIG_DIR.glob("*.json") if not p.name.startswith("_"))


def test_configs_found():
    assert CONFIGS, "конфиги не найдены"


def test_every_config_loads():
    """Раскрытие _base_ + валидация. Ловит опечатку в имени ветки и лишний
    ключ (extra='forbid') до запуска обучения."""
    for path in CONFIGS:
        cfg = load_config(path)
        assert isinstance(cfg, ExperimentConfig), path.name


def test_every_branch_has_a_config():
    """Ветка без конфига до сетки сравнения не доедет."""
    named = {load_config(p).context.name for p in CONFIGS}
    missing = registry.CONTEXT_MODULES - named
    assert not missing, f"нет конфига для веток: {sorted(missing)}"


def test_overrides():
    cfg = load_config(CONFIG_DIR / "dummy.json", ["data.context_k=8", "train.lr=0.5"])
    assert cfg.data.context_k == 8
    assert cfg.train.lr == 0.5


def test_unknown_branch_rejected():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"context": {"name": "нет такой ветки"}})


def test_extra_key_rejected():
    """Опечатка в имени ключа не должна молча игнорироваться — иначе человек
    неделю думает, что менял параметр."""
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"train": {"learning_rate": 1e-3}})


def test_dim_divisible_by_heads():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"detector": {"dim": 30, "num_heads": 4}})


def test_memory_branch_needs_multiframe_clip():
    """clip_len=1 с памятью — тихая бессмыслица: писать есть куда, читать
    никогда не будут."""
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {"context": {"name": "ema_slot"}, "data": {"clip_len": 1}}
        )


def test_cross_attn_needs_context_frames():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {"context": {"name": "cross_attn"}, "data": {"context_k": 0}}
        )


def test_registry_matches_configs():
    """Реестр — единственный источник правды; конфиги не должны его обгонять."""
    for path in CONFIGS:
        raw = json.loads(path.read_text(encoding="utf-8"))
        name = raw.get("context", {}).get("name")
        if name is not None:
            assert name in registry.CONTEXT_MODULES, path.name


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_"):
            _fn()
            print("ok", _name)
