"""Конфиги и реестр имён. Без torch — этот файл должен проходить в пустом
окружении, чтобы опечатка в конфиге ловилась за секунды, а не за минуты.

Запуск: pytest -q   или   python tests/test_configs.py
"""

from __future__ import annotations

import pathlib

import pytest
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf
from pydantic import ValidationError

from context_detection import registry
from context_detection.config import ExperimentConfig, load_config

CONFIG_DIR = pathlib.Path(__file__).parent.parent / "configs"
CONFIGS = sorted(p for p in CONFIG_DIR.glob("*.yaml") if not p.name.startswith("_"))

#: Путь к датасету машинный, в репозитории его нет. Конфиги хранят root=null,
#: и это не заглушка — валидатор обязан ронять конфиг без root (см.
#: test_missing_root_rejected). Тесты подставляют его так же, как это сделает
#: Человек 5 на своей машине.
ROOT = ["data.root=/tmp/dataset"]


def test_configs_found():
    assert CONFIGS, "конфиги не найдены"


def test_every_config_loads():
    """Hydra composition + валидация. Ловит опечатку в имени ветки и лишний
    ключ (extra='forbid') до запуска обучения."""
    for path in CONFIGS:
        cfg = load_config(path, ROOT)
        assert isinstance(cfg, ExperimentConfig), path.name


def test_missing_root_rejected():
    """Конфиг без root не должен «проходить». Плейсхолдер вида "TODO" здесь
    страшнее отсутствия проверки: он делает тест выше зелёным и врёт."""
    non_dummy = [p for p in CONFIGS if p.name != "dummy.yaml"]
    assert non_dummy, "нечего проверять"
    for path in non_dummy:
        with pytest.raises(ValidationError):
            load_config(path)


def test_every_branch_has_a_config():
    """Ветка без конфига до сетки сравнения не доедет."""
    named = {load_config(p, ROOT).context.name for p in CONFIGS}
    missing = registry.CONTEXT_MODULES - named
    assert not missing, f"нет конфига для веток: {sorted(missing)}"


def test_overrides():
    cfg = load_config(CONFIG_DIR / "dummy.yaml", ["data.context_k=8", "train.lr=0.5"])
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
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        assert isinstance(raw, dict)
        name = raw.get("context", {}).get("name")
        if name is not None:
            assert name in registry.CONTEXT_MODULES, path.name


def test_every_config_uses_hydra_defaults_list():
    for path in CONFIGS:
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        assert isinstance(raw, dict)
        assert raw.get("defaults") == ["_base_", "_self_"]


def test_base_values_are_composed():
    cfg = load_config(CONFIG_DIR / "dummy.yaml")
    assert cfg.train.epochs == 12
    assert cfg.context.fusion == "residual"


def test_unknown_hydra_override_is_rejected():
    with pytest.raises(ConfigCompositionException):
        load_config(CONFIG_DIR / "dummy.yaml", ["train.learning_rate=0.5"])


def test_legacy_json_format_is_rejected():
    with pytest.raises(ValueError, match="Hydra YAML"):
        load_config(CONFIG_DIR / "dummy.json")


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_"):
            _fn()
            print("ok", _name)
