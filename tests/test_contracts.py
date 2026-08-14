"""Проверка совместимости контрактов и конфигов. Без torch.

Ловит ровно один класс ошибок, но самый частый при работе впятером: кто-то
переименовал поле / добавил ветку / опечатался в конфиге, а узнают об этом
через полчаса запуска обучения.

Запуск: pytest -q   или   python tests/test_contracts.py
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

from context_detection import build, contracts
from context_detection.models.detector import DetectorAdapter
from context_detection.models.memory import ContextModule

CONFIGS = sorted((pathlib.Path(__file__).parent.parent / "configs").glob("*.json"))


def _fields(cls) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def test_contracts_import_without_torch():
    """Контракты должны читаться в пустом окружении — на них смотрят все пятеро."""
    import sys

    assert "torch" not in sys.modules, "контракты не должны тянуть torch на импорте"


def test_batch_fields():
    assert {"images", "targets", "sequence_id", "frame_id", "timestamp"} <= _fields(
        contracts.DetectionBatch
    )
    assert {"images", "valid_mask", "time_offsets"} <= _fields(contracts.ContextBatch)


def test_detector_output_exposes_context_hooks():
    """Без этих полей memory-ветки подключить не к чему."""
    assert {
        "logits",
        "boxes",
        "queries",
        "reference_points",
        "features",
        "decoder_layers",
    } <= _fields(contracts.DetectorOutput)


def test_memory_slot_shape():
    """Состав слота из литобзора. Без timestamp и motion пропагация слепая,
    без observed невозможен гейт записи (TSA)."""
    assert {
        "feature",
        "box",
        "timestamp",
        "confidence",
        "age",
        "valid",
        "observed",
        "motion",
    } <= _fields(contracts.MemoryState)


def test_context_output_carries_diagnostics():
    assert {"query_delta", "memory_state", "diagnostics"} <= _fields(
        contracts.ContextOutput
    )


def test_protocol_methods():
    """read/write/reset разделены — иначе гейт записи некуда встроить."""
    for name in ("read", "write", "reset"):
        assert hasattr(ContextModule, name), f"ContextModule.{name} пропал"
    for name in ("forward", "encode_context_frames", "freeze"):
        assert hasattr(DetectorAdapter, name), f"DetectorAdapter.{name} пропал"


def test_configs_are_valid():
    """Каждый конфиг ссылается на существующие имена веток."""
    assert CONFIGS, "конфиги не найдены"
    for path in CONFIGS:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        where = path.name
        if "_base_" in cfg:
            assert (path.parent / cfg["_base_"]).exists(), f"{where}: битый _base_"
        if "context" in cfg and "name" in cfg["context"]:
            assert cfg["context"]["name"] in build.CONTEXT_MODULES, where
        if "context" in cfg and "fusion" in cfg["context"]:
            assert cfg["context"]["fusion"] in build.FUSION_MODES, where
        if "data" in cfg and "name" in cfg["data"]:
            assert cfg["data"]["name"] in build.DATASETS, where
        if "data" in cfg and "context_strategy" in cfg["data"]:
            assert cfg["data"]["context_strategy"] in build.CONTEXT_STRATEGIES, where
        if "detector" in cfg and "name" in cfg["detector"]:
            assert cfg["detector"]["name"] in build.DETECTORS, where


def test_every_branch_has_a_config():
    """Ветка без конфига до сетки сравнения не доедет."""
    named = {
        json.loads(p.read_text(encoding="utf-8")).get("context", {}).get("name")
        for p in CONFIGS
    }
    missing = build.CONTEXT_MODULES - named
    assert not missing, f"нет конфига для веток: {sorted(missing)}"


if __name__ == "__main__":
    # Чтобы запускалось и без pytest.
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_"):
            _fn()
            print("ok", _name)
