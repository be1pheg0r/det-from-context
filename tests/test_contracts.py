"""Контракты и сквозной dummy-пайплайн. Требует torch + pydantic.

⚠️ Раньше этот файл был torch-free. Контракты стали Pydantic-моделями поверх
Tensor, поэтому проверить их без torch физически нельзя. Быстрые torch-free
проверки живут в test_configs.py — они и остаются первой линией.

Запуск: pytest -q   или   python tests/test_contracts.py
"""

from __future__ import annotations

import pathlib

import pytest
import torch
from pydantic import ValidationError

from context_detection import registry
from context_detection.build import _CONTEXT_MODULES, build_model, make_dummy_batch
from context_detection.config import ExperimentConfig, load_config
from context_detection.contracts import (
    ContextBatch,
    ContextOutput,
    DetectionBatch,
    DetectorOutput,
    MemoryState,
)
from context_detection.models.detector import DetectorAdapter
from context_detection.models.memory import ContextModule

CONFIG_DIR = pathlib.Path(__file__).parent.parent / "configs"

DUMMY_CFG = ExperimentConfig.model_validate(
    {
        "data": {"name": "dummy", "clip_len": 2, "image_size": 32},
        "detector": {"name": "dummy", "num_queries": 8, "dim": 16, "num_heads": 4},
        "context": {"name": "none"},
    }
)


def _fields(model) -> set[str]:
    return set(model.model_fields)


# --- состав контрактов -------------------------------------------------------


def test_detector_output_exposes_context_hooks():
    """Без этих полей memory-ветки подключить не к чему."""
    assert {
        "logits",
        "boxes",
        "queries",
        "reference_points",
        "features",
        "decoder_layers",
    } <= _fields(DetectorOutput)


def test_memory_slot_shape():
    """Состав слота из литобзора: без timestamp и motion пропагация слепая,
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
    } <= _fields(MemoryState)


def test_protocol_methods():
    """read/write/reset разделены — иначе гейт записи некуда встроить."""
    for name in ("read", "write", "reset", "init_state"):
        assert hasattr(ContextModule, name), f"ContextModule.{name} пропал"
    for name in ("forward", "initial_queries", "encode_context_frames", "freeze"):
        assert hasattr(DetectorAdapter, name), f"DetectorAdapter.{name} пропал"


# --- валидаторы --------------------------------------------------------------


def test_dummy_batch_is_valid():
    batch, context = make_dummy_batch(batch_size=2, context_k=4, image_size=32)
    assert batch.batch_size == 2
    assert context.num_slots == 4
    assert not context.valid_mask.all(), "dummy обязан содержать невалидные слоты"


def test_dummy_batch_exercises_the_reset_path():
    """Без is_sequence_start=True хотя бы на одном элементе ContextDetector
    никогда не заходит в context_module.reset, и утечку памяти между сценами
    не ловит вообще ничто — ни один dummy-прогон, ни CI."""
    batch, context = make_dummy_batch(batch_size=3, context_k=4, image_size=32)
    assert batch.is_sequence_start.any(), "reset-ветка не будет вызвана ни разу"
    # Начало сцены — это отсутствие истории. Иначе фикстура противоречит сама
    # себе и Человек 4 отлаживается против невозможного состояния.
    starts = batch.is_sequence_start
    assert not context.valid_mask[starts].any()
    assert (batch.frame_id == context.valid_mask.sum(1)).all()


def test_unnormalized_boxes_rejected():
    """Самый частый баг датасета: забыли поделить на размер изображения."""
    batch, _ = make_dummy_batch(batch_size=1, context_k=2, image_size=32)
    bad = [
        {
            "boxes": torch.tensor([[10.0, 20.0, 5.0, 5.0]]),
            "labels": torch.zeros(1, dtype=torch.int64),
        }
    ]
    with pytest.raises(ValidationError, match="вне"):
        DetectionBatch(
            images=batch.images,
            targets=bad,
            sequence_id=["s"],
            frame_id=batch.frame_id,
            timestamp=batch.timestamp,
        )


def test_batch_size_mismatch_rejected():
    batch, _ = make_dummy_batch(batch_size=2, context_k=2, image_size=32)
    with pytest.raises(ValidationError):
        DetectionBatch(
            images=batch.images,
            targets=batch.targets,
            sequence_id=["only-one"],
            frame_id=batch.frame_id,
            timestamp=batch.timestamp,
        )


def test_future_context_rejected():
    """time_offset <= 0 в валидном слоте — это утечка будущего. Тихий баг,
    который улучшает метрики и делает результат бессмысленным."""
    with pytest.raises(ValidationError, match="утечк|> 0"):
        ContextBatch(
            valid_mask=torch.ones(1, 2, dtype=torch.bool),
            time_offsets=torch.tensor([[0.04, -0.04]]),
        )


def test_offsets_must_be_zero_in_invalid_slots():
    with pytest.raises(ValidationError):
        ContextBatch(
            valid_mask=torch.tensor([[True, False]]),
            time_offsets=torch.tensor([[0.04, 0.08]]),
        )


def test_is_sequence_start_defaults_to_false():
    batch, _ = make_dummy_batch(batch_size=2, context_k=2, image_size=32)
    plain = DetectionBatch(
        images=batch.images,
        targets=batch.targets,
        sequence_id=batch.sequence_id,
        frame_id=batch.frame_id,
        timestamp=batch.timestamp,
    )
    assert plain.is_sequence_start is not None
    assert not plain.is_sequence_start.any()


def test_non_finite_and_out_of_range_boxes_rejected():
    """Проверка боксов свёрнута в один перенос на хост (aminmax вместо
    isfinite+min+max) — покрытие NaN/±Inf при этом теряется первым."""
    from context_detection.contracts import _boxes_normalized

    for bad in (float("nan"), float("inf"), float("-inf"), 10.0):
        with pytest.raises(ValueError):
            _boxes_normalized("b", torch.tensor([[bad, 0.5, 0.1, 0.1]]))


def test_query_delta_is_required():
    """None — осмысленный ответ «контекста нет», и ветка обязана произнести
    его вслух. С дефолтом забытое поле тихо вырождает прогон в baseline."""
    with pytest.raises(ValidationError):
        ContextOutput(diagnostics={"active_slots": 0})


def test_dim_not_divisible_by_eight_builds():
    """GroupNorm(8, dim) требовал dim % 8 == 0, а конфиг проверяет только
    dim % num_heads: dim=12 проходил валидацию и падал внутри torch."""
    cfg = ExperimentConfig.model_validate(
        {
            "data": {"name": "dummy", "clip_len": 2},
            "detector": {"name": "dummy", "dim": 12, "num_heads": 4},
        }
    )
    batch, context = make_dummy_batch(batch_size=1, context_k=2, image_size=32)
    assert build_model(cfg)(batch, context)[0].queries.shape[-1] == 12


def test_nan_in_query_delta_rejected():
    with pytest.raises(ValidationError):
        ContextOutput(query_delta=torch.full((1, 2, 4), float("nan")))


# --- MemoryState -------------------------------------------------------------


def test_memory_reset_clears_only_masked_rows():
    """Память не должна течь между последовательностями — и не должна
    обнуляться там, где последовательность продолжается."""
    state = MemoryState.empty(2, 4, 8)
    state.feature.fill_(1.0)
    state.valid.fill_(True)
    out = state.reset(torch.tensor([True, False]))
    assert out.feature[0].abs().sum() == 0
    assert not out.valid[0].any()
    assert out.feature[1].abs().sum() > 0
    assert out.valid[1].all()


def test_memory_detach_breaks_graph():
    """Без detach BPTT растёт на всю последовательность."""
    state = MemoryState.empty(1, 2, 4)
    state.feature.requires_grad_(True)
    assert state.feature.requires_grad
    assert not state.detach().feature.requires_grad


def test_negative_age_rejected():
    with pytest.raises(ValidationError):
        MemoryState(
            feature=torch.zeros(1, 2, 4),
            box=torch.full((1, 2, 4), 0.5),
            timestamp=torch.zeros(1, 2),
            confidence=torch.zeros(1, 2),
            age=torch.full((1, 2), -1, dtype=torch.int64),
            valid=torch.zeros(1, 2, dtype=torch.bool),
        )


# --- сквозной пайплайн: DoD Человека 1 --------------------------------------


def test_dummy_pipeline_runs():
    """model = build_model(cfg); output = model(*make_dummy_batch())
    без датасета и без RF-DETR."""
    model = build_model(DUMMY_CFG)
    batch, context = make_dummy_batch(batch_size=2, context_k=3, image_size=32)
    output, state = model(batch, context)
    assert isinstance(output, DetectorOutput)
    assert output.logits.shape == (2, 8, DUMMY_CFG.detector.num_classes)
    assert output.queries.shape == (2, 8, 16)
    assert state is None, "NoContext не должен создавать состояние"


def test_pipeline_optimizes():
    """Главное требование: всё складывается в nn.Module и обучается.
    Проверяем, что градиенты доходят до параметров и шаг оптимизатора их меняет.
    """
    torch.manual_seed(0)
    model = build_model(DUMMY_CFG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    batch, context = make_dummy_batch(batch_size=2, context_k=3, image_size=32)

    before = model.detector.query_embed.weight.detach().clone()
    output, _ = model(batch, context)
    # Заглушка вместо настоящего лосса — он зона Человека 5.
    loss = output.logits.pow(2).mean() + output.boxes.pow(2).mean()
    loss.backward()

    without_grad = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not without_grad, f"градиент не дошёл до: {without_grad}"

    optimizer.step()
    assert not torch.allclose(before, model.detector.query_embed.weight)


def test_fusion_modes_all_run():
    """Все три fusion-режима должны работать и пропускать градиент."""
    from context_detection.models.wrapper import Fusion

    queries = torch.randn(2, 5, 16, requires_grad=True)
    delta = torch.randn(2, 5, 16)
    for mode in ("residual", "gated_residual", "concat_proj"):
        out = Fusion(mode, 16)(queries, delta)
        assert out.shape == queries.shape, mode
        out.sum().backward(retain_graph=True)
    assert queries.grad is not None


def test_fusion_passthrough_without_context():
    """NoContext обязан быть побитово равен детектору без обёртки — иначе
    baseline, относительно которого считаются все ΔAP, не является baseline."""
    from context_detection.models.wrapper import Fusion

    queries = torch.randn(2, 5, 16)
    for mode in ("residual", "gated_residual", "concat_proj"):
        assert Fusion(mode, 16)(queries, None) is queries, mode


def test_no_context_equals_bare_detector():
    torch.manual_seed(0)
    model = build_model(DUMMY_CFG)
    batch, context = make_dummy_batch(batch_size=2, context_k=3, image_size=32)
    model.eval()
    with torch.no_grad():
        wrapped, _ = model(batch, context)
        bare = model.detector(batch)
    assert torch.equal(wrapped.logits, bare.logits)
    assert torch.equal(wrapped.boxes, bare.boxes)


def test_dummy_config_file_builds():
    model = build_model(load_config(CONFIG_DIR / "dummy.json"))
    batch, context = make_dummy_batch(batch_size=1, context_k=2, image_size=32)
    output, _ = model(batch, context)
    assert output.logits.shape[0] == 1


def test_dummy_config_trains_the_backbone():
    """configs/dummy.json — точка входа для отладки цикла обучения. Если он
    унаследует freeze_backbone из _base.json, лосс будет падать за счёт голов,
    и «сквозной прогон обучается» окажется неправдой."""
    model = build_model(load_config(CONFIG_DIR / "dummy.json"))
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert not frozen, f"на dummy-конфиге заморожено: {frozen}"


def test_builder_covers_every_registered_branch():
    """_CONTEXT_MODULES — второй список имён рядом с registry. Разъедется —
    ветка упадёт голым KeyError на сборке."""
    assert set(_CONTEXT_MODULES) == registry.CONTEXT_MODULES


def test_needs_context_frames_matches_registry():
    """registry.NEEDS_CONTEXT_FRAMES (валидация конфига, без torch) и атрибут
    класса (рантайм) — одно знание в двух местах. Разъедется — ветка пройдёт
    валидацию с context_k=0 и молча получит encoded_context=None."""
    actual = {n for n, cls in _CONTEXT_MODULES.items() if cls.needs_context_frames}
    assert actual == registry.NEEDS_CONTEXT_FRAMES


def test_unimplemented_branch_reports_itself():
    """Заготовки Человека 4 не переопределяют read/write и потому абстрактны.
    Наружу должно лететь имя ветки, а не 'Can't instantiate abstract class'."""
    cfg = ExperimentConfig.model_validate(
        {"context": {"name": "ema_slot"}, "data": {"clip_len": 2, "name": "dummy"}}
    )
    with pytest.raises(NotImplementedError, match="ema_slot"):
        build_model(cfg)


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_"):
            _fn()
            print("ok", _name)
