"""Лёгкие contract-тесты RF-DETR adapter без загрузки настоящих весов."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from torch import Tensor, nn

from context_detection.build import build_model
from context_detection.config import ExperimentConfig
from context_detection.contracts import ContextBatch, DetectionBatch
from context_detection.models import rfdetr as rfdetr_module
from context_detection.models.rfdetr import RFDetrAdapter


class _Feature:
    def __init__(self, tensors: Tensor) -> None:
        self.tensors: Tensor = tensors


class _NestedInput:
    def __init__(self, tensors: list[Tensor]) -> None:
        self.tensors: Tensor = torch.stack(tensors)


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 4, kernel_size=1)
        self.received_nested: bool = False

    def forward(
        self,
        images: Tensor | _NestedInput,
    ) -> tuple[list[_Feature], list[Tensor], None]:
        self.received_nested = isinstance(images, _NestedInput)
        if isinstance(images, _NestedInput):
            images = images.tensors
        first: Tensor = self.projection(images)
        second: Tensor = nn.functional.avg_pool2d(first, kernel_size=2)
        features = [_Feature(first), _Feature(second)]
        positions = [torch.zeros_like(first), torch.zeros_like(second)]
        return features, positions, None


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(
        self,
        target: Tensor,
        memory: Tensor,
        *,
        refpoints_unsigmoid: Tensor | None = None,
        **ignored: object,
    ) -> tuple[Tensor, Tensor]:
        del memory, ignored
        if refpoints_unsigmoid is None:
            raise ValueError("fake decoder requires reference points")
        layers: Tensor = torch.stack((target + self.scale, target + 2 * self.scale))
        references: Tensor = refpoints_unsigmoid.unsqueeze(0)
        return layers, references


class _Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = _Decoder()

    def forward(
        self,
        sources: list[Tensor],
        masks: list[Tensor],
        positions: list[Tensor],
        reference_points: Tensor,
        query_features: Tensor,
    ) -> tuple[Tensor, Tensor, None, None]:
        del masks, positions, reference_points
        batch_size: int = sources[0].shape[0]
        target: Tensor = query_features.unsqueeze(0).expand(batch_size, -1, -1)
        memory: Tensor = sources[0].flatten(2).transpose(1, 2)
        centers: Tensor = sources[0].mean(dim=(1, 2, 3)).sigmoid()
        encoder_references: Tensor = torch.stack(
            (
                centers,
                1 - centers,
                centers.new_full(centers.shape, 0.25),
                centers.new_full(centers.shape, 0.4),
            ),
            dim=-1,
        )
        encoder_references = encoder_references[:, None].expand(-1, target.shape[1], -1)
        queries, references = self.decoder(
            target,
            memory,
            refpoints_unsigmoid=encoder_references,
        )
        return queries, references, None, None


class _UpstreamModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _Backbone()
        self.transformer = _Transformer()
        self.query_feat = nn.Embedding(3, 4)
        self.class_embed = nn.Linear(4, 2)
        self.box_embed = nn.Linear(4, 4)
        self.bbox_reparam = True

    def forward(self, images: Tensor) -> dict[str, Any]:
        features, positions, _ = self.backbone(images)
        sources: list[Tensor] = [feature.tensors for feature in features]
        masks: list[Tensor] = [
            torch.zeros(
                source.shape[0],
                *source.shape[-2:],
                dtype=torch.bool,
                device=source.device,
            )
            for source in sources
        ]
        references: Tensor = images.new_zeros(self.query_feat.num_embeddings, 4)
        queries, _, _, _ = self.transformer(
            sources,
            masks,
            positions,
            references,
            self.query_feat.weight,
        )
        logits: Tensor = self.class_embed(queries)
        boxes: Tensor = self.box_embed(queries).sigmoid()
        return {
            "pred_logits": logits[-1],
            "pred_boxes": boxes[-1],
            "aux_outputs": [{"pred_logits": logits[0], "pred_boxes": boxes[0]}],
        }


class _Provider:
    last_kwargs: dict[str, object] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = kwargs
        self.model_config = SimpleNamespace(hidden_dim=4, num_queries=3)
        self.model = SimpleNamespace(model=_UpstreamModel())


def _install_fake_rfdetr(monkeypatch: pytest.MonkeyPatch) -> None:
    package = ModuleType("rfdetr")
    package.RFDETRNano = _Provider
    package.RFDETRSmall = _Provider
    package.RFDETRMedium = _Provider
    package.RFDETRLarge = _Provider
    tensor_utilities = ModuleType("rfdetr.utilities.tensors")
    tensor_utilities.nested_tensor_from_tensor_list = _NestedInput

    def fake_import(name: str) -> ModuleType:
        if name == "rfdetr":
            return package
        if name == "rfdetr.utilities.tensors":
            return tensor_utilities
        raise ImportError(name)

    monkeypatch.setattr(rfdetr_module, "import_module", fake_import)


def _batch(batch_size: int = 2) -> DetectionBatch:
    return DetectionBatch(
        images=torch.rand(batch_size, 3, 8, 8),
        targets=[
            {
                "boxes": torch.empty(0, 4),
                "labels": torch.empty(0, dtype=torch.long),
            }
            for _ in range(batch_size)
        ],
        sequence_id=[f"sequence-{index}" for index in range(batch_size)],
        frame_id=torch.zeros(batch_size, dtype=torch.long),
        timestamp=torch.zeros(batch_size),
        is_sequence_start=torch.ones(batch_size, dtype=torch.bool),
    )


def test_adapter_without_query_override_matches_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rfdetr(monkeypatch)
    adapter = RFDetrAdapter("nano")
    adapter.eval()
    batch: DetectionBatch = _batch()

    expected: dict[str, Any] = adapter.model(batch.images)
    output = adapter(batch)

    torch.testing.assert_close(output.logits, expected["pred_logits"])
    torch.testing.assert_close(output.boxes, expected["pred_boxes"])
    assert output.queries.shape == (2, 3, 4)
    assert len(output.features) == 2
    assert len(output.decoder_layers) == 2
    torch.testing.assert_close(
        output.decoder_layers[0]["logits"],
        expected["aux_outputs"][0]["pred_logits"],
    )


def test_adapter_injects_batch_specific_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rfdetr(monkeypatch)
    adapter = RFDetrAdapter("small")
    adapter.eval()
    batch: DetectionBatch = _batch()
    initial: Tensor = adapter.initial_queries(batch)
    modified: Tensor = initial + 3

    baseline = adapter(batch)
    output = adapter(batch, query_init=modified)

    torch.testing.assert_close(output.queries, modified + 2)
    assert not torch.equal(output.logits, baseline.logits)


def test_adapter_transforms_queries_at_decoder_boundary_and_preserves_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rfdetr(monkeypatch)
    adapter = RFDetrAdapter("small")
    adapter.eval()
    batch: DetectionBatch = _batch()
    observed: dict[str, Tensor] = {}

    def transform(queries: Tensor, references: Tensor | None) -> Tensor:
        assert references is not None
        observed["queries"] = queries.detach().clone()
        observed["references"] = references.detach().clone()
        return queries + 3

    output = adapter(batch, query_transform=transform)

    torch.testing.assert_close(output.queries, observed["queries"] + 5)
    torch.testing.assert_close(output.reference_points, observed["references"])
    assert not torch.equal(output.reference_points, output.boxes)


def test_adapter_encodes_context_and_freezes_upstream_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rfdetr(monkeypatch)
    adapter = RFDetrAdapter("medium")
    context = ContextBatch(
        images=torch.rand(2, 3, 3, 8, 8),
        valid_mask=torch.ones(2, 3, dtype=torch.bool),
        time_offsets=torch.ones(2, 3),
    )

    features = adapter.encode_context_frames(context)

    assert features is not None
    assert adapter.model.backbone.received_nested
    assert [feature.shape[:3] for feature in features] == [(2, 3, 4), (2, 3, 4)]
    adapter.freeze(backbone=True, decoder=True)
    assert not any(
        parameter.requires_grad for parameter in adapter.model.backbone.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in adapter.model.transformer.decoder.parameters()
    )
    adapter.freeze(backbone=False, decoder=False)
    assert all(
        parameter.requires_grad for parameter in adapter.model.backbone.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in adapter.model.transformer.decoder.parameters()
    )


def test_adapter_validates_variant_and_forwards_weight_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rfdetr(monkeypatch)

    adapter = RFDetrAdapter("rfdetr-large", weights="checkpoint.pth")

    assert adapter.variant.value == "large"
    assert _Provider.last_kwargs == {"pretrain_weights": "checkpoint.pth"}
    with pytest.raises(ValueError, match="неизвестный RF-DETR variant"):
        RFDetrAdapter("base")


def test_rfdetr_directory_component_builds_model_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rfdetr(monkeypatch)
    config = ExperimentConfig.model_validate(
        {
            "data": {
                "name": "dummy",
                "context_k": 0,
                "context_strategy": "empty",
                "clip_len": 1,
            },
            "detector": {
                "name": "rfdetr",
                "component_path": "models/rfdetr",
                "config_path": "models/rfdetr/config.yaml",
                "dim": 4,
                "num_heads": 1,
            },
            "context": {"name": "none"},
        }
    )

    model: nn.Module = build_model(config)
    batch: DetectionBatch = _batch()
    context = ContextBatch(
        valid_mask=torch.zeros(batch.batch_size, 0, dtype=torch.bool),
        time_offsets=torch.zeros(batch.batch_size, 0),
    )
    output, state = model(batch, context)

    assert type(model).__name__ == "ContextDetector"
    assert output.logits.shape == (2, 3, 2)
    assert state is None
    assert _Provider.last_kwargs["resolution"] == 512
    assert _Provider.last_kwargs["num_classes"] == 31
    assert _Provider.last_kwargs["pretrain_weights"] == "rf-detr-small.pth"


def test_rfdetr_no_context_is_exactly_bare_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rfdetr(monkeypatch)
    config = ExperimentConfig.model_validate(
        {
            "data": {
                "name": "dummy",
                "context_k": 0,
                "context_strategy": "empty",
                "clip_len": 1,
            },
            "detector": {
                "name": "rfdetr",
                "component_path": "models/rfdetr",
                "config_path": "models/rfdetr/config.yaml",
                "dim": 4,
                "num_heads": 1,
            },
            "context": {"name": "none"},
        }
    )
    model = build_model(config)
    model.eval()
    batch = _batch()
    context = ContextBatch(
        valid_mask=torch.zeros(batch.batch_size, 0, dtype=torch.bool),
        time_offsets=torch.zeros(batch.batch_size, 0),
    )

    with torch.no_grad():
        wrapped, _ = model(batch, context)
        bare = model.detector(batch)

    assert torch.equal(wrapped.logits, bare.logits)
    assert torch.equal(wrapped.boxes, bare.boxes)
    assert torch.equal(wrapped.reference_points, bare.reference_points)


def test_rfdetr_memot_runs_two_frames_with_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rfdetr(monkeypatch)
    config = ExperimentConfig.model_validate(
        {
            "data": {
                "name": "dummy",
                "context_k": 1,
                "context_strategy": "prev_k",
                "clip_len": 2,
            },
            "detector": {
                "name": "rfdetr",
                "component_path": "models/rfdetr",
                "config_path": "models/rfdetr/config.yaml",
                "dim": 4,
                "num_heads": 1,
                "group_detr": 1,
            },
            "context": {
                "name": "memot",
                "fusion": "gated_residual",
                "num_slots": 3,
                "memory_length": 2,
                "short_memory_length": 1,
                "write_threshold": 0.0,
            },
        }
    )
    model = build_model(config)
    model.train()
    context = ContextBatch(
        valid_mask=torch.ones(2, 1, dtype=torch.bool),
        time_offsets=torch.ones(2, 1),
    )
    first = _batch()
    first.timestamp = torch.zeros(2)
    _, state = model(first, context)
    assert state is not None

    second = _batch()
    second.timestamp = torch.ones(2)
    second.frame_id = torch.ones(2, dtype=torch.long)
    second.is_sequence_start = torch.zeros(2, dtype=torch.bool)
    output, next_state = model(second, context, state)
    (output.logits.sum() + output.boxes.sum()).backward()

    assert next_state is not None
    assert _Provider.last_kwargs["group_detr"] == 1
    assert any(
        parameter.grad is not None
        for parameter in model.context_module.parameters()
        if parameter.requires_grad
    )


def test_rfdetr_memot_rejects_grouped_training_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rfdetr(monkeypatch)
    config = ExperimentConfig.model_validate(
        {
            "data": {"name": "dummy", "clip_len": 2},
            "detector": {
                "name": "rfdetr",
                "component_path": "models/rfdetr",
                "config_path": "models/rfdetr/config.yaml",
                "dim": 4,
                "num_heads": 1,
            },
            "context": {"name": "memot", "num_slots": 3},
        }
    )

    with pytest.raises(ValueError, match="group_detr=1"):
        build_model(config)


@pytest.mark.filterwarnings(
    "ignore:`torch.jit.script` is deprecated:DeprecationWarning"
)
def test_component_config_passes_upstream_model_validation() -> None:
    upstream_config_module: Any = pytest.importorskip(
        "rfdetr.config",
        reason="официальная схема доступна только с установленным rfdetr",
    )

    config_path: Path = Path("models/rfdetr/config.yaml")
    raw: Any = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    assert isinstance(raw, dict)

    upstream_config = upstream_config_module.RFDETRSmallConfig(**raw["model"])

    assert upstream_config.hidden_dim == 256
    assert upstream_config.num_classes == 31
    assert upstream_config.resolution == 512
