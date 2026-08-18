"""RF-DETR:  DINOv3 + deformable-декодер (DAB-DETR anchors).
Бейзлайн без временного контекста. Точки расширения (context_after_backbone/
context_after_projector/context_before_decoder),
начинку добавит memory.py.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from ..contracts import ContextBatch, DetectionBatch, DetectorOutput
from .detector import DetectorAdapter

DINOV3_HF_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"

# variant -> HF repo. "base" оставлен дефолтом ради обратной совместимости
# с текущим ноутбуком, где variant не передавался.
DINOV3_VARIANTS = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class _DINOv3Backbone(nn.Module):
    """(B,3,H,W) -> (B,embed_dim,H/patch,W/patch)."""

    def __init__(self, hf_name: str = DINOV3_HF_NAME, freeze: bool = False):
        super().__init__()
        # Токен только из окружения (huggingface-cli login / HF_TOKEN / Kaggle
        # secret) — никогда не хардкодить и не класть в конфиг, который может
        # уйти в git. None здесь безопасен: from_pretrained тогда попробует
        # уже закешированный логин, если он есть.
        token = os.environ.get("HF_TOKEN")
        self.model = AutoModel.from_pretrained(hf_name, token=token)
        self.patch_size = self.model.config.patch_size
        self.embed_dim = self.model.config.hidden_size
        self.num_special_tokens = 1 + getattr(
            self.model.config, "num_register_tokens", 0
        )
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        h, w = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size
        tokens = self.model(pixel_values=images).last_hidden_state
        patch_tokens = tokens[
            :, self.num_special_tokens :, :
        ]  # без CLS/регистр-токенов
        return patch_tokens.transpose(1, 2).reshape(B, self.embed_dim, h, w)


class _MultiScaleProjector(nn.Module):
    """Одна карта фичей -> n_levels карт убывающего разрешения (stride 1,2,4,...).
    GroupNorm вместо BatchNorm — не зависит от размера батча."""

    def __init__(self, in_dim: int, hidden_dim: int, n_levels: int = 3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1), nn.GroupNorm(32, hidden_dim)
        )
        self.downsample = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1
                    ),
                    nn.GroupNorm(32, hidden_dim),
                )
                for _ in range(n_levels - 1)
            ]
        )

    def forward(self, feat: torch.Tensor) -> list[torch.Tensor]:
        x = self.input_proj(feat)
        feats = [x]
        for down in self.downsample:
            x = down(x)
            feats.append(x)
        return feats


class _SinePositionEmbedding2D(nn.Module):
    """Синус-косинусное позиционное кодирование карты (B,C,H,W) -> (B,H*W,C)."""

    def __init__(self, num_pos_feats: int, temperature: int = 10000):
        super().__init__()
        assert num_pos_feats % 2 == 0
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        device = feat.device
        y = (
            torch.arange(H, dtype=torch.float32, device=device)
            .unsqueeze(1)
            .expand(H, W)
        )
        x = (
            torch.arange(W, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .expand(H, W)
        )
        y = y / (H + 1e-6) * 2 * math.pi
        x = x / (W + 1e-6) * 2 * math.pi
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x[..., None] / dim_t
        pos_y = y[..., None] / dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
        ).flatten(-2)
        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
        ).flatten(-2)
        pos = torch.cat([pos_y, pos_x], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        return pos.flatten(1, 2)


class _MSDeformAttn(nn.Module):
    """Multi-scale deformable cross-attention (Deformable DETR)."""

    def __init__(
        self, d_model: int = 256, n_levels: int = 3, n_heads: int = 8, n_points: int = 4
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_levels, self.n_heads, self.n_points = n_levels, n_heads, n_points
        self.head_dim = d_model // n_heads
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """reference_points: (B,Q,2) нормализ. (cx,cy) — центр anchor-бокса."""
        B, Q, C = query.shape
        H, P, D, L = self.n_heads, self.n_points, self.head_dim, self.n_levels
        value = self.value_proj(value).view(B, -1, H, D)
        value_list = value.split([h * w for h, w in spatial_shapes], dim=1)

        offsets = self.sampling_offsets(query).view(B, Q, H, L, P, 2)
        attn = self.attention_weights(query).view(B, Q, H, L * P)
        attn = F.softmax(attn, dim=-1).view(B, Q, H, L, P)

        out = query.new_zeros(B, H, D, Q)
        for lvl, (h, w) in enumerate(spatial_shapes):
            v = value_list[lvl].permute(0, 2, 3, 1).reshape(B * H, D, h, w)
            norm = query.new_tensor([w, h])
            loc = reference_points[:, :, None, None, :] + offsets[:, :, :, lvl] / norm
            loc = loc.clamp(0.0, 1.0) * 2 - 1
            loc = loc.permute(0, 2, 1, 3, 4).reshape(B * H, Q, P, 2)
            sampled = F.grid_sample(
                v, loc, mode="bilinear", padding_mode="zeros", align_corners=False
            )
            sampled = sampled.view(B, H, D, Q, P)
            w_lvl = attn[:, :, :, lvl, :].permute(0, 2, 1, 3)
            out = out + torch.einsum("bhdqp,bhqp->bhdq", sampled, w_lvl)
        out = out.permute(0, 3, 1, 2).reshape(B, Q, C)
        return self.output_proj(out)


def _bbox_mlp(
    dim_in: int, dim_hidden: int, dim_out: int, num_layers: int = 3
) -> nn.Sequential:
    dims = [dim_in] + [dim_hidden] * (num_layers - 1) + [dim_out]
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < num_layers - 1:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class _DecoderLayer(nn.Module):
    def __init__(
        self, d_model=256, n_heads=8, n_levels=3, n_points=4, d_ffn=1024, dropout=0.1
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = _MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, query_pos, ref_center, memory, spatial_shapes):
        q = k = tgt + query_pos
        sa_out, _ = self.self_attn(q, k, tgt)
        tgt = self.norm1(tgt + self.dropout(sa_out))
        ca_out = self.cross_attn(tgt + query_pos, ref_center, memory, spatial_shapes)
        tgt = self.norm2(tgt + self.dropout(ca_out))
        tgt = self.norm3(tgt + self.dropout(self.ffn(tgt)))
        return tgt


class _DeformableDecoder(nn.Module):
    """Стек слоёв с итеративным уточнением anchor-боксов (DAB-DETR, cxcywh).
    Голова класса/бокса на каждом слое — для aux-лосса и честного decoder_layers."""

    def __init__(
        self,
        d_model=256,
        n_heads=8,
        n_levels=3,
        n_points=4,
        d_ffn=1024,
        num_layers=3,
        num_classes=10,
        dropout=0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _DecoderLayer(d_model, n_heads, n_levels, n_points, d_ffn, dropout)
                for _ in range(num_layers)
            ]
        )
        self.class_heads = nn.ModuleList(
            [nn.Linear(d_model, num_classes) for _ in range(num_layers)]
        )
        self.bbox_heads = nn.ModuleList(
            [_bbox_mlp(d_model, d_model, 4, 3) for _ in range(num_layers)]
        )
        for bh in self.bbox_heads:
            nn.init.constant_(bh[-1].weight, 0.0)
            nn.init.constant_(bh[-1].bias, 0.0)

    def forward(self, tgt, query_pos, ref_boxes, memory, spatial_shapes):
        """ref_boxes: (B,Q,4) начальные anchor-боксы cxcywh.
        Возвращает per-layer списки logits / boxes / queries."""
        layer_logits, layer_boxes, layer_queries = [], [], []
        boxes = ref_boxes
        for i, layer in enumerate(self.layers):
            ref_center = boxes[..., :2]
            tgt = layer(tgt, query_pos, ref_center, memory, spatial_shapes)
            delta = self.bbox_heads[i](tgt)
            boxes = (inverse_sigmoid(boxes) + delta).sigmoid()
            logits = self.class_heads[i](tgt)
            layer_logits.append(logits)
            layer_boxes.append(boxes)
            layer_queries.append(tgt)
            boxes = boxes.detach()  # без этого градиент течёт через все слои сразу
        return layer_logits, layer_boxes, layer_queries


class RFDetrAdapter(DetectorAdapter):
    """RF-DETR: DINOv3 ViT-B/16 + multi-scale projector + deformable decoder
    с DAB-DETR anchor-боксами.

    Точка подключения контекста — единственная, что даёт протокол
    `DetectorAdapter`: `initial_queries` отдаёт query_content наружу,
    `ContextDetector` читает по нему память и возвращает уже слитые queries
    в `query_init`. Свои context_after_backbone/after_projector хуки убраны:
    `ContextDetector` их не вызывает, а не вызываемый код — источник багов,
    которые не ловятся regression-тестом. Возврат к ним — через отдельный
    callback между слоями decoder, согласованный с Человеком 1 (см. TODO).
    """

    def __init__(
        self,
        variant: str = "base",
        weights: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = config or {}
        if variant not in DINOV3_VARIANTS:
            raise ValueError(
                f"неизвестный variant {variant!r}, есть: {sorted(DINOV3_VARIANTS)}"
            )

        self.hidden_dim = self.config.get("hidden_dim", 256)
        self.n_levels = self.config.get("n_levels", 3)
        self.num_queries = self.config.get("num_queries", 300)
        self.num_classes = self.config.get("num_classes", 10)  # BDD100K detection

        self.backbone = _DINOv3Backbone(
            DINOV3_VARIANTS[variant], freeze=self.config.get("freeze_backbone", False)
        )
        self.projector = _MultiScaleProjector(
            self.backbone.embed_dim, self.hidden_dim, n_levels=self.n_levels
        )
        self.pos_embed = _SinePositionEmbedding2D(self.hidden_dim)
        self.level_embed = nn.Parameter(
            torch.randn(self.n_levels, self.hidden_dim) * 0.02
        )

        # content queries и обучаемые anchor-боксы (DAB-DETR)
        self.query_content = nn.Embedding(self.num_queries, self.hidden_dim)
        self.query_pos_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.anchor_boxes = nn.Parameter(
            torch.rand(self.num_queries, 4)
        )  # cxcywh в [0,1]

        self.decoder = _DeformableDecoder(
            d_model=self.hidden_dim,
            n_heads=self.config.get("n_heads", 8),
            n_levels=self.n_levels,
            n_points=self.config.get("n_points", 4),
            d_ffn=self.config.get("d_ffn", 1024),
            num_layers=self.config.get("num_decoder_layers", 3),
            num_classes=self.num_classes,
            dropout=self.config.get("dropout", 0.1),
        )

        if weights is not None:
            self.load_state_dict(torch.load(weights, map_location="cpu"))

    @property
    def dim(self) -> int:
        return self.hidden_dim

    def initial_queries(self, batch: DetectionBatch) -> torch.Tensor:
        b = batch.images.shape[0]
        return self.query_content.weight.unsqueeze(0).expand(b, -1, -1)

    def encode_context_frames(self, context: ContextBatch) -> list[torch.Tensor] | None:
        # MeMOT (и остальные рекуррентные ветки) читают MemoryState, а не
        # пиксели контекстных кадров — ContextModule.needs_context_frames=False
        # для них, и ContextDetector этот метод вообще не вызывает.
        return None

    def _flatten_levels(self, feats: list[torch.Tensor]):
        srcs, poss, shapes = [], [], []
        for lvl, f in enumerate(feats):
            H, W = f.shape[-2:]
            shapes.append((H, W))
            srcs.append(f.flatten(2).transpose(1, 2))
            poss.append(self.pos_embed(f) + self.level_embed[lvl].view(1, 1, -1))
        return torch.cat(srcs, dim=1), torch.cat(poss, dim=1), shapes

    def _anchor_to_pos(self, centers: torch.Tensor) -> torch.Tensor:
        """(B,Q,2) центры anchor'ов -> (B,Q,hidden_dim) синус-эмбеддинг."""
        device = centers.device
        num_pos_feats = self.hidden_dim // 2
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
        dim_t = 10000 ** (2 * (dim_t // 2) / num_pos_feats)
        pos = centers[..., None] * 2 * math.pi / dim_t  # (B,Q,2,num_pos_feats)
        pos = torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=-1).flatten(
            -2
        )
        return pos.flatten(2)  # (B,Q,hidden_dim)

    def forward(
        self, batch: DetectionBatch, query_init: torch.Tensor | None = None
    ) -> DetectorOutput:
        B = batch.images.shape[0]

        features = self.backbone(batch.images)
        ms_features = self.projector(features)
        memory, pos_flat, shapes = self._flatten_levels(ms_features)

        init_boxes = self.anchor_boxes.sigmoid().unsqueeze(0).expand(B, -1, -1)
        query_pos = self.query_pos_mlp(self._anchor_to_pos(init_boxes[..., :2]))

        # query_init=None -> оригинальный RF-DETR (regression baseline).
        # Фьюжн с памятью уже сделан снаружи, в ContextDetector.fusion —
        # здесь его повторять нельзя, иначе память смешивается дважды.
        queries = self.initial_queries(batch) if query_init is None else query_init

        layer_logits, layer_boxes, layer_queries = self.decoder(
            queries, query_pos, init_boxes, memory, shapes
        )

        decoder_layers = [
            {"queries": q, "boxes": b, "logits": logit}
            for q, b, logit in zip(
                layer_queries, layer_boxes, layer_logits, strict=True
            )
        ]

        return DetectorOutput(
            logits=layer_logits[-1],
            boxes=layer_boxes[-1],
            queries=layer_queries[-1],
            reference_points=layer_boxes[-1],
            features=ms_features,
            decoder_layers=decoder_layers,
            aux={},
        )

    def freeze(self, backbone: bool = True, decoder: bool = False) -> None:
        if backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if decoder:
            for p in self.decoder.parameters():
                p.requires_grad = False
