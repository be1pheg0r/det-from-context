"""Интерфейс детектора и dummy-реализация. Человек 1.

Протокол согласован так, чтобы контекст подключался снаружи, без правки кода
адаптера: детектор отдаёт начальные queries, кто-то другой их модифицирует,
детектор доводит forward до конца.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from ..contracts import ContextBatch, DetectionBatch, DetectorOutput


class DetectorAdapter(nn.Module, ABC):
    """Детектор, из которого вытащены точки подключения контекста.

    Почему `initial_queries` вынесен отдельным методом: memory-модуль должен
    прочитать память ДО декодирования, а ключом чтения служат текущие queries.
    Если бы queries рождались только внутри forward, единственным способом
    добраться до них был бы хук или форк кода детектора — то самое, чего мы
    хотим избежать.
    """

    @abstractmethod
    def initial_queries(self, batch: DetectionBatch) -> Tensor:
        """[B, N, D] — queries до декодирования."""

    @abstractmethod
    def forward(
        self, batch: DetectionBatch, query_init: Tensor | None = None
    ) -> DetectorOutput:
        """query_init=None должно быть строго эквивалентно оригинальному
        детектору. Это проверяется regression-тестом Человека 3."""

    def encode_context_frames(self, context: ContextBatch) -> list[Tensor] | None:
        """Прогнать контекстные кадры через backbone/projector.

        Возвращает None, если пикселей нет: рекуррентные ветки берут контекст
        из MemoryState, и гонять для них backbone по K кадрам незачем.
        """
        return None

    def freeze(
        self,
        backbone: bool | None = None,
        decoder: bool = False,
        *,
        encoder: bool | None = None,
        bbox_embed: bool = False,
        cls_embed: bool = False,
    ) -> None:
        """Режимы заморозки detector-блоков.

        ``backbone`` остаётся legacy-алиасом для ``encoder``.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        """Размерность query. Нужна модулю памяти для создания слотов."""


class DummyDetector(DetectorAdapter):
    """Крошечный, но настоящий DETR-подобный детектор. Человек 1.

    Существует ради двух вещей: Люди 4 и 5 начинают работу не дожидаясь
    RF-DETR, и весь пайплайн проверяется на CPU за секунды. Не заглушка со
    случайными числами — у него есть обучаемые параметры и градиенты доходят
    до всех, иначе на нём нельзя было бы отладить ни память, ни цикл обучения.
    """

    def __init__(
        self,
        num_queries: int = 100,
        dim: int = 64,
        num_classes: int = 31,
        num_decoder_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self._dim = dim
        self.num_queries = num_queries

        # Трёхуровневый «backbone + projector»: каждый уровень вдвое мельче.
        # gcd, а не 8: конфиг проверяет только dim % num_heads, и dim=12 с
        # num_heads=4 проходил валидацию, а падал внутри GroupNorm без единого
        # упоминания ключа конфига, который это вызвал.
        groups = math.gcd(8, dim)
        self.stem = nn.Sequential(
            nn.Conv2d(3, dim, 3, stride=2, padding=1),
            nn.GroupNorm(groups, dim),
            nn.ReLU(inplace=True),
        )
        self.scales = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, stride=2, padding=1),
                nn.GroupNorm(groups, dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(2)
        )

        self.query_embed = nn.Embedding(num_queries, dim)
        self.ref_point_head = nn.Linear(dim, 4)
        self.decoder = nn.ModuleList(
            nn.MultiheadAttention(dim, num_heads, batch_first=True)
            for _ in range(num_decoder_layers)
        )
        self.decoder_norm = nn.ModuleList(
            nn.LayerNorm(dim) for _ in range(num_decoder_layers)
        )
        self.class_head = nn.Linear(dim, num_classes)
        self.box_head = nn.Linear(dim, 4)

    @property
    def dim(self) -> int:
        return self._dim

    def _features(self, images: Tensor) -> list[Tensor]:
        feats = [self.stem(images)]
        for scale in self.scales:
            feats.append(scale(feats[-1]))
        return feats

    def initial_queries(self, batch: DetectionBatch) -> Tensor:
        b = batch.batch_size
        return self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)

    def forward(
        self, batch: DetectionBatch, query_init: Tensor | None = None
    ) -> DetectorOutput:
        features = self._features(batch.images)
        # Плоская память для cross-attention: конкатенация всех масштабов.
        memory = torch.cat([f.flatten(2).transpose(1, 2) for f in features], dim=1)

        queries = self.initial_queries(batch) if query_init is None else query_init
        layers: list[dict[str, Tensor]] = []
        for attn, norm in zip(self.decoder, self.decoder_norm, strict=True):
            delta, _ = attn(queries, memory, memory, need_weights=False)
            queries = norm(queries + delta)
            logits, boxes, refs = self._heads(queries)
            layers.append(
                {
                    "queries": queries,
                    "logits": logits,
                    "boxes": boxes,
                    "reference_points": refs,
                }
            )

        # Последний слой уже посчитан в цикле — пересчёт дал бы три лишних
        # matmul и три лишних узла графа за forward. num_decoder_layers > 0
        # гарантирован конфигом (Field(gt=0)), так что layers непуст.
        last = layers[-1]
        return DetectorOutput(
            logits=last["logits"],
            boxes=last["boxes"],
            queries=queries,
            reference_points=last["reference_points"],
            features=features,
            decoder_layers=layers,
        )

    def _heads(self, queries: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """DAB-DETR: бокс предсказывается не с нуля, а как поправка к anchor'у
        в logit-пространстве.

        Не косметика: reference_points — это геометрическая часть слота памяти,
        и именно её модуль памяти переносит во времени. Если бы боксы жили
        отдельно от anchor'ов, перенесённый anchor ни на что бы не влиял, а
        память «не работала» бы по причине, которую очень трудно найти.
        """
        anchor = self.ref_point_head(queries)  # logit-пространство
        boxes = (anchor + self.box_head(queries)).sigmoid()
        return self.class_head(queries), boxes, anchor.sigmoid()

    def encode_context_frames(self, context: ContextBatch) -> list[Tensor] | None:
        if context.images is None:
            return None
        b, k = context.images.shape[:2]
        flat = context.images.flatten(0, 1)  # [B*K, C, H, W]
        return [f.unflatten(0, (b, k)) for f in self._features(flat)]

    def freeze(
        self,
        backbone: bool | None = None,
        decoder: bool = False,
        *,
        encoder: bool | None = None,
        bbox_embed: bool = False,
        cls_embed: bool = False,
    ) -> None:
        del bbox_embed, cls_embed
        freeze_encoder: bool = encoder if encoder is not None else bool(backbone)
        for p in (*self.stem.parameters(), *self.scales.parameters()):
            p.requires_grad_(not freeze_encoder)
        for p in self.decoder.parameters():
            p.requires_grad_(not decoder)
