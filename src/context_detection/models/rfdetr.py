"""RF-DETR за DetectorAdapter. Человек 3.

RF-DETR: DINOv2 backbone + multi-scale projector + неглубокий decoder с
multi-scale deformable cross-attention. Сам по себе не temporal. Неглубокий
decoder делает контроль добавленной latency особенно важным — memory-блок
здесь стоит заметную долю полного forward.

Ориентир по структуре — DummyDetector в detector.py: те же методы, та же
раскладка DetectorOutput, только настоящие веса.
"""

from __future__ import annotations

from torch import Tensor

from ..contracts import ContextBatch, DetectionBatch, DetectorOutput
from .detector import DetectorAdapter


class RFDetrAdapter(DetectorAdapter):
    """TODO(чел.3).

    Логические границы (не обязательно отдельные классы, но явные):
    backbone → projector → подготовка queries → decoder → heads.

    Точки подключения контекста конфигом, без правки кода адаптера:
      after_backbone / after_projector / before_decoder / decoder_layer.
    `before_decoder` уже покрыт базовым интерфейсом: initial_queries отдаёт
    queries наружу, ContextDetector их модифицирует и возвращает в forward.
    Для attach между слоями decoder понадобится callback — согласовать с
    Человеком 1 до реализации.

    ⚠️ Грабля №1 (MOTR/MOTRv2): memory-queries не должны замещать свежие.
    Держать два раздельных набора и сливать после получения независимых
    гипотез, а не подмешивать память в общий набор.

    ⚠️ Грабля №2 (DN-DETR): рекуррентный вход дестабилизирует венгерский
    матчинг. Denoising-ветку предусмотреть сразу, включать конфигом.
    """

    def __init__(self, variant: str, weights: str | None = None) -> None:
        super().__init__()
        raise NotImplementedError("Человек 3")

    @property
    def dim(self) -> int:
        raise NotImplementedError("Человек 3")

    def initial_queries(self, batch: DetectionBatch) -> Tensor:
        raise NotImplementedError("Человек 3")

    def forward(
        self, batch: DetectionBatch, query_init: Tensor | None = None
    ) -> DetectorOutput:
        raise NotImplementedError("Человек 3")

    def encode_context_frames(self, context: ContextBatch) -> list[Tensor] | None:
        raise NotImplementedError("Человек 3")

    def freeze(self, backbone: bool = True, decoder: bool = False) -> None:
        raise NotImplementedError("Человек 3")


# TODO(чел.3): regression-тест — на одном изображении RFDetrAdapter(query_init=None)
# и оригинальный RF-DETR дают совпадающие logits/boxes в пределах допуска.
# Единственная защита от «улучшений», которые на самом деле сломали baseline.
