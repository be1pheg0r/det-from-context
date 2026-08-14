"""RF-DETR за DetectorAdapter. Человек 3.

RF-DETR: DINOv2 backbone + multi-scale projector + неглубокий decoder с
multi-scale deformable cross-attention. Сам по себе не temporal. Неглубокий
decoder делает контроль добавленной latency особенно важным — любой memory-блок
здесь стоит заметную долю от полного forward.
"""

from __future__ import annotations

from ..contracts import ContextBatch, ContextOutput, DetectionBatch, DetectorOutput


class RFDetrAdapter:
    """TODO(чел.3).

    Логическое разделение (не обязательно на отдельные классы, но границы
    должны быть явные): backbone → projector → подготовка queries → decoder →
    heads.

    Точки подключения контекста — конфигом, без правки кода адаптера:
      * after_backbone   — контекст на уровне dense-фич
      * after_projector  — то же, но на multi-scale
      * before_decoder   — инициализация/дополнение queries из памяти
      * decoder_layer_i  — между слоями decoder

    ⚠️ Грабля №1 (MOTR/MOTRv2): memory-queries не должны замещать свежие.
    Держать два раздельных набора и сливать после получения независимых
    гипотез, а не подмешивать память в общий набор queries.

    ⚠️ Грабля №2 (DN-DETR): рекуррентный вход дестабилизирует венгерский
    матчинг. Denoising-ветку предусмотреть сразу, включать конфигом.
    """

    def __init__(self, variant: str, weights: str | None = None) -> None:
        raise NotImplementedError("Человек 3")

    def forward(
        self, batch: DetectionBatch, context: ContextOutput | None = None
    ) -> DetectorOutput:
        raise NotImplementedError("Человек 3")

    def encode_context_frames(self, context: ContextBatch) -> list:
        raise NotImplementedError("Человек 3")

    def freeze(self, backbone: bool = True, decoder: bool = False) -> None:
        raise NotImplementedError("Человек 3")


# TODO(чел.3): regression-тест — на одном изображении RFDetrAdapter(context=None)
# и оригинальный RF-DETR дают совпадающие logits/boxes в пределах допуска.
# Это единственная защита от «улучшений», которые на самом деле сломали baseline.
