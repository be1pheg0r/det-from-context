"""Тесты выбора контекста. Человек 2.

Без torch — эти тесты должны проходить в любом окружении. Всё, что здесь
проверяется, — чистая логика индексации, и именно она ломается тише всего:
утечка будущего не роняет обучение, она просто даёт красивые цифры.

Запуск: python -m pytest tests/ (или assert-скриптом, если pytest не ставим).
"""

from __future__ import annotations

# TODO(чел.2): раскомментировать по мере реализации.
# from context_detection.data.sequence_index import FrameRef, SequenceIndex
# from context_detection.data.context_sampler import sample_context


def test_no_future_leak():
    """TODO(чел.2). Для каждого кадра каждая позиция контекста имеет
    frame_id строго меньше текущего и тот же sequence_id. Проверить для ВСЕХ
    стратегий, включая random и uniform."""


def test_sequence_start_padding():
    """TODO(чел.2). Первый кадр последовательности: истории нет, все K слотов
    valid=False, длина результата всё равно ровно K."""


def test_partial_history():
    """TODO(чел.2). Кадр №2 при K=4: два валидных слота, два заглушки.
    Валидные идут от старых к новым."""


def test_scene_boundary():
    """TODO(чел.2). Кадры двух последовательностей вперемешку во входном
    списке — контекст никогда не берётся из чужой последовательности."""


def test_shuffled_keeps_offsets():
    """TODO(чел.2). Стратегия shuffled: множество выбранных кадров совпадает
    с prev_k, но соответствие «кадр ↔ time_offset» разорвано. Если этот тест
    написать неправильно, контрольный эксперимент ничего не контролирует."""


def test_empty_strategy():
    """TODO(чел.2). Все слоты невалидны независимо от длины истории."""


def test_duplicate_frames_rejected():
    """TODO(чел.2). Дубликат (sequence_id, frame_id) → ValueError при
    построении индекса, а не молчаливое поглощение."""
