"""
tests/test_rag.py — юнит-тесты конвейера качества (PRPs/rag-answer-quality.md).
OpenAI замокан; сетевых вызовов нет. Импортируем только app/rag.py (не telegram_bot).
"""
import os
import sys

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import rag  # noqa: E402


# ---------------------------------------------------------------------------
# Фикстуры: крошечный фейковый индекс
# ---------------------------------------------------------------------------
def _make_index():
    chunks = [
        {"part": "PART1", "part_title": "Основы", "type": "transcript",
         "lecture": "", "text": "ценность продукта первопричина выбора клиента", "id": 0},
        {"part": "PART1", "part_title": "Основы", "type": "transcript",
         "lecture": "", "text": "совершенно другая тема про погоду и дождь", "id": 1},
    ]
    # chunk0 → ось x, chunk1 → ось y
    embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    bm25 = BM25Okapi([c["text"].lower().split() for c in chunks])
    return chunks, embeddings, bm25


def _make_cases(n_titles=25, n_annot=11):
    # n_titles реальных кейсов (транскрипты с описательными заголовками)
    # + n_annot курируемых аннотаций «Кейс N — …» (summary только для первых кейсов).
    transcripts = [
        {"part": "PART6", "part_title": "Кейсы AJTBD", "type": "transcript",
         "lecture": f"Кейс-история {i}", "text": f"Тело кейса {i}.", "id": 1000 + i}
        for i in range(1, n_titles + 1)
    ]
    annotations = [
        {"part": "PART6", "part_title": "Кейсы AJTBD", "type": "annotation",
         "lecture": f"Кейс {i} — Компания {i}", "text": f"Полный текст кейса {i}.", "id": i}
        for i in range(1, n_annot + 1)
    ]
    return transcripts + annotations


# ---------------------------------------------------------------------------
# P3 — rank_and_filter: happy / порог-отсечка / пусто
# ---------------------------------------------------------------------------
def test_rank_and_filter_happy(monkeypatch):
    monkeypatch.setattr(rag, "MIN_COSINE", 0.3)
    chunks, embeddings, bm25 = _make_index()
    qvec = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # совпадает с chunk0

    out = rag.rank_and_filter(qvec, chunks, embeddings, bm25,
                              query="ценность продукта", part_filter=None)

    assert len(out) == 1
    assert out[0]["id"] == 0
    assert "raw_cosine" in out[0]
    assert out[0]["raw_cosine"] == pytest.approx(1.0, abs=1e-3)


def test_rank_and_filter_threshold_drops_irrelevant(monkeypatch):
    monkeypatch.setattr(rag, "MIN_COSINE", 0.3)
    chunks, embeddings, bm25 = _make_index()
    qvec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    out = rag.rank_and_filter(qvec, chunks, embeddings, bm25,
                              query="ценность", part_filter=None)
    # chunk1 (косинус 0.0) не должен пройти порог
    assert all(c["id"] != 1 for c in out)


def test_rank_and_filter_empty_when_nothing_passes(monkeypatch):
    monkeypatch.setattr(rag, "MIN_COSINE", 0.3)
    chunks, embeddings, bm25 = _make_index()
    qvec = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # ортогонален обоим → косинус 0

    out = rag.rank_and_filter(qvec, chunks, embeddings, bm25,
                              query="нерелевантно", part_filter=None)
    assert out == []


def test_rrf_prefers_doc_ranked_high_by_both(monkeypatch):
    # RRF-свойство: документ, высокий и по dense, и по BM25, обходит высоких лишь по одному.
    monkeypatch.setattr(rag, "MIN_COSINE", 0.0)  # без отсечки — проверяем чистый порядок
    chunks = [
        {"part": "P", "part_title": "", "type": "transcript", "lecture": "",
         "text": "alpha beta gamma delta", "id": 0},   # высокий cos + высокий BM25
        {"part": "P", "part_title": "", "type": "transcript", "lecture": "",
         "text": "epsilon zeta eta", "id": 1},          # высокий cos, нулевой BM25
        {"part": "P", "part_title": "", "type": "transcript", "lecture": "",
         "text": "alpha theta", "id": 2},               # низкий cos, средний BM25
    ]
    embeddings = np.array([[1.0, 0.0, 0.0], [0.95, 0.05, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    bm25 = BM25Okapi([c["text"].lower().split() for c in chunks])
    qvec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    out = rag.rank_and_filter(qvec, chunks, embeddings, bm25,
                              query="alpha beta gamma", part_filter=None)
    assert out[0]["id"] == 0  # высок по обоим ретриверам → первый по RRF


# ---------------------------------------------------------------------------
# P7 — try_meta_answer
# ---------------------------------------------------------------------------
def test_meta_count_cases():
    # Считаем реальные кейсы (25 транскриптов), а не отставшие аннотации (11).
    out = rag.try_meta_answer("Сколько всего кейсов?", _make_cases())
    assert out is not None
    assert "25" in out
    assert "Кейс-история 8" in out


def test_meta_give_case_n_uses_annotation():
    # Для первых 11 кейсов есть курируемая аннотация — отдаём её текст.
    out = rag.try_meta_answer("Дай кейс 8", _make_cases())
    assert out == "Полный текст кейса 8."


def test_meta_give_case_n_without_annotation():
    # Кейс 15 есть в базе (25), но без аннотации — отдаём заголовок.
    out = rag.try_meta_answer("Дай кейс 15", _make_cases())
    assert out is not None
    assert "Кейс-история 15" in out


def test_meta_give_case_n_not_found():
    out = rag.try_meta_answer("Дай кейс 99", _make_cases())
    assert out is not None
    assert "99" in out
    assert "25" in out


def test_meta_list_parts():
    chunks = _make_cases(n_titles=2, n_annot=2) + [
        {"part": "PART1", "part_title": "Основы AJTBD", "type": "transcript",
         "lecture": "", "text": "...", "id": 100},
    ]
    out = rag.try_meta_answer("Какие части курса есть?", chunks)
    assert out is not None
    assert "Части курса" in out
    assert "PART1: Основы AJTBD" in out


def test_meta_returns_none_for_normal_question():
    out = rag.try_meta_answer("Что такое ценность продукта?", _make_cases())
    assert out is None


# ---------------------------------------------------------------------------
# P1/P2/P4 — rewrite_query
# ---------------------------------------------------------------------------
def test_rewrite_skips_short_standalone_question(monkeypatch):
    def boom():
        raise AssertionError("OpenAI не должен вызываться для короткого автономного вопроса")

    monkeypatch.setattr(rag, "_client", boom)
    msg = "Что такое граф работ?"
    assert rag.rewrite_query([], msg) == msg


def test_rewrite_does_not_skip_multiline_paste(monkeypatch):
    # Короткое, но МНОГОСТРОЧНОЕ сообщение (вставка + вопрос) не должно скипаться (P2)
    called = {"n": 0}

    class _Msg:
        content = "ценность продукта"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            called["n"] += 1
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _FakeClient:
        chat = _Chat()

    monkeypatch.setattr(rag, "_client", lambda: _FakeClient())
    out = rag.rewrite_query([], "Мой план: сделать бота.\nКакая здесь ценность продукта?")
    assert called["n"] == 1
    assert out == "ценность продукта"


def test_rewrite_uses_llm_with_history(monkeypatch):
    class _Msg:
        content = "методология AJTBD граф работ подробно"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            # история должна попасть в user-сообщение
            user_msg = kwargs["messages"][-1]["content"]
            assert "О чём книга" in user_msg
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _FakeClient:
        chat = _Chat()

    monkeypatch.setattr(rag, "_client", lambda: _FakeClient())
    history = [
        {"role": "user", "content": "О чём книга?"},
        {"role": "assistant", "content": "Книга про методологию AJTBD."},
    ]
    out = rag.rewrite_query(history, "Можешь разобрать подробнее?")
    assert out == "методология AJTBD граф работ подробно"


def test_rewrite_falls_back_on_api_error(monkeypatch):
    def _raise():
        raise RuntimeError("API down")

    monkeypatch.setattr(rag, "_client", _raise)
    # длинное сообщение → не сработает skip-эвристика → пойдёт в LLM → упадёт → fallback
    long_msg = "Вот мой большой план развития продукта. " * 20
    assert rag.rewrite_query([{"role": "user", "content": "x"}], long_msg) == long_msg


# ---------------------------------------------------------------------------
# format_citation — одна строка-цитата
# ---------------------------------------------------------------------------
def test_citation_lecture_with_minute():
    chunk = {"part": "PART1", "part_title": "Часть 1: Основы",
             "lecture": "Лекция 6 — Как проводить глубинные интервью", "lecture_minute": 12}
    out = rag.format_citation(chunk)
    assert "📺" in out
    assert "Лекция 6" in out
    assert "~12 мин" in out


def test_citation_lecture_without_minute():
    chunk = {"part": "PART3", "part_title": "Часть 3", "lecture": "Лекция 2", "lecture_minute": None}
    out = rag.format_citation(chunk)
    assert "📺" in out
    assert "мин" not in out


def test_citation_case():
    chunk = {"part": "PART6", "part_title": "Кейсы AJTBD",
             "lecture": "Кейс 8 — Kotlin Multiplatform", "lecture_minute": None}
    out = rag.format_citation(chunk)
    assert out.startswith("📁")
    assert "Кейс 8" in out
    assert "мин" not in out


def test_citation_book():
    chunk = {"part": "PART5", "part_title": "Книга AJTBD",
             "lecture": "Глава 2 — Введение", "lecture_minute": None}
    out = rag.format_citation(chunk)
    assert "📖" in out
    assert "Глава 2" in out


def test_citation_fallback_no_lecture():
    chunk = {"part": "PART2", "part_title": "Часть 2: Ценность", "lecture": "", "lecture_minute": None}
    out = rag.format_citation(chunk)
    assert out == "📚 Часть 2: Ценность"
    assert "мин" not in out


# ---------------------------------------------------------------------------
# P1 — обрезка истории (логика из telegram_bot, проверяем константу+поведение)
# ---------------------------------------------------------------------------
def test_history_cap():
    hist = [{"role": "user", "content": str(i)} for i in range(10)]
    del hist[:-rag.HISTORY_MAX_MSGS]
    assert len(hist) == rag.HISTORY_MAX_MSGS
    assert hist[0]["content"] == str(10 - rag.HISTORY_MAX_MSGS)
