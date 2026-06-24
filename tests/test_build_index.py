"""
tests/test_build_index.py — юнит-тесты чистых функций build_index (без сети).
Импорт build_index безопасен: openai импортируется лениво внутри embed_chunks.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import build_index as bi  # noqa: E402


def test_clean_strips_share_artifact():
    raw = "Заголовок кейсаСсылка скопирована\nПолезный текст про граф работ."
    out = bi.clean_source_text(raw)
    assert "Ссылка скопирована" not in out
    assert "Заголовок кейса" in out
    assert "граф работ" in out


def test_clean_drops_nav_lines():
    raw = "Лого\nО курсе\nКнига\nРеальное содержание главы.\nЛичный кабинет"
    out = bi.clean_source_text(raw)
    lines = out.splitlines()
    assert "Лого" not in lines
    assert "Книга" not in lines
    assert "Личный кабинет" not in lines
    assert "Реальное содержание главы." in lines


def test_clean_drops_all_cases_boilerplate():
    raw = ("Все кейсы — результат внедрения методологии AJTBD участниками тренинга в свой бизнес\n"
           "Компания: PLEADA.")
    out = bi.clean_source_text(raw)
    assert "Все кейсы — результат внедрения" not in out
    assert "Компания: PLEADA." in out


def test_clean_keeps_body_with_inline_title_phrase():
    # «Как делать продукт» как фраза в предложении НЕ должна вырезаться (удаляем только строку-пункт меню)
    raw = "Я прошёл тренинг «Как делать продукт» и вырос."
    out = bi.clean_source_text(raw)
    assert "Как делать продукт" in out


# --- split_sections ---
def test_split_sections_basic():
    raw = ("Лого\nОглавление\n"  # преамбула до первого маркера — отбрасывается
           "Глава перваяСсылка скопирована\nТело первой главы.\n"
           "Кейс PLEADAСсылка скопирована\nТело кейса PLEADA.")
    secs = bi.split_sections(raw)
    assert len(secs) == 2
    assert secs[0][0] == "Глава первая"          # маркер убран из заголовка
    assert "Тело первой главы." in secs[0][1]
    assert secs[1][0] == "Кейс PLEADA"
    assert "Преамбула" not in secs[0][1]
    assert "Лого" not in secs[0][1]


def test_split_sections_empty_when_no_markers():
    assert bi.split_sections("Просто текст без маркеров.\nЕщё строка.") == []


# --- dedup PART5 (книга) против PART6 (кейсы), по заголовку секции ---
def test_dedup_drops_book_chunks_duplicating_cases():
    chunks = [
        {"part": "PART6", "type": "transcript", "lecture": "Кейс PLEADA", "text": "ROMI вырос."},
        # тот же заголовок-кейс в книге, тело слегка иное → всё равно удалить
        {"part": "PART5", "type": "transcript", "lecture": "Кейс PLEADA", "text": "ROMI вырос (хвост иной)."},
        {"part": "PART5", "type": "transcript", "lecture": "Глава 1 — JTBD", "text": "Про граф работ."},
        {"part": "PART5", "type": "annotation", "lecture": "Кейс PLEADA", "text": "Аннотация."},
    ]
    kept, dropped = bi.dedup_part5_against_part6(chunks)
    assert dropped == 1
    lectures = [(c["part"], c["lecture"], c["type"]) for c in kept]
    assert ("PART5", "Глава 1 — JTBD", "transcript") in lectures   # книга — оставить
    assert ("PART5", "Кейс PLEADA", "annotation") in lectures      # аннотацию — оставить
    assert ("PART5", "Кейс PLEADA", "transcript") not in lectures  # дубль кейса — удалён


def test_make_context():
    assert bi.make_context("Кейсы AJTBD", "Кейс 1 — PLEADA") == "[Кейсы AJTBD · Кейс 1 — PLEADA]"
    assert bi.make_context("Основы AJTBD", "") == "[Основы AJTBD]"
