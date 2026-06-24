"""
scripts/build_index.py
Parses markdown lecture files → chunks → embeds via OpenAI text-embedding-3-small → saves cache.
Run once: python scripts/build_index.py
Requires OPENAI_API_KEY in .env
"""

import hashlib
import json
import os
import re
import sys
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DATA_DIR        = os.path.join(os.path.dirname(__file__), "..", "data")
INDEX_DIR       = os.path.join(DATA_DIR, "zamezin_indexed")
CHUNKS_PATH     = os.path.join(DATA_DIR, "chunks_cache.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings_cache.npy")
META_PATH       = EMBEDDINGS_PATH + ".meta.json"  # хеш embed-входов для безопасного reuse

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE  = 100

# PART5/PART6 — плоские файлы-склейки (книга + кейсы) → режем на секции по заголовкам.
SECTION_PARTS = {"PART5", "PART6"}
# Маркер начала секции (главы/кейса): строка, оканчивающаяся артефактом кнопки шаринга.
SECTION_MARKER = "Ссылка скопирована"

PARTS = [
    ("PART1", "PART1_annotations.md", "PART1_osnovy.md",      "Основы AJTBD"),
    ("PART2", "PART2_annotations.md", "PART2_cennost.md",     "Ценность продукта"),
    ("PART3", "PART3_annotations.md", "PART3_zapusk.md",      "Запуск и сегменты"),
    ("PART4", "PART4_annotations.md", "PART4_strategiya.md",  "Стратегия"),
    # transcript paths with ../  are relative to INDEX_DIR, resolving into DATA_DIR
    ("PART5", "PART5_annotations.md", "../book_zamesin.txt",  "Книга AJTBD"),
    ("PART6", "PART6_annotations.md", "../cases_zamesin.txt", "Кейсы AJTBD"),
]

CHUNK_SIZE = 600  # approx tokens (chars / 4)

# Scraping-мусор в book_zamesin.txt / cases_zamesin.txt (nav-меню сайта + кнопка шаринга).
JUNK_LINES = {
    "Лого", "О курсе", "Открытая лекция", "Кейсы", "Книга",
    "Личный кабинет", "Оглавление книги", "Как делать продукт",
}
JUNK_SUBSTR = ["Ссылка скопирована", "Все кейсы — результат внедрения"]


def clean_source_text(text: str) -> str:
    """Убирает nav/share-мусор из исходного текста (book/cases) построчно.

    Удаляет строки-пункты меню (точное совпадение с JUNK_LINES), вырезает подстроки-артефакты
    (JUNK_SUBSTR, напр. суффикс кнопки «Ссылка скопирована»), отбрасывает пустые строки.

    ВАЖНО: для cases границы кейсов детектятся ПО суффиксу «Ссылка скопирована», поэтому чистку
    применять к сегменту ПОСЛЕ нарезки на кейсы/главы, а не к сырому файлу целиком.
    """
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s in JUNK_LINES:
            continue
        for j in JUNK_SUBSTR:
            s = s.replace(j, "")
        s = s.strip()
        if s:
            out.append(s)
    return "\n".join(out)


def split_sections(raw_text: str) -> list[tuple[str, str]]:
    """Режет плоский файл на секции по строкам-заголовкам, оканчивающимся на SECTION_MARKER.

    Возвращает [(title, body)] в порядке файла. Преамбула до первого маркера (nav/оглавление)
    отбрасывается. Подходит и для book (главы+гайды+кейсы), и для cases (25 кейсов).
    """
    sections = []
    cur_title = None
    cur_body: list[str] = []
    for line in raw_text.splitlines():
        if line.strip().endswith(SECTION_MARKER):
            if cur_title is not None:
                sections.append((cur_title, "\n".join(cur_body)))
            cur_title = line.strip()[: -len(SECTION_MARKER)].strip()
            cur_body = []
        elif cur_title is not None:
            cur_body.append(line)
    if cur_title is not None:
        sections.append((cur_title, "\n".join(cur_body)))
    return sections


def make_context(part_title: str, lecture: str) -> str:
    """Контекстный префикс к чанку (для эмбеддинга и BM25, не для показа)."""
    return f"[{part_title} · {lecture}]" if lecture else f"[{part_title}]"


def dedup_part5_against_part6(all_chunks: list[dict]) -> tuple[list[dict], int]:
    """Убирает из PART5 (книга) секции-кейсы, продублированные в PART6.

    book_zamesin.txt содержит все кейсы; их заголовки секций совпадают с заголовками кейсов
    PART6. Дедуп по ЗАГОЛОВКУ секции (а не по точному тексту: тело кейса в книге и в cases.txt
    может слегка отличаться хвостом → нарезка на чанки расходится). Возвращает (chunks, удалено).
    """
    case_titles = {c["lecture"] for c in all_chunks
                   if c["part"] == "PART6" and c["type"] == "transcript"}
    kept, dropped = [], 0
    for c in all_chunks:
        if (c["part"] == "PART5" and c["type"] == "transcript"
                and c["lecture"] in case_titles):
            dropped += 1
            continue
        kept.append(c)
    return kept, dropped


def parse_annotations(path: str, part_id: str, part_title: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        content = f.read()

    sections = re.split(r"(?=^## (?:Лекция|Глава|Кейс))", content, flags=re.MULTILINE)
    chunks = []
    for sec in sections:
        sec = sec.strip()
        if not re.match(r"^## (?:Лекция|Глава|Кейс)", sec):
            continue
        title = sec.splitlines()[0].lstrip("#").strip()
        chunks.append({
            "type":           "annotation",
            "part":           part_id,
            "part_title":     part_title,
            "lecture":        title,
            "lecture_minute": None,
            "text":           sec,
        })
    return chunks


def parse_lecture_meta(path: str) -> list[dict]:
    """Упорядоченный список лекций/глав/кейсов части: title + duration_min.

    duration_min = None, если у секции нет поля «Продолжительность» (книга/кейсы).
    Порядок секций совпадает с порядком в исходном транскрипте части.
    """
    with open(path, encoding="utf-8") as f:
        content = f.read()

    sections = re.split(r"(?=^## (?:Лекция|Глава|Кейс))", content, flags=re.MULTILINE)
    out = []
    for sec in sections:
        sec = sec.strip()
        if not re.match(r"^## (?:Лекция|Глава|Кейс)", sec):
            continue
        title = sec.splitlines()[0].lstrip("#").strip()
        m = re.search(r"Продолжительность.*?(\d+)\s*мин", sec)
        out.append({"title": title, "duration_min": int(m.group(1)) if m else None})
    return out


def assign_lecture_minutes(all_chunks: list[dict], part_id: str, lectures: list[dict]):
    """Назначает transcript-чанкам части `lecture` и `lecture_minute` грубой оценкой.

    Позиция чанка в части (его порядковый индекс среди transcript-чанков) трактуется как доля
    пройденного времени. Для видео-лекций (есть длительности) минута = доля × суммарную
    длительность части минус начало лекции. Для книги/кейсов (длительностей нет) секция
    назначается равными весами, минута не считается (None).
    """
    if not lectures:
        return
    tchunks = [c for c in all_chunks if c["part"] == part_id and c["type"] == "transcript"]
    n = len(tchunks)
    if n == 0:
        return

    durations = [lec["duration_min"] for lec in lectures]
    has_durations = all(d is not None for d in durations)

    if has_durations:
        total = sum(durations)
        starts, acc = [], 0
        for d in durations:
            starts.append(acc)
            acc += d
        for k, c in enumerate(tchunks):
            elapsed = ((k + 0.5) / n) * total
            # Reason: первая лекция, чей конец превышает elapsed_min
            i = len(lectures) - 1
            for j in range(len(lectures)):
                if elapsed < starts[j] + durations[j]:
                    i = j
                    break
            c["lecture"] = lectures[i]["title"]
            c["lecture_minute"] = max(0, round(elapsed - starts[i]))
    else:
        for k, c in enumerate(tchunks):
            i = min(int((k + 0.5) / n * len(lectures)), len(lectures) - 1)
            c["lecture"] = lectures[i]["title"]
            c["lecture_minute"] = None


def chunk_text(text: str, part_id: str, part_title: str) -> list[dict]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    buffer = []
    buf_len = 0

    def flush():
        t = " ".join(buffer).strip()
        if len(t) > 100:
            chunks.append({
                "type":           "transcript",
                "part":           part_id,
                "part_title":     part_title,
                "lecture":        "",
                "lecture_minute": None,
                "text":           t,
            })

    for sent in sentences:
        buffer.append(sent)
        buf_len += len(sent) // 4
        if buf_len >= CHUNK_SIZE:
            flush()
            overlap = buffer[-3:]
            buffer = list(overlap)
            buf_len = sum(len(s) // 4 for s in buffer)

    if buffer:
        flush()
    return chunks


def parse_transcript(path: str, part_id: str, part_title: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return chunk_text(text, part_id, part_title)


def _embed_inputs(chunks: list[dict]) -> list[str]:
    """Текст для эмбеддинга: контекстный префикс + тело (contextual retrieval, лёгкий вариант)."""
    return [((c.get("context", "") + "\n" + c["text"]).strip()[:6000] or "пустой фрагмент")
            for c in chunks]


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    from openai import OpenAI
    # Reason: WSL2→OpenAI связь флапает (ConnectTimeout); timeout+max_retries с бэкоффом riding-through.
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30.0, max_retries=5)

    texts = _embed_inputs(chunks)
    print(f"Embedding {len(texts)} chunks via OpenAI {EMBED_MODEL} (batch={BATCH_SIZE})...")

    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embs.extend([d.embedding for d in resp.data])
        print(f"  {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

    return np.array(all_embs, dtype=np.float32)


def build_index():
    print("=== Building Zamesin RAG Index ===")
    all_chunks = []

    for part_id, ann_file, transcript_file, part_title in PARTS:
        ann_path = os.path.join(INDEX_DIR, ann_file)
        tr_path  = os.path.join(INDEX_DIR, transcript_file)

        # --- Аннотации (summary-чанки) ---
        if os.path.exists(ann_path):
            ann_chunks = parse_annotations(ann_path, part_id, part_title)
            for c in ann_chunks:
                c["context"] = make_context(part_title, c["lecture"])
            print(f"  {part_id} annotations: {len(ann_chunks)} lecture chunks")
            all_chunks.extend(ann_chunks)
        else:
            print(f"  WARN: {ann_file} not found")

        if not os.path.exists(tr_path):
            print(f"  WARN: {transcript_file} not found")
            continue

        if part_id in SECTION_PARTS:
            # --- Книга/кейсы: нарезка по секциям-заголовкам, точная атрибуция ---
            with open(tr_path, encoding="utf-8") as f:
                sections = split_sections(f.read())
            sec_chunks_n = 0
            for title, body in sections:
                for c in chunk_text(clean_source_text(body), part_id, part_title):
                    c["lecture"] = title
                    c["lecture_minute"] = None
                    c["context"] = make_context(part_title, title)
                    all_chunks.append(c)
                    sec_chunks_n += 1
            print(f"  {part_id}: {len(sections)} секций → {sec_chunks_n} transcript-чанков (по заголовкам)")
        else:
            # --- Видео-лекции PART1–4: чанкинг + грубая минута по длительностям ---
            tr_chunks = parse_transcript(tr_path, part_id, part_title)
            print(f"  {part_id} transcript: {len(tr_chunks)} text chunks")
            all_chunks.extend(tr_chunks)
            lectures = parse_lecture_meta(ann_path)
            assign_lecture_minutes(all_chunks, part_id, lectures)
            for c in all_chunks:
                if c["part"] == part_id and c["type"] == "transcript":
                    c["context"] = make_context(part_title, c["lecture"])

    # --- Дедуп: убрать из PART5 (книга) кейсы, продублированные в PART6 ---
    all_chunks, dropped = dedup_part5_against_part6(all_chunks)
    print(f"\nDedup PART5↔PART6: удалено {dropped} дублирующих кейс-чанков из книги")

    for i, c in enumerate(all_chunks):
        c["id"] = i

    print(f"Total chunks: {len(all_chunks)}")

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Chunks saved: {CHUNKS_PATH}")

    # Reason: переиспользуем эмбеддинги ТОЛЬКО если хеш embed-входов совпал (защита от stale-reuse
    # при изменении текста чанков). --force-embed принудительно пересобирает.
    inputs = _embed_inputs(all_chunks)
    content_hash = hashlib.sha256("\n\n".join(inputs).encode("utf-8")).hexdigest()
    force = "--force-embed" in sys.argv
    embeddings = None
    if not force and os.path.exists(EMBEDDINGS_PATH) and os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        existing = np.load(EMBEDDINGS_PATH)
        if meta.get("hash") == content_hash and existing.shape[0] == len(all_chunks):
            print(f"Reusing embeddings {existing.shape} — hash совпал, пропускаю re-embed.")
            embeddings = existing

    if embeddings is None:
        embeddings = embed_chunks(all_chunks)
        np.save(EMBEDDINGS_PATH, embeddings)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump({"hash": content_hash, "count": len(all_chunks)}, f)
        print(f"Embeddings saved: {EMBEDDINGS_PATH} — shape: {embeddings.shape}")

    print("\nDone! Запусти бота: python app/telegram_bot.py")


if __name__ == "__main__":
    build_index()
