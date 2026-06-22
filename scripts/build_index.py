"""
scripts/build_index.py
Parses markdown lecture files → chunks → embeds via OpenAI text-embedding-3-small → saves cache.
Run once: python scripts/build_index.py
Requires OPENAI_API_KEY in .env
"""

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

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE  = 100

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


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    texts = [c["text"][:6000].strip() or "пустой фрагмент" for c in chunks]
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

        if os.path.exists(ann_path):
            ann_chunks = parse_annotations(ann_path, part_id, part_title)
            print(f"  {part_id} annotations: {len(ann_chunks)} lecture chunks")
            all_chunks.extend(ann_chunks)
        else:
            print(f"  WARN: {ann_file} not found")

        if os.path.exists(tr_path):
            tr_chunks = parse_transcript(tr_path, part_id, part_title)
            print(f"  {part_id} transcript: {len(tr_chunks)} text chunks")
            all_chunks.extend(tr_chunks)
        else:
            print(f"  WARN: {transcript_file} not found")

        # Назначить transcript-чанкам части лекцию + грубую минуту по аннотациям
        if os.path.exists(ann_path):
            lectures = parse_lecture_meta(ann_path)
            assign_lecture_minutes(all_chunks, part_id, lectures)
            tagged = sum(1 for c in all_chunks
                         if c["part"] == part_id and c["type"] == "transcript" and c["lecture"])
            print(f"  {part_id}: размечено {tagged} transcript-чанков "
                  f"({len(lectures)} секций, минуты={'да' if lectures and lectures[0]['duration_min'] else 'нет'})")

    for i, c in enumerate(all_chunks):
        c["id"] = i

    print(f"\nTotal chunks: {len(all_chunks)}")

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Chunks saved: {CHUNKS_PATH}")

    # Reason: текст чанков не изменился → переиспользуем эмбеддинги, если их число совпадает.
    # Пересборка только по флагу --force-embed или при рассинхроне count.
    force = "--force-embed" in sys.argv
    embeddings = None
    if not force and os.path.exists(EMBEDDINGS_PATH):
        existing = np.load(EMBEDDINGS_PATH)
        if existing.shape[0] == len(all_chunks):
            print(f"Reusing existing embeddings {existing.shape} — chunk count matches, skipping re-embed.")
            embeddings = existing
        else:
            print(f"WARN: embeddings rows {existing.shape[0]} != chunks {len(all_chunks)} → пере-эмбеддинг.")

    if embeddings is None:
        embeddings = embed_chunks(all_chunks)
        np.save(EMBEDDINGS_PATH, embeddings)
        print(f"Embeddings saved: {EMBEDDINGS_PATH} — shape: {embeddings.shape}")

    print("\nDone! Запусти бота: python app/telegram_bot.py")


if __name__ == "__main__":
    build_index()
