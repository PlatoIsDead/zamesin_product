"""
scripts/build_index.py
Parses markdown lecture files → chunks → embeds via sentence-transformers → saves cache.
Run once: python scripts/build_index.py
"""

import json
import os
import re
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DATA_DIR        = os.path.join(os.path.dirname(__file__), "..", "data")
INDEX_DIR       = os.path.join(DATA_DIR, "zamezin_indexed")
CHUNKS_PATH     = os.path.join(DATA_DIR, "chunks_cache.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings_cache.npy")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
BATCH_SIZE  = 64

PARTS = [
    ("PART1", "PART1_annotations.md", "PART1_osnovy.md",      "Основы AJTBD"),
    ("PART2", "PART2_annotations.md", "PART2_cennost.md",     "Ценность продукта"),
    ("PART3", "PART3_annotations.md", "PART3_zapusk.md",      "Запуск и сегменты"),
    ("PART4", "PART4_annotations.md", "PART4_strategiya.md",  "Стратегия"),
]

CHUNK_SIZE = 600  # approx tokens (chars / 4)


def parse_annotations(path: str, part_id: str, part_title: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        content = f.read()

    sections = re.split(r"(?=^## Лекция)", content, flags=re.MULTILINE)
    chunks = []
    for sec in sections:
        sec = sec.strip()
        if not sec.startswith("## Лекция"):
            continue
        title = sec.splitlines()[0].lstrip("#").strip()
        chunks.append({
            "type":       "annotation",
            "part":       part_id,
            "part_title": part_title,
            "lecture":    title,
            "text":       sec,
        })
    return chunks


def chunk_text(text: str, part_id: str, part_title: str) -> list[dict]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    buffer = []
    buf_len = 0

    def flush():
        t = " ".join(buffer).strip()
        if len(t) > 100:
            chunks.append({
                "type":       "transcript",
                "part":       part_id,
                "part_title": part_title,
                "lecture":    "",
                "text":       t,
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
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    # Use first 512 chars for embedding (model max tokens)
    texts = [c["text"][:512].strip() or "пустой фрагмент" for c in chunks]
    print(f"Embedding {len(texts)} chunks (batch={BATCH_SIZE})...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


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

    for i, c in enumerate(all_chunks):
        c["id"] = i

    print(f"\nTotal chunks: {len(all_chunks)}")

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Chunks saved: {CHUNKS_PATH}")

    embeddings = embed_chunks(all_chunks)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Embeddings saved: {EMBEDDINGS_PATH} — shape: {embeddings.shape}")
    print("\nDone! Run: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    build_index()
