"""
app/rag.py
Embeddings: OpenAI text-embedding-3-small
Chat:       OpenAI gpt-4o-mini
"""
import json
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DATA_DIR        = os.path.join(os.path.dirname(__file__), "..", "data")
CHUNKS_PATH     = os.path.join(DATA_DIR, "chunks_cache.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings_cache.npy")

OPENAI_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL   = "text-embedding-3-small"

SYSTEM_PROMPT = """Ты — ассистент по курсу Advanced Jobs To Be Done (AJTBD) Ильи Замезина.
Отвечай ТОЛЬКО на основе предоставленных фрагментов лекций.
Отвечай на русском языке, чётко и по делу.
Если ответа в материалах нет — скажи об этом прямо.
Не придумывай информацию."""

LENGTH_HINT = {
    "Коротко":    "Ответ — 2-3 предложения.",
    "Стандартно": "Ответ — до 150 слов.",
    "Подробно":   "Развёрнутый ответ со всеми деталями.",
}
LENGTH_TOKENS = {"Коротко": 150, "Стандартно": 400, "Подробно": 800}


def _client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_index():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(EMBEDDINGS_PATH)
    return chunks, embeddings


def build_embeddings_if_needed():
    """Build embeddings_cache.npy using OpenAI if it doesn't exist or dims don't match."""
    if os.path.exists(EMBEDDINGS_PATH):
        emb = np.load(EMBEDDINGS_PATH)
        if emb.shape[1] == 1536:  # text-embedding-3-small dimension
            return
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    client = _client()
    texts = [c["text"][:6000] for c in chunks]
    print(f"Building embeddings for {len(texts)} chunks via OpenAI...")
    all_embs = []
    batch = 100
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i+batch])
        all_embs.extend([d.embedding for d in resp.data])
        print(f"  {min(i+batch, len(texts))}/{len(texts)}")
    np.save(EMBEDDINGS_PATH, np.array(all_embs, dtype=np.float32))
    print("Embeddings built and saved.")


def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return b @ a


def embed_query(text: str) -> np.ndarray:
    resp = _client().embeddings.create(model=EMBED_MODEL, input=[text.strip()])
    return np.array(resp.data[0].embedding, dtype=np.float32)


def retrieve(query, chunks, embeddings, part_filter, top_k=5):
    qvec = embed_query(query)
    scores = cosine_sim(qvec, embeddings)

    if part_filter and part_filter != "Все":
        mask = np.array([1.0 if c["part"] == part_filter else 0.0 for c in chunks])
        scores = scores * mask

    top = np.argsort(scores)[::-1][:top_k]
    return [{**chunks[i], "score": float(scores[i])} for i in top if scores[i] > 0.01]


def answer_stream(query, chunks, embeddings, part_filter, answer_length):
    """Generator: yields text tokens, then (None, relevant) as final sentinel."""
    relevant = retrieve(query, chunks, embeddings, part_filter)

    if not relevant:
        yield "Не найдено релевантных фрагментов."
        yield None, []
        return

    context = "\n\n---\n\n".join(
        f"[{c['part_title']} | {c.get('lecture', '')}]\n{c['text'][:800]}"
        for c in relevant
    )

    client = _client()
    stream = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": (
                f"{LENGTH_HINT.get(answer_length, '')}\n\n"
                f"Фрагменты лекций:\n{context}\n\n"
                f"Вопрос: {query}"
            )},
        ],
        max_tokens=LENGTH_TOKENS.get(answer_length, 400),
        temperature=0.2,
        stream=True,
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token

    yield None, relevant
