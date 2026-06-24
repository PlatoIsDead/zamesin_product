"""
app/rag.py
Embeddings: OpenAI text-embedding-3-small
Chat:       OpenAI gpt-4o-mini

Конвейер качества ответов (см. PRPs/rag-answer-quality.md):
- rank_and_filter — чистое ранжирование (BM25+vector) + ОТСЕЧКА по сырому косинусу (P3)
- rewrite_query   — один LLM-вызов: память диалога + извлечение вопроса + переформулировка (P1/P2/P4)
- try_meta_answer — детерминированные мета-ответы про структуру базы, без сети (P7)
- SYSTEM_PROMPT   — голос методологии AJTBD (P5)
"""
import json
import os
import re
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

DATA_DIR        = os.path.join(os.path.dirname(__file__), "..", "data")
CHUNKS_PATH     = os.path.join(DATA_DIR, "chunks_cache.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings_cache.npy")

OPENAI_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBED_MODEL   = "text-embedding-3-small"

# Reason: абсолютный порог по СЫРОМУ косинусу (не по combined-скору) — отсекает нерелевантные
# чанки, которые раньше всегда проходили gate `combined > 0.01`. Калибруется scripts/eval_quality.py.
MIN_COSINE = float(os.getenv("MIN_COSINE", "0.30"))
# Reason: сколько последних сообщений диалога держим в промпте — не раздувать контекст.
HISTORY_MAX_MSGS = 6
# Reason: константа RRF (Reciprocal Rank Fusion), стандарт из Cormack 2009 / Elastic.
RRF_K = 60

SYSTEM_PROMPT = (
    "Ты — Илья Замесин, автор методологии Advanced Jobs To Be Done (AJTBD).\n"
    "Отвечай ТОЛЬКО на основе фрагментов лекций, книги и кейсов, приведённых ниже.\n"
    "ВСЕГДА применяй понятийный аппарат AJTBD: работы (jobs), граф работ, ценность как "
    "первопричину выбора, сегменты, Consideration Set. Если просят улучшить или разобрать "
    "план/продукт — разбирай его именно через эти понятия, а НЕ давай дженерик-советы "
    "вроде «внедрите аналитику», «соберите обратную связь», «добавьте геймификацию».\n"
    "Если ответа в материалах нет — скажи об этом прямо, ничего не выдумывай.\n"
    "Отвечай на русском языке, чётко и по делу."
)

# Reason: query-rewrite превращает (история + новое сообщение) в один автономный поисковый
# запрос — решает P1 (кореференс), P2 (извлечение вопроса из длинной вставки), P4 (разговорный → термины).
REWRITE_SYSTEM_PROMPT = (
    "Ты помогаешь поисковой системе по курсу AJTBD Ильи Замесина.\n"
    "Переформулируй последнюю реплику пользователя в ОДИН короткий автономный поисковый запрос.\n"
    "Правила:\n"
    "- Разрешай местоимения и отсылки («подробнее», «а ещё», «это») по истории диалога.\n"
    "- Если в сообщении вставлен длинный текст (план, документ) — извлеки суть ВОПРОСА "
    "пользователя, игнорируя объём вставки.\n"
    "- Переводи разговорные формулировки в термины методологии AJTBD.\n"
    "- Верни ТОЛЬКО текст запроса, без кавычек и пояснений."
)

LENGTH_HINT = {
    "Коротко":    "Ответ — 2-3 предложения.",
    "Стандартно": "Ответ — до 150 слов.",
    "Подробно":   "Развёрнутый ответ со всеми деталями.",
}
LENGTH_TOKENS = {"Коротко": 150, "Стандартно": 400, "Подробно": 800}


def _client() -> OpenAI:
    """Создаёт OpenAI-клиент из OPENAI_API_KEY."""
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_index():
    """Загружает чанки, эмбеддинги и строит BM25-индекс."""
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(EMBEDDINGS_PATH)
    # Reason: BM25 индексирует контекст (имена лекций/кейсов) + тело → лучше exact-match.
    # .get("context","") безопасно для индекса, собранного до добавления поля context.
    tokenized = [((c.get("context", "") + " " + c["text"]).lower().split()) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return chunks, embeddings, bm25


def build_embeddings_if_needed():
    """Собирает embeddings_cache.npy через OpenAI, если его нет или размерность не совпадает."""
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


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Косинусная близость вектора a ко всем строкам матрицы b. Возвращает СЫРОЙ косинус в [-1, 1]."""
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return b @ a


def embed_query(text: str) -> np.ndarray:
    """Эмбеддит строку запроса через OpenAI (сетевой вызов)."""
    resp = _client().embeddings.create(model=EMBED_MODEL, input=[text.strip()])
    return np.array(resp.data[0].embedding, dtype=np.float32)


def _ranks(scores: np.ndarray) -> np.ndarray:
    """Ранги документов при сортировке по убыванию скора (0 = лучший)."""
    order = np.argsort(scores)[::-1]
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(len(scores))
    return ranks


def rank_and_filter(qvec, chunks, embeddings, bm25, query, part_filter, top_k=6):
    """Чистое ранжирование БЕЗ сетевых вызовов — тестируемо.

    Фьюжн через RRF (Reciprocal Rank Fusion): score = Σ 1/(k + rank_i). RRF фьюзит РАНГИ
    dense- и BM25-ретриверов, а не несравнимые сырые скоры (косинус vs BM25). ОТСЕКАЕТ
    результаты по сырому косинусу (MIN_COSINE) — порог не зависит от метода фьюжна.

    Args:
        qvec: вектор запроса (уже посчитанный embed_query).
        chunks, embeddings, bm25: индекс из load_index().
        query: текст запроса (для BM25-токенизации).
        part_filter: код части ("PART3") или None.
        top_k: сколько кандидатов вернуть.

    Returns:
        Список чанков с полями score (RRF, для отладки) и raw_cosine (для порога/показа).
        Может быть пустым, если ничего не прошло порог.
    """
    vector_scores = cosine_sim(qvec, embeddings)  # СЫРОЙ косинус (для порога/показа)
    bm25_scores = np.array(bm25.get_scores(query.lower().split()), dtype=np.float32)

    rrf = 1.0 / (RRF_K + _ranks(vector_scores)) + 1.0 / (RRF_K + _ranks(bm25_scores))

    if part_filter:
        mask = np.array([1.0 if c["part"] == part_filter else 0.0 for c in chunks])
        rrf = rrf * mask

    order = np.argsort(rrf)[::-1][:top_k]
    out = []
    for i in order:
        raw = float(vector_scores[i])
        # Reason: честный порог по косинусу, а не по RRF-скору
        if raw < MIN_COSINE:
            continue
        out.append({**chunks[i], "score": float(rrf[i]), "raw_cosine": raw})
    return out


def retrieve(query, chunks, embeddings, bm25, part_filter, top_k=6):
    """Эмбеддит запрос и возвращает релевантные чанки (обёртка над rank_and_filter)."""
    qvec = embed_query(query)
    return rank_and_filter(qvec, chunks, embeddings, bm25, query, part_filter, top_k=top_k)


def rewrite_query(history, message: str) -> str:
    """Переформулирует реплику в автономный поисковый запрос (P1/P2/P4).

    Args:
        history: список сообщений [{"role","content"}, ...] БЕЗ текущего сообщения.
        message: текущая реплика пользователя.

    Returns:
        Строка поискового запроса. При ошибке/skip — возвращает исходное сообщение.
    """
    # Reason: короткий ОДНОСТРОЧНЫЙ автономный вопрос без истории переписывать незачем —
    # экономим вызов и не ломаем рабочие сценарии («Дай кейс 8», «Покажи реальный кейс AJTBD»).
    # Многострочное сообщение почти всегда содержит вставку (план/документ) → нужен rewrite (P2).
    if not history and len(message) < 200 and "?" in message and "\n" not in message:
        return message

    hist_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in history[-HISTORY_MAX_MSGS:]
    )
    user_content = (
        (f"История диалога:\n{hist_text}\n\n" if hist_text else "")
        + f"Новое сообщение пользователя:\n{message}"
    )
    try:
        resp = _client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=120,
            temperature=0,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        return rewritten or message
    except Exception:
        # Reason: rewrite — улучшение, не блокер; при сбое API ищем по исходному сообщению
        return message


def try_meta_answer(query: str, chunks) -> str | None:
    """Детерминированные ответы на мета-вопросы о структуре базы (P7), без сети.

    Returns:
        Текст ответа, либо None если запрос не мета (тогда идём в обычный RAG).
    """
    q = query.lower()
    # Реальное число кейсов = уникальные заголовки PART6-транскриптов (в порядке файла).
    # Reason: аннотации (type=annotation) описывают только первые 11 кейсов, а сам
    # корпус кейсов вырос до 25 — считаем по содержимому, а не по отставшим summary.
    case_titles = list(dict.fromkeys(
        c["lecture"] for c in chunks
        if c["part"] == "PART6" and c["type"] == "transcript" and c.get("lecture")
    ))
    # Курируемые аннотации «Кейс N — …» — есть только для первых 11, дают краткий ответ.
    annotations = [c for c in chunks if c["part"] == "PART6" and c["type"] == "annotation"]

    if re.search(r"скольк\w*.*кейс", q):
        listing = "\n".join(f"{i}. {t}" for i, t in enumerate(case_titles, 1))
        return f"Всего {len(case_titles)} кейсов:\n{listing}"

    if re.search(r"(спис\w*|перечисл\w*|какие)\s*.*кейс", q):
        listing = "\n".join(f"{i}. {t}" for i, t in enumerate(case_titles, 1))
        return f"Кейсы курса:\n{listing}"

    m = re.search(r"кейс\s*№?\s*(\d+)", q)
    if m:
        n = int(m.group(1))
        hit = next(
            (c for c in annotations if c["lecture"].startswith(f"Кейс {n} ")
             or c["lecture"].startswith(f"Кейс {n}—")
             or c["lecture"].startswith(f"Кейс {n} —")),
            None,
        )
        if hit:
            return hit["text"]
        if 1 <= n <= len(case_titles):
            # Кейс есть в базе, но без курируемой аннотации — отдаём заголовок.
            return f"Кейс {n}: {case_titles[n - 1]}"
        # Reason: явный запрос «кейс N», но такого нет — честно говорим, а не отдаём в RAG мусор
        return f"Кейса №{n} в базе нет. Всего кейсов: {len(case_titles)}."

    if re.search(r"(какие|сколько)\s*.*част", q) or re.search(r"из чего.*курс", q):
        parts = sorted({(c["part"], c["part_title"]) for c in chunks})
        listing = "\n".join(f"{p}: {t}" for p, t in parts)
        return f"Части курса:\n{listing}"

    return None


def format_citation(chunk) -> str:
    """Одна строка-цитата для верхнего источника: где смотреть/читать.

    PART1–4 (видео-лекции): лекция + примерная минута старта.
    PART5 (книга) / PART6 (кейсы): глава/кейс без минуты.
    Фолбэк (нет лекции): только часть курса.
    """
    part = chunk.get("part", "")
    part_title = chunk.get("part_title", "")
    lecture = chunk.get("lecture", "")
    minute = chunk.get("lecture_minute")

    if part == "PART5":
        return f"📖 Книга «Как делать продукт»" + (f" · {lecture}" if lecture else "")
    if part == "PART6":
        return f"📁 {lecture}" if lecture else "📁 Кейсы AJTBD"

    # PART1–4
    if lecture:
        base = f"📺 {part_title} · {lecture}"
        if minute is not None:
            base += f" — смотреть с ~{minute} мин"
        return base
    return f"📚 {part_title}" if part_title else ""


def answer_stream(query, chunks, embeddings, bm25, part_filter, answer_length,
                  history=None, search_query=None):
    """Генератор: стримит токены ответа, в конце выдаёт (None, relevant) как сентинел.

    Args:
        query: исходное сообщение пользователя (уходит в промпт как «Вопрос»).
        search_query: запрос для retrieval (rewrite_query); если None — используется query.
        history: список сообщений диалога для контекста генерации (P1).
    """
    sq = search_query or query
    relevant = retrieve(sq, chunks, embeddings, bm25, part_filter)

    if not relevant:
        yield "В материалах курса нет ответа на этот вопрос."
        yield None, []
        return

    context = "\n\n---\n\n".join(
        f"[{c['part_title']} | {c.get('lecture', '')}]\n{c['text'][:800]}"
        for c in relevant
    )

    if isinstance(answer_length, int):
        hint = f"Ответ — не более {answer_length} токенов."
        max_tok = answer_length
    else:
        hint = LENGTH_HINT.get(answer_length, "")
        max_tok = LENGTH_TOKENS.get(answer_length, 400)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages += history[-HISTORY_MAX_MSGS:]  # P1: контекст диалога в генерацию
    messages.append({
        "role": "user",
        "content": (
            f"{hint}\n\n"
            f"Фрагменты лекций:\n{context}\n\n"
            f"Вопрос: {query}"
        ),
    })

    client = _client()
    stream = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        max_tokens=max_tok,
        temperature=0.2,
        stream=True,
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token

    yield None, relevant
