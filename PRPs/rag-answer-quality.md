name: "RAG Answer Quality — AJTBD Telegram Bot"
description: |
  Поднять качество и достоверность ответов RAG-бота по курсу Замесина: память диалога,
  извлечение намерения из длинных сообщений, честный порог релевантности, голос AJTBD,
  фильтрация источников, мета-вопросы. Источник — INITIAL.md (7 провалов из 2 продакшен-диалогов).

## Purpose
Один проход реализации, чтобы бот перестал вести себя как generic ChatGPT с нерелевантными
цитатами и начал: (а) держать нить диалога, (б) находить релевантные фрагменты на длинных и
разговорных запросах, (в) говорить голосом методологии, (г) не показывать мусорные источники.

## Core Principles
1. **Context is King** — весь нужный контекст ниже, агент его не доищет сам.
2. **Validation Loops** — есть исполняемые тесты (pytest, mock OpenAI) + eval-скрипт на реальном API.
3. **Information Dense** — паттерны и имена из реального кода `app/rag.py`, `app/telegram_bot.py`.
4. **Progressive Success** — сначала чистый рефакторинг ранжирования (тестируемо без сети), потом LLM-слой.
5. **Global rules** — CLAUDE.md: русский UTF-8, не выдумывать метрики, try/except на API.

---

## Goal
Переписать retrieval/answer-конвейер так, чтобы 5 воспроизводимых провалов из диалогов
(INITIAL.md) давали корректные ответы, а 3 рабочих сценария НЕ регрессировали. Конкретно:
память диалога (P1), извлечение вопроса из длинного текста (P2), сырой косинус + порог (P3),
query-rewrite для разговорных запросов (P4), системный промпт в голосе AJTBD (P5), честные
источники по порогу с дедупом (P6), детерминированный обработчик мета-вопросов (P7).

## Why
- Бот построен на материалах Замесина, но в диалогах отвечает шаблонами и цитирует кофе-обжарщиков
  на вопрос про ценность продукта — это подрывает доверие клиента.
- «Скор 0.89» в источниках — это нормализованный гибрид, не косинус; пользователь видит фейковую
  релевантность.
- Самое больное — отсутствие памяти: «разобрать подробнее?» ломает бота, хотя ответ он только что дал.

## What
### Success Criteria
- [ ] **P1** «Ты можешь разобрать подробнее?»/«А еще как?» после ответа → бот продолжает ТУ ЖЕ тему (не теряет нить).
- [ ] **P2** Вставка длинного плана + «Какая здесь ценность продукта?» → retrieval по извлечённому вопросу, источники релевантны теме «ценность продукта», а не случайны.
- [ ] **P3** В источниках показан СЫРОЙ косинус; чанки ниже порога не возвращаются; если ничего не прошло — честный ответ «в материалах нет».
- [ ] **P4** «Как замесин говорил нужно сделать план» → находит релевантные фрагменты (не «нет информации»).
- [ ] **P5** «Улучши этот план» → ответ в терминах AJTBD (работы/ценность/сегменты), без generic-советов «внедрите аналитику».
- [ ] **P6** Источники: только прошедшие порог, дедуп идентичных пар книга/кейсы, честный косинус.
- [ ] **P7** «Сколько всего кейсов?» → «11 кейсов: …»; «Дай кейс 8» → кейс 8 детерминированно.
- [ ] **Регрессии НЕ сломаны:** «Дай кейс 8», «Покажи реальный кейс AJTBD», «О чём книга?» отвечают корректно.
- [ ] Тесты зелёные (`pytest tests/ -v`), линт чистый.

## All Needed Context

### Documentation & References
```yaml
- url: https://platform.openai.com/docs/guides/embeddings
  why: text-embedding-3-small даёт нормализованные векторы → dot product = косинус в [-1,1].
       Для русского релевантные пары обычно ~0.3–0.55, нерелевантные ~0.1–0.25 — основа порога P3.

- url: https://python.langchain.com/docs/how_to/qa_chat_history_how_to/
  section: "create_history_aware_retriever / contextualize_question"
  critical: КАНОНИЧЕСКИЙ паттерн для P1+P2+P4 — один LLM-вызов переписывает (история + новое
            сообщение) в STANDALONE поисковый запрос ДО retrieval. Не копировать LangChain,
            а воспроизвести идею своим вызовом gpt-4o-mini.

- url: https://docs.python-telegram-bot.org/en/v20.0/telegram.ext.callbackcontext.html
  why: context.user_data — per-user dict, ЖИВЁТ В ПАМЯТИ процесса, теряется при рестарте.
       Хранить историю там; ограничить длину (полётный MVP, не БД).

- file: app/rag.py
  why: ВСЯ логика retrieval/answer здесь. cosine_sim уже возвращает СЫРОЙ косинус (оба вектора
       нормализуются). retrieve() смешивает в combined и теряет сырой косинус — это и чиним.

- file: app/telegram_bot.py
  why: message_handler (строки ~236–277) — точка интеграции истории, мета-роутинга, фильтра источников.
       ВАЖНО: модуль на импорте грузит индекс и зовёт OpenAI (build_embeddings_if_needed) —
       НЕ импортировать telegram_bot в юнит-тестах. Тестировать только app/rag.py.

- file: scripts/build_index.py
  why: структура чанка-источника. Поля чанка: type, part, part_title, lecture, text, id.

- docfile: agent_docs/python_conventions.md
  why: pytest на каждую фичу (1 happy + 1 edge + 1 failure), type hints, Google-docstrings,
       max 500 строк/файл, "# Reason:" для неочевидного.

- docfile: agent_docs/rag_pipeline.md
  why: проектные дефолты — top-6, косинус, тримминг контекста до ~6000 символов.
```

### Current state (факты из кода, не угадывать)
```text
Индекс: data/chunks_cache.json (1248 чанков) + data/embeddings_cache.npy (1248, 1536), text-embedding-3-small.
Чат: gpt-4o-mini (OPENAI_MODEL).
Поля чанка: type ∈ {transcript(1202), annotation(46)}, part ∈ {PART1..PART6}, part_title, lecture, text, id.
PART6 «Кейсы»: 11 annotation-чанков с чистыми заголовками "Кейс N — …". PART5 «Книга» 162, и т.д.
cosine_sim(a, embeddings): нормализует оба → возвращает СЫРОЙ косинус (vector_scores) в [-1,1].
retrieve(): combined = alpha*norm(vector)+(1-alpha)*norm(bm25), gate combined>0.01 (пропускает почти всё),
            сырой косинус ТЕРЯЕТСЯ. Показываемая "схожесть" = combined (min-max по корпусу) → фейк.
answer_stream(query, ...): эмбеддит ВЕСЬ query; query целиком уходит в промпт как "Вопрос:".
message_handler: зовёт answer_stream только с текущим text; истории нет (user_data только settings/menu_state).
Тестов НЕТ. requirements.txt: numpy, openai, python-dotenv, python-telegram-bot, httpx, rank-bm25 (pytest нет).
```

### Desired file changes
```text
MODIFY app/rag.py
  - split retrieve → embed_query (сеть) + rank_and_filter (чистая, тестируемая)
  - rank_and_filter возвращает raw_cosine отдельно + порог MIN_COSINE
  - rewrite_query(history, message) — один gpt-4o-mini вызов (P1/P2/P4)
  - try_meta_answer(query, chunks) — детерминированный мета-обработчик (P7), без сети
  - усиленный SYSTEM_PROMPT в голосе AJTBD (P5)
  - answer_stream принимает search_query + history; ретрив по search_query, промпт по original query

MODIFY app/telegram_bot.py
  - история в user_data["history"] (cap последних N), мета-роутинг перед RAG,
    источники по порогу + дедуп + честный косинус (P6)

CREATE tests/test_rag.py
  - юнит-тесты чистых функций (rank_and_filter, try_meta_answer, history-cap) + mock OpenAI для rewrite_query

CREATE tests/__init__.py  (пустой)

CREATE scripts/eval_quality.py
  - прогон провальных+рабочих запросов по реальному API: печатает ответ + raw_cosine источников.
    Двойная роль: калибровка порога MIN_COSINE и приёмочная проверка.

MODIFY requirements.txt  (добавить pytest>=8.0)
MODIFY README.md (раздел про качество ответов / память диалога — кратко)
MODIFY TASK.md (Session Log 2026-06-22)
```

### Known Gotchas
```python
# CRITICAL: cosine_sim уже возвращает СЫРОЙ косинус (оба вектора L2-нормализуются). Не путать
#           с combined-скором. raw_cosine = vector_scores[i].
# CRITICAL: НЕ импортировать app/telegram_bot.py в тестах — на импорте грузится индекс и дёргается OpenAI.
#           Вся тестируемая логика должна жить в app/rag.py и не делать сетевых вызовов на импорте.
# CRITICAL: user_data в python-telegram-bot v20 — in-memory, per-user; теряется при рестарте. MVP ок,
#           но историю ограничить (N=6 сообщений), а контент ассистента обрезать (~600 симв.) — не раздувать промпт.
# GOTCHA: P2 — ВСТАВЛЕННЫЙ план НУЖЕН для ОТВЕТА (юзер просит разобрать СВОЙ план), но НЕ для embedding.
#         Решение: embed = rewrite_query(...) (извлечённый вопрос); в промпт-«Вопрос» уходит original text целиком.
# GOTCHA: rewrite_query НЕ вызывать впустую. Эвристика-skip: нет истории И len(message)<200 И '?' в message
#         → искать по original (экономит латентность; "Дай кейс 8"/"Покажи кейс" не регрессируют).
# GOTCHA: дедуп источников — книга(PART5)/кейсы(PART6) дают идентичные тексты (в диалоге пары 0.76/0.76).
#         Дедуп по первым ~80 символам нормализованного текста.
# GOTCHA: temperature=0 для rewrite_query и try-meta (детерминизм); answer_stream оставить 0.2 + stream=True.
# GOTCHA: MIN_COSINE калибровать ЭМПИРИЧЕСКИ через scripts/eval_quality.py — не хардкодить вслепую.
#         Стартовое значение 0.30; печатать сырые косинусы рабочих и провальных запросов, выбрать порог МЕЖДУ.
```

## Implementation Blueprint

### Pseudocode (app/rag.py)
```python
MIN_COSINE = float(os.getenv("MIN_COSINE", "0.30"))  # калибруется eval_quality.py
HISTORY_MAX_MSGS = 6

# --- P3: чистая, тестируемая (без сети) ---
def rank_and_filter(qvec, chunks, embeddings, bm25, query, part_filter, top_k=6, alpha=0.6):
    vector_scores = cosine_sim(qvec, embeddings)          # СЫРОЙ косинус
    bm25_scores = np.array(bm25.get_scores(query.lower().split()), dtype=np.float32)
    combined = alpha*norm(vector_scores) + (1-alpha)*norm(bm25_scores)   # ранжирование
    if part_filter:
        combined = combined * mask(part_filter)
    order = np.argsort(combined)[::-1][:top_k]
    out = []
    for i in order:
        raw = float(vector_scores[i])
        if raw < MIN_COSINE:        # Reason: честный порог по косинусу, не по combined
            continue
        out.append({**chunks[i], "score": float(combined[i]), "raw_cosine": raw})
    return out                      # может быть пустым → честный "нет в материалах"

def retrieve(query, chunks, embeddings, bm25, part_filter, top_k=6):
    return rank_and_filter(embed_query(query), chunks, embeddings, bm25, query, part_filter, top_k)

# --- P1+P2+P4: один LLM-вызов ---
def rewrite_query(history, message) -> str:
    # эвристика-skip (см. gotcha): нет истории и короткий вопрос → вернуть message как есть
    if not history and len(message) < 200 and "?" in message:
        return message
    # gpt-4o-mini, temperature=0, короткий max_tokens:
    # system: "Ты переформулируешь реплику пользователя в один автономный поисковый запрос
    #   по методологии AJTBD Замесина. Разреши местоимения по истории. Если в сообщении вставлен
    #   длинный текст/план — извлеки суть ВОПРОСА, игнорируя объём вставки. Верни ТОЛЬКО запрос."
    # user: история (последние N) + "Новое сообщение: {message}"
    # try/except → при ошибке вернуть message (graceful fallback)

# --- P7: детерминированно, без сети ---
def try_meta_answer(query, chunks) -> str | None:
    q = query.lower()
    cases = [c for c in chunks if c["part"]=="PART6" and c["type"]=="annotation"]
    if re.search(r"скольк.*кейс|сколько всего кейс", q):
        return f"Всего {len(cases)} кейсов:\n" + "\n".join(c["lecture"] for c in cases)
    if re.search(r"(спис|перечисл|какие).*кейс", q):
        return "Кейсы:\n" + "\n".join(c["lecture"] for c in cases)
    m = re.search(r"кейс\s*№?\s*(\d+)", q)               # "дай кейс 8"
    if m:
        n = m.group(1)
        hit = next((c for c in cases if c["lecture"].startswith(f"Кейс {n} ")
                    or c["lecture"].startswith(f"Кейс {n}—") or c["lecture"].startswith(f"Кейс {n} —")), None)
        return hit["text"] if hit else None
    if re.search(r"(какие|сколько).*част|из чего.*курс", q):
        parts = sorted({(c["part"], c["part_title"]) for c in chunks})
        return "Части курса:\n" + "\n".join(f"{p}: {t}" for p,t in parts)
    return None

# --- P5 + P1: усиленный промпт, история в генерацию ---
SYSTEM_PROMPT = (
  "Ты — Илья Замесин, автор методологии Advanced Jobs To Be Done (AJTBD).\n"
  "Отвечай ТОЛЬКО на основе фрагментов лекций/книги/кейсов ниже.\n"
  "ВСЕГДА применяй понятийный аппарат AJTBD: работы (jobs), граф работ, ценность как "
  "первопричину выбора, сегменты, Consideration Set. Если просят улучшить/разобрать план — "
  "разбирай через эти понятия, а НЕ давай generic-советы («внедрите аналитику», «соберите фидбэк»).\n"
  "Если ответа в материалах нет — скажи прямо, не выдумывай. Русский язык."
)

def answer_stream(query, chunks, embeddings, bm25, part_filter, answer_length, history=None, search_query=None):
    sq = search_query or query
    relevant = retrieve(sq, chunks, embeddings, bm25, part_filter)
    if not relevant:
        yield "В материалах курса нет ответа на этот вопрос."; yield None, []; return
    context = "\n\n---\n\n".join(...)  # как сейчас
    messages = [{"role":"system","content":SYSTEM_PROMPT}]
    if history: messages += history[-HISTORY_MAX_MSGS:]     # P1: контекст диалога в генерацию
    messages.append({"role":"user","content": f"{hint}\n\nФрагменты:\n{context}\n\nВопрос: {query}"})
    # stream=True, temperature=0.2 — как сейчас; yield токены; финально yield None, relevant
```

### Pseudocode (app/telegram_bot.py — message_handler, RAG-ветка)
```python
# после блока навигации:
hist = context.user_data.setdefault("history", [])
# P7: мета-роутинг ДО RAG
meta = try_meta_answer(text, CHUNKS)
if meta:
    await update.message.reply_text(meta)          # без блока источников
    return
hist.append({"role":"user","content": text[:1500]})
search_query = rewrite_query(hist[:-1], text)      # P1/P2/P4 (история без текущего + текущий)
thinking = await update.message.reply_text("🔍 Ищу ответ…")
# answer_stream(..., history=hist[:-1], search_query=search_query)
# собрать full_text + relevant как сейчас
hist.append({"role":"assistant","content": full_text[:600]})
del hist[:-HISTORY_MAX_MSGS]                        # cap
# P6: источники — relevant уже отфильтрован порогом; дедуп + честный косинус
seen=set(); lines=[]
for c in relevant[:4]:
    key = c["text"][:80].lower().strip()
    if key in seen: continue
    seen.add(key)
    lines.append(f"📎 {header} (косинус: {c['raw_cosine']:.2f})\n<i>{preview}…</i>")
if lines: await update.message.reply_html("📚 <b>Источники:</b>\n\n" + "\n\n".join(lines))
```

### Список задач (по порядку)
```yaml
Task 1 — requirements + test scaffold:
  MODIFY requirements.txt: add "pytest>=8.0"
  CREATE tests/__init__.py (пустой)
  Reason: bootstrap тест-инфры (тестов в проекте нет).

Task 2 — P3 рефакторинг ранжирования (чистая функция, без сети):
  MODIFY app/rag.py:
    - ADD MIN_COSINE, HISTORY_MAX_MSGS константы
    - SPLIT retrieve → embed_query + rank_and_filter(qvec, ...) с raw_cosine и порогом
    - PRESERVE сигнатуру retrieve() для обратной совместимости (внутри зовёт rank_and_filter)
  Reason: первым — потому что чисто тестируется без API, фундамент для P6.

Task 3 — P7 мета-обработчик (чистая функция):
  MODIFY app/rag.py: ADD try_meta_answer(query, chunks)

Task 4 — P1/P2/P4 query-rewrite:
  MODIFY app/rag.py: ADD rewrite_query(history, message) с эвристикой-skip и try/except fallback

Task 5 — P5 + P1 генерация:
  MODIFY app/rag.py: усилить SYSTEM_PROMPT; answer_stream принимает history + search_query

Task 6 — интеграция в бот (P1/P6/P7):
  MODIFY app/telegram_bot.py: история в user_data, мета-роутинг, rewrite, источники по порогу+дедуп+косинус

Task 7 — тесты:
  CREATE tests/test_rag.py: rank_and_filter (happy/порог/пусто), try_meta_answer (счёт/кейс N/не-мета),
    rewrite_query (skip-эвристика + mock OpenAI), history-cap

Task 8 — eval/калибровка:
  CREATE scripts/eval_quality.py: прогон провальных+рабочих запросов на реальном API,
    печать raw_cosine → выбрать MIN_COSINE; подставить в .env/константу

Task 9 — docs:
  MODIFY README.md (кратко: память диалога, честный порог, мета-вопросы)
  MODIFY TASK.md (Session Log 2026-06-22)
```

## Validation Loop

### Level 1: Syntax & Style
```bash
cd /home/nikita/code/PlatoIsDead/zamesin_product
python -m py_compile app/rag.py app/telegram_bot.py scripts/eval_quality.py
# опционально, если установлены:
pip install ruff >/dev/null 2>&1; ruff check app/ scripts/ tests/ --fix || true
# Ожидание: компилируется без ошибок.
```

### Level 2: Unit Tests (mock OpenAI — без сети)
```python
# tests/test_rag.py — ключевые кейсы:
# rank_and_filter: фейковые embeddings (np), один чанк с cos>порога, один ниже → возвращается только первый,
#   в нём есть raw_cosine; всё ниже порога → []  (happy + edge + failure)
# try_meta_answer: "сколько всего кейсов?" → "11"; "дай кейс 8" → текст с "Кейс 8"; "что такое ценность?" → None
# rewrite_query: skip-эвристика (нет истории, короткий, "?") → возвращает message без вызова OpenAI (mock не вызван);
#   с историей → OpenAI замокан (monkeypatch _client), возвращает канон-строку, проверить что history попала в messages
# history-cap: список из 10 → после среза HISTORY_MAX_MSGS длина == 6
```
```bash
cd /home/nikita/code/PlatoIsDead/zamesin_product
python -m pytest tests/ -v
# Итерировать до зелёного. НИКОГДА не мокать ради прохождения — чинить логику.
```

### Level 3: Integration / калибровка (реальный API, ключ в .env)
```bash
cd /home/nikita/code/PlatoIsDead/zamesin_product
python scripts/eval_quality.py
# Прогоняет: 5 провальных (подробнее/ценность+план/замесин-план/улучши-план/сколько кейсов)
#            + 3 рабочих (кейс 8 / покажи кейс / о чём книга).
# Печатает ответ + raw_cosine источников. Подобрать MIN_COSINE так, чтобы провальные нашли
# релевантное, а мусор отсёкся; рабочие не сломались. Записать значение в .env (MIN_COSINE=…).
```

## Final validation Checklist
- [ ] `python -m pytest tests/ -v` — зелёный
- [ ] `python -m py_compile app/*.py scripts/*.py` — чисто
- [ ] eval_quality.py: 5 провалов исправлены, 3 рабочих не регрессировали
- [ ] MIN_COSINE откалиброван и записан (не хардкод вслепую)
- [ ] Источники показывают СЫРОЙ косинус, отфильтрованы порогом, дедуплицированы
- [ ] try/except на всех OpenAI-вызовах (rewrite_query fallback на message)
- [ ] README.md и TASK.md обновлены
- [ ] Ни один файл не превысил 500 строк (python_conventions)

## Anti-Patterns to Avoid
- ❌ Не эмбеддить весь длинный текст сообщения (корень P2).
- ❌ Не показывать combined-скор под видом «схожести» (корень P3/P6).
- ❌ Не хардкодить MIN_COSINE без прогона eval (сломает рабочие ответы).
- ❌ Не импортировать telegram_bot в тестах (тянет индекс + OpenAI).
- ❌ Не делать 3 отдельных LLM-вызова для P1/P2/P4 — один rewrite_query решает всё.
- ❌ Не ломать гибридный BM25+vector ранкинг — raw_cosine добавить как отдельное поле.
- ❌ Не выдумывать metrики/цифры в ответах (CLAUDE.md hard rule).

---

## Confidence Score: 8/10
Высокая уверенность: код мал и понятен, паттерны каноничны (history-aware retrieval), мета-данные
чанков чистые (11 кейсов как annotation), логика разбита на тестируемые чистые функции.
Минус 2: (1) MIN_COSINE требует эмпирической калибровки на реальном API — единственный неполностью
детерминированный шаг; (2) тест-инфра создаётся с нуля (pytest не был в проекте), плюс качество
голоса AJTBD (P5) оценивается вручную через eval, не автотестом.
