# AJTBD RAG Bot

Telegram-бот для ответов на вопросы по курсу **Advanced Jobs To Be Done** Ильи Замезина. RAG по 6 частям: лекции PART1–4, книга PART5, кейсы PART6. Бот отвечает строго по материалам курса.

## Структура проекта

```
zamesin_product/
├── app/
│   ├── telegram_bot.py     # Telegram-интерфейс
│   └── rag.py              # Retrieval + generation
├── data/
│   ├── chunks_cache.json   # 1159 чанков (все 6 частей; book/cases — посекционно, дедуп)
│   ├── embeddings_cache.npy  # OpenAI text-embedding-3-small (1159, 1536); .meta.json — хеш для reuse
│   ├── book_zamesin.txt    # Книга AJTBD (PART5)
│   ├── cases_zamesin.txt   # Кейсы (PART6)
│   └── zamezin_indexed/    # Лекции PART1–4 в markdown
├── scripts/
│   ├── build_index.py      # Пересборка индекса
│   └── eval_quality.py     # Приёмка + калибровка MIN_COSINE (реальный API)
├── tests/
│   └── test_rag.py         # Юнит-тесты конвейера (mock OpenAI)
├── Dockerfile
└── requirements.txt
```

## Тесты

```bash
python -m pytest tests/ -v
```

## Запуск локально

```bash
cp .env.example .env  # вставить ключи
python scripts/build_index.py  # только если нужно пересобрать индекс
python app/telegram_bot.py
```

## Production

Сервер: Timeweb Cloud VPS `147.45.137.205`, Ubuntu 24.04, 2 vCPU / 2 GB RAM.

```bash
# Скопировать файлы
scp -r app/ data/ Dockerfile requirements.txt root@147.45.137.205:/root/zamesin_product/

# Собрать и запустить
ssh root@147.45.137.205 "cd /root/zamesin_product && docker build -t zamesin-bot . && docker rm -f zamesin-bot; docker run -d --name zamesin-bot --restart unless-stopped --env-file /root/zamesin_product/.env zamesin-bot"

# Логи
ssh root@147.45.137.205 "docker logs zamesin-bot --tail 50"
```

> **Важно:** docker-compose v1 сломан на этом сервере. Использовать только plain `docker`.

## Переменные окружения

```
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
TELEGRAM_BOT_TOKEN=...
```

## Модели

- Embeddings: `text-embedding-3-small` (1536 dim)
- Chat (ответ + переформулировка запроса): `gpt-4o` (`OPENAI_MODEL`)

## Части курса

| Код   | Содержание                   | Чанков |
|-------|------------------------------|--------|
| PART1 | Основы AJTBD                 | 262    |
| PART2 | Ценность продукта            | 121    |
| PART3 | Запуск и сегменты            | 331    |
| PART4 | Стратегия                    | 266    |
| PART5 | Книга AJTBD (главы+гайды, без дублей кейсов) | 70 |
| PART6 | Кейсы AJTBD (25 кейсов)       | 109    |

> Всего 1159 чанков. PART6 содержит **25** кейсов (раньше book дублировал их → дедуп).

## Качество ответов (PRPs/rag-answer-quality.md)

Конвейер ответов усилен по 7 проблемам, выявленным в реальных диалогах:

- **Память диалога (P1).** История переписки хранится в `user_data["history"]` (последние
  `HISTORY_MAX_MSGS` сообщений) и передаётся в генерацию. Кореференс («разобрать подробнее?»,
  «а ещё?») разрешается через `rewrite_query`.
- **Извлечение вопроса из вставки (P2).** Длинные/многострочные сообщения (вставленный план,
  документ) проходят через `rewrite_query` — для retrieval эмбеддится извлечённый ВОПРОС, а в
  промпт-«Вопрос» уходит исходный текст целиком.
- **Гибридный поиск с RRF (P3).** `rank_and_filter` фьюзит dense (косинус) и BM25 через
  **Reciprocal Rank Fusion** (`score = Σ 1/(k+rank)`, k=60) — фьюжн по РАНГАМ, а не по
  несравнимым сырым скорам. СЫРОЙ косинус хранится отдельно и используется для отсечки
  ниже `MIN_COSINE` (0.30, откалибровано `scripts/eval_quality.py`) и для показа. Если ничего
  не прошло порог — честный ответ «в материалах нет».
- **Контекстный префикс (contextual retrieval, лёгкий).** При сборке индекса к каждому чанку
  добавляется `context` (`[Часть · Лекция/Глава/Кейс]`); эмбеддинг и BM25 строятся по
  `context + text` (лучше exact-match на именах: «граф работ», «PLEADA»), а показ/LLM — по
  чистому `text`.
- **Точная атрибуция book/cases + дедуп.** `book_zamesin.txt` содержал все кейсы (дубль
  `cases_zamesin.txt`). Теперь PART5/PART6 режутся на секции по заголовкам, scraping-мусор
  чистится, а PART5-чанки, дублирующие кейсы PART6, удаляются (`dedup_part5_against_part6`).
- **Цитата вместо блока «Источники» (P6).** В конец ответа добавляется ОДНА строка по верхнему
  источнику (`format_citation`): для видео-лекций (PART1–4) — `📺 Часть · Лекция — смотреть с
  ~N мин` (минута оценивается грубо по объёму текста, таймкодов в транскриптах нет); для книги
  (PART5) — `📖 Книга · Глава`; для кейсов (PART6) — `📁 Кейс`. Метаданные лекции/минуты
  проставляются при сборке индекса (`scripts/build_index.py` по `PART*_annotations.md`).
- **Query rewrite (P4).** Разговорные запросы («как замесин говорил…») переформулируются в
  термины методологии перед поиском. Один LLM-вызов `gpt-4o-mini`; короткие однострочные
  автономные вопросы его пропускают (экономия).
- **Голос AJTBD (P5).** `SYSTEM_PROMPT` заставляет разбирать запрос через понятия методологии
  (работы, граф работ, ценность как первопричину, сегменты), без дженерик-советов.
- **Мета-вопросы (P7).** `try_meta_answer` детерминированно (без сети) отвечает на «сколько
  кейсов?», «список кейсов», «дай кейс N», «какие части курса» по метаданным чанков.

Калибровка/приёмка: `python scripts/eval_quality.py` — прогон провальных и рабочих сценариев
на реальном API с печатью сырых косинусов.

## Пересборка индекса

Нужна если изменились исходные тексты в `data/`:

```bash
python scripts/build_index.py
# Затем скопировать data/ на сервер и пересобрать образ
```
