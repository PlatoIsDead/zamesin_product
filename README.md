# AJTBD RAG Bot

Telegram-бот для ответов на вопросы по курсу **Advanced Jobs To Be Done** Ильи Замезина. RAG по 6 частям: лекции PART1–4, книга PART5, кейсы PART6. Бот отвечает строго по материалам курса.

## Структура проекта

```
zamesin_product/
├── app/
│   ├── telegram_bot.py     # Telegram-интерфейс
│   └── rag.py              # Retrieval + generation
├── data/
│   ├── chunks_cache.json   # 1248 чанков (все 6 частей)
│   ├── embeddings_cache.npy  # OpenAI text-embedding-3-small (1248, 1536)
│   ├── book_zamesin.txt    # Книга AJTBD (PART5)
│   ├── cases_zamesin.txt   # Кейсы (PART6)
│   └── zamezin_indexed/    # Лекции PART1–4 в markdown
├── scripts/
│   └── build_index.py      # Пересборка индекса
├── Dockerfile
└── requirements.txt
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
- Chat: `gpt-4o-mini`

## Части курса

| Код   | Содержание                   | Чанков |
|-------|------------------------------|--------|
| PART1 | Основы AJTBD                 | ~240   |
| PART2 | Ценность продукта            | ~270   |
| PART3 | Запуск и сегменты            | ~240   |
| PART4 | Стратегия                    | ~230   |
| PART5 | Книга AJTBD                  | 162    |
| PART6 | Кейсы AJTBD                  | 106    |

## Retrieval: гарантированные источники для стартеров

Стандартный retrieval (top-5 по косинусному сходству) работает хорошо для свободных вопросов, но плохо справляется с conversation starters — запросами вида «Покажи реальный кейс AJTBD». Проблема в том, что слово «кейс» встречается во всех 6 частях, и PART6 (где реальные кейсы) может не попасть в топ-5 по score.

**Пример сбоя:** запрос «Покажи реальный кейс AJTBD» вернул чанки из PART1, PART2, PART5 — но не из PART6. Scores были низкими (0.41–0.46), и бот ответил что «конкретных кейсов нет».

**Решение:** для каждого conversation starter в `STARTER_BOOSTS` (`rag.py`) прописаны «домашние» части. При совпадении запроса со стартером retrieve дополнительно гарантирует 2 лучших чанка из этих частей поверх обычного top-5.

```python
STARTER_BOOSTS = {
    "Что такое граф работ?":          ["PART1", "PART5"],
    "Как провести AJTBD-интервью?":   ["PART1"],
    "Как создать ценность продукта?": ["PART2"],
    "Покажи реальный кейс AJTBD":     ["PART6"],
}
```

Boost применяется только когда пользователь не выбрал ручной фильтр по части.

## Пересборка индекса

Нужна если изменились исходные тексты в `data/`:

```bash
python scripts/build_index.py
# Затем скопировать data/ на сервер и пересобрать образ
```
