# Вертикаль — RAG Бот по стандартам

Минимальный RAG-бот для ответов на вопросы по документу стандартов сети апарт-отелей «Вертикаль».

## Структура проекта

```
vertical_standards/
├── app/
│   ├── rag.py              # Retrieval + generation логика
│   └── streamlit_app.py    # UI
├── data/
│   ├── standards.docx      # Исходный документ (положить сюда)
│   ├── chunks_cache.json   # Генерируется автоматически
│   └── embeddings_cache.npy
├── scripts/
│   └── build_index.py      # Запустить один раз для индексации
├── .env.example
├── .gitignore
└── requirements.txt
```

## Установка

```bash
cd vertical_standards
pip install -r requirements.txt
```

## Настройка

```bash
cp .env.example .env
# Открыть .env и вставить OPENAI_API_KEY
```

## Запуск

**Шаг 1 — индексация документа (один раз):**
```bash
python scripts/build_index.py
```
Это создаст `data/chunks_cache.json` и `data/embeddings_cache.npy`.

**Шаг 2 — запуск UI:**
```bash
streamlit run app/streamlit_app.py
```

Откроется браузер на `http://localhost:8501`

## UI

- **Раздел** — фильтрует поиск по конкретному разделу документа или ищет по всему
- **Длина ответа** — Коротко / Стандартно / Подробно
- Внизу ответа — источники (заголовки из документа)

## Разделы

| Код    | Раздел                          |
|--------|---------------------------------|
| VA.ADM | Административная политика       |
| VA.ADM.3 | Чрезвычайные ситуации         |
| VA.HR  | Управление персоналом           |
| VA.FO  | Служба приёма и размещения      |
| VA.OW  | Работа с собственниками         |
| VA.SD  | Отдел продаж                    |
| VA.MD  | Маркетинг и продвижение         |
| VA.HSK | Хозяйственная служба            |

## Модели

- Embeddings: `text-embedding-3-small` (~$0.002 за индексацию)
- Chat: `gpt-4o-mini` (~$0.001 за запрос)
