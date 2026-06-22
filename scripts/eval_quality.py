"""
scripts/eval_quality.py — приёмочный прогон + калибровка MIN_COSINE (реальный API).

Гоняет провальные сценарии из диалогов (INITIAL.md) и рабочие (регрессии) через реальный
конвейер, печатает ответ и СЫРОЙ косинус источников. По выводу выбирается MIN_COSINE так,
чтобы провальные находили релевантное, а мусор отсекался, и рабочие не ломались.

Запуск: python scripts/eval_quality.py   (нужен OPENAI_API_KEY в .env)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import rag  # noqa: E402

# (label, история, сообщение)
SCENARIOS = [
    # --- провальные из диалогов ---
    ("P1 кореференс",
     [{"role": "user", "content": "О чём книга?"},
      {"role": "assistant", "content": "Книга про методологию AJTBD: ценность, работы, сегменты."}],
     "Ты можешь разобрать подробнее?"),
    ("P2 длинная вставка + вопрос",
     [],
     "Что уже сделано: MVP корпоративный портал с чат-ботами, RAG по документу, автотесты, "
     "геймификация, продакшн. Итого 144 часа, 288 000 рублей.\n\nКакая здесь ценность продукта?"),
    ("P4 разговорный про методологию",
     [],
     "Как замесин говорил нужно сделать план"),
    ("P5 улучши план (голос AJTBD)",
     [],
     "Улучши этот план запуска корпоративного обучающего бота"),
    ("P7 сколько кейсов (мета)",
     [],
     "Сколько всего кейсов?"),
    # --- рабочие сценарии (регрессии не должны сломаться) ---
    ("REG кейс 8",
     [],
     "Дай кейс 8"),
    ("REG покажи кейс",
     [],
     "Покажи реальный кейс AJTBD"),
    ("REG о чём книга",
     [],
     "О чём книга?"),
]


def main():
    print(f"MIN_COSINE = {rag.MIN_COSINE}\n")
    rag.build_embeddings_if_needed()
    chunks, embeddings, bm25 = rag.load_index()
    print(f"Индекс: {len(chunks)} чанков\n" + "=" * 70)

    for label, history, message in SCENARIOS:
        print(f"\n### {label}\nСообщение: {message[:90]}")

        # P7: мета-обработчик перехватывает до RAG
        meta = rag.try_meta_answer(message, chunks)
        if meta:
            print(f"[META] {meta[:200]}")
            print("-" * 70)
            continue

        search_query = rag.rewrite_query(history, message)
        print(f"search_query → {search_query[:90]}")

        answer = ""
        relevant = []
        for item in rag.answer_stream(
            query=message, chunks=chunks, embeddings=embeddings, bm25=bm25,
            part_filter=None, answer_length="Стандартно",
            history=history, search_query=search_query,
        ):
            if isinstance(item, tuple):
                _, relevant = item
                break
            answer += item

        print(f"Ответ: {answer[:300]}")
        if relevant:
            print(f"Цитата: {rag.format_citation(relevant[0])}  (косинус {relevant[0]['raw_cosine']:.3f})")
        else:
            print("Цитата: НЕТ (ничего не прошло порог)")
        print("-" * 70)


if __name__ == "__main__":
    main()
