"""
app/telegram_bot.py — Telegram interface for Zamesin AJTBD RAG bot
"""
import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from rag import load_index, build_embeddings_if_needed, answer_stream

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Index — loaded once at startup
# ---------------------------------------------------------------------------
logger.info("Loading RAG index…")
build_embeddings_if_needed()
CHUNKS, EMBEDDINGS, BM25 = load_index()
logger.info("Index ready: %d chunks", len(CHUNKS))

# ---------------------------------------------------------------------------
# Button constants
# ---------------------------------------------------------------------------
BTN_SETTINGS   = "⚙️ Настройки"
BTN_BACK       = "← Назад"
BTN_LENGTH     = "📏 Длина ответа"
BTN_PART       = "📚 Часть курса"
BTN_LEN_CUSTOM = "✏️ Свой лимит токенов"

PART_BUTTONS = {
    "Все части":         "all",
    "PART1 — Основы":    "PART1",
    "PART2 — Ценность":  "PART2",
    "PART3 — Запуск":    "PART3",
    "PART4 — Стратегия": "PART4",
    "PART5 — Книга":     "PART5",
    "PART6 — Кейсы":     "PART6",
}
LENGTH_BUTTONS = {"Коротко", "Стандартно", "Подробно"}

# ---------------------------------------------------------------------------
# Parts metadata
# ---------------------------------------------------------------------------
PARTS = {
    "all":   (None,    "Все части"),
    "PART1": ("PART1", "PART1 — Основы"),
    "PART2": ("PART2", "PART2 — Ценность"),
    "PART3": ("PART3", "PART3 — Запуск"),
    "PART4": ("PART4", "PART4 — Стратегия"),
    "PART5": ("PART5", "PART5 — Книга"),
    "PART6": ("PART6", "PART6 — Кейсы"),
}

DEFAULT_SETTINGS = {"part": "all", "length": "Подробно"}

# ---------------------------------------------------------------------------
# Settings & state helpers
# ---------------------------------------------------------------------------
def get_settings(context: ContextTypes.DEFAULT_TYPE) -> dict:
    if "settings" not in context.user_data:
        context.user_data["settings"] = dict(DEFAULT_SETTINGS)
    return context.user_data["settings"]


def get_menu_state(context: ContextTypes.DEFAULT_TYPE) -> str:
    return context.user_data.get("menu_state", "main")


def set_menu_state(context: ContextTypes.DEFAULT_TYPE, state: str):
    context.user_data["menu_state"] = state


# ---------------------------------------------------------------------------
# Keyboard builders
# ---------------------------------------------------------------------------
def _kb_main() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            ["Что такое граф работ?", "Как провести AJTBD-интервью?"],
            ["Как создать ценность продукта?", "Покажи реальный кейс AJTBD"],
            [BTN_SETTINGS],
        ],
        resize_keyboard=True,
        input_field_placeholder="Задай вопрос или выбери тему...",
    )


def _kb_settings() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [[BTN_LENGTH, BTN_PART], [BTN_BACK]],
        resize_keyboard=True,
    )


def _kb_length() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [["Коротко", "Стандартно", "Подробно"], [BTN_LEN_CUSTOM], [BTN_BACK]],
        resize_keyboard=True,
    )


def _kb_part() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            ["Все части"],
            ["PART1 — Основы", "PART2 — Ценность"],
            ["PART3 — Запуск", "PART4 — Стратегия"],
            ["PART5 — Книга", "PART6 — Кейсы"],
            [BTN_BACK],
        ],
        resize_keyboard=True,
    )


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    set_menu_state(context, "main")
    await update.message.reply_html(
        "👋 <b>Привет!</b> Я ассистент по курсу <b>Advanced Jobs To Be Done</b> Ильи Замезина.\n\n"
        "Задавай вопросы по курсу — отвечаю строго по материалам лекций, книги и кейсов.\n\n"
        "Настройки (длина ответа, часть курса) — кнопка ⚙️ Настройки внизу.",
        reply_markup=_kb_main(),
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = get_settings(context)
    part_label = PARTS[s["part"]][1]
    length_label = str(s["length"]) + (" токенов" if isinstance(s["length"], int) else "")
    await update.message.reply_html(
        "📚 <b>AJTBD Ассистент</b>\n\n"
        "Задавай вопросы по курсу AJTBD Ильи Замезина — отвечаю строго по материалам.\n\n"
        "<b>Текущие настройки:</b>\n"
        f"• Часть курса: {part_label}\n"
        f"• Длина ответа: {length_label}\n\n"
        "Изменить настройки — кнопка ⚙️ Настройки.",
        reply_markup=_kb_main(),
    )


# ---------------------------------------------------------------------------
# Message handler — navigation + RAG
# ---------------------------------------------------------------------------
async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if not text:
        return

    state = get_menu_state(context)
    s = get_settings(context)

    # --- Навигация ---
    if text == BTN_SETTINGS:
        set_menu_state(context, "settings")
        await update.message.reply_text("Настройки:", reply_markup=_kb_settings())
        return

    if text == BTN_BACK:
        parent = "main" if state in ("settings", "awaiting_custom_length") else "settings"
        set_menu_state(context, parent)
        kb = _kb_main() if parent == "main" else _kb_settings()
        label = "Главное меню:" if parent == "main" else "Настройки:"
        await update.message.reply_text(label, reply_markup=kb)
        return

    if text == BTN_LENGTH:
        set_menu_state(context, "length")
        cur = s["length"]
        label = str(cur) + (" токенов" if isinstance(cur, int) else "")
        await update.message.reply_text(f"Длина ответа (сейчас: {label}):", reply_markup=_kb_length())
        return

    if text == BTN_PART:
        set_menu_state(context, "part")
        await update.message.reply_text(
            f"Часть курса (сейчас: {PARTS[s['part']][1]}):", reply_markup=_kb_part()
        )
        return

    if text in LENGTH_BUTTONS:
        s["length"] = text
        set_menu_state(context, "settings")
        await update.message.reply_text(f"✅ Длина ответа: {text}", reply_markup=_kb_settings())
        return

    if text == BTN_LEN_CUSTOM:
        set_menu_state(context, "awaiting_custom_length")
        await update.message.reply_text(
            "Введи максимальное количество токенов (50–2000).\n\n"
            "Токен ≈ 0.75 слова.\n"
            "Пресеты: Коротко = 150, Стандартно = 400, Подробно = 800.",
            reply_markup=_kb_length(),
        )
        return

    if state == "awaiting_custom_length":
        try:
            n = int(text)
            if 50 <= n <= 2000:
                s["length"] = n
                set_menu_state(context, "settings")
                await update.message.reply_text(f"✅ Лимит: {n} токенов", reply_markup=_kb_settings())
            else:
                await update.message.reply_text("Число должно быть от 50 до 2000.")
        except ValueError:
            await update.message.reply_text("Введи целое число, например 300.")
        return

    if text in PART_BUTTONS:
        key = PART_BUTTONS[text]
        s["part"] = key
        set_menu_state(context, "settings")
        await update.message.reply_text(f"✅ Часть курса: {text}", reply_markup=_kb_settings())
        return

    # --- RAG ---
    part_filter = PARTS[s["part"]][0]
    answer_length = s["length"]

    thinking = await update.message.reply_text("🔍 Ищу ответ…")

    try:
        full_text = ""
        relevant = []

        for item in answer_stream(
            query=text,
            chunks=CHUNKS,
            embeddings=EMBEDDINGS,
            bm25=BM25,
            part_filter=part_filter,
            answer_length=answer_length,
        ):
            if isinstance(item, tuple):
                _, relevant = item
                break
            full_text += item

        await thinking.edit_text(full_text or "Не удалось получить ответ.")

        if relevant:
            lines = []
            for c in relevant[:4]:
                title = c.get("part_title", "")
                lecture = c.get("lecture", "")
                score = c["score"]
                preview = c["text"][:150].replace("\n", " ")
                header = f"<b>{title}</b>" + (f" | {lecture}" if lecture else "")
                lines.append(f"📎 {header} (схожесть: {score:.2f})\n<i>{preview}…</i>")

            await update.message.reply_html(
                "📚 <b>Источники:</b>\n\n" + "\n\n".join(lines)
            )

    except Exception:
        logger.exception("Error answering question: %s", text)
        await thinking.edit_text("⚠️ Произошла ошибка при обработке запроса. Попробуй ещё раз.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment / .env")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    logger.info("Bot polling started")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
