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

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
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
CHUNKS, EMBEDDINGS = load_index()
logger.info("Index ready: %d chunks", len(CHUNKS))

# ---------------------------------------------------------------------------
# Settings schema
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
LENGTHS = ["Коротко", "Стандартно", "Подробно"]
DEFAULT_SETTINGS = {"part": "all", "length": "Подробно"}


def get_settings(context: ContextTypes.DEFAULT_TYPE) -> dict:
    if "settings" not in context.user_data:
        context.user_data["settings"] = dict(DEFAULT_SETTINGS)
    return context.user_data["settings"]


# ---------------------------------------------------------------------------
# Keyboard builders
# ---------------------------------------------------------------------------
def _part_keyboard(current: str) -> InlineKeyboardMarkup:
    rows = []
    for key, (_, label) in PARTS.items():
        marker = "✅ " if key == current else ""
        rows.append([InlineKeyboardButton(marker + label, callback_data=f"part:{key}")])
    return InlineKeyboardMarkup(rows)


def _length_keyboard(current: str) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(
            ("✅ " if l == current else "") + l,
            callback_data=f"length:{l}",
        )
        for l in LENGTHS
    ]
    return InlineKeyboardMarkup([buttons])


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = get_settings(context)
    await update.message.reply_html(
        "👋 <b>Привет!</b> Я ассистент по курсу <b>Advanced Jobs To Be Done</b> Ильи Замезина.\n\n"
        "Просто напиши свой вопрос — я отвечу, опираясь на материалы курса.\n\n"
        "<b>Команды:</b>\n"
        "/part — выбрать часть курса\n"
        f"/length — длина ответа (сейчас: <b>{s['length']}</b>)\n"
        "/help — справка"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = get_settings(context)
    part_label = PARTS[s["part"]][1]
    await update.message.reply_html(
        "📚 <b>AJTBD Ассистент</b>\n\n"
        "Задавай вопросы по курсу AJTBD Ильи Замезина — отвечаю строго по материалам.\n\n"
        "<b>Текущие настройки:</b>\n"
        f"• Часть курса: {part_label}\n"
        f"• Длина ответа: {s['length']}\n\n"
        "/part — сменить часть курса\n"
        "/length — сменить длину ответа"
    )


async def cmd_part(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = get_settings(context)
    await update.message.reply_text("Выбери часть курса:", reply_markup=_part_keyboard(s["part"]))


async def cmd_length(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = get_settings(context)
    await update.message.reply_text("Выбери длину ответа:", reply_markup=_length_keyboard(s["length"]))


# ---------------------------------------------------------------------------
# Inline button callbacks
# ---------------------------------------------------------------------------
async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    s = get_settings(context)
    data = query.data

    if data.startswith("part:"):
        key = data.split(":", 1)[1]
        s["part"] = key
        label = PARTS[key][1]
        await query.edit_message_text(
            f"✅ Часть курса: <b>{label}</b>",
            parse_mode="HTML",
            reply_markup=_part_keyboard(key),
        )
    elif data.startswith("length:"):
        val = data.split(":", 1)[1]
        s["length"] = val
        await query.edit_message_text(
            f"✅ Длина ответа: <b>{val}</b>",
            parse_mode="HTML",
            reply_markup=_length_keyboard(val),
        )


# ---------------------------------------------------------------------------
# Message handler — main RAG flow
# ---------------------------------------------------------------------------
async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text.strip()
    if not question:
        return

    s = get_settings(context)
    part_filter = PARTS[s["part"]][0]
    answer_length = s["length"]

    thinking = await update.message.reply_text("🔍 Ищу ответ…")

    try:
        full_text = ""
        relevant = []

        for item in answer_stream(
            query=question,
            chunks=CHUNKS,
            embeddings=EMBEDDINGS,
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
        logger.exception("Error answering question: %s", question)
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
    app.add_handler(CommandHandler("part", cmd_part))
    app.add_handler(CommandHandler("length", cmd_length))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    logger.info("Bot polling started")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
