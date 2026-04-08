"""
app/streamlit_app.py — Zamesin AJTBD RAG Bot
"""
import streamlit as st
from rag import load_index, build_embeddings_if_needed, answer_stream

st.set_page_config(page_title="AJTBD Ассистент", page_icon="🎯", layout="centered")
st.title("🎯 AJTBD Ассистент")
st.caption("Задавай вопросы по курсу Advanced Jobs To Be Done Ильи Замезина")

@st.cache_resource(show_spinner="Подготовка индекса...")
def get_index():
    build_embeddings_if_needed()
    return load_index()

chunks, embeddings = get_index()

PARTS = {
    "Все части":            None,
    "PART1 — Основы":       "PART1",
    "PART2 — Ценность":     "PART2",
    "PART3 — Запуск":       "PART3",
    "PART4 — Стратегия":    "PART4",
}

with st.sidebar:
    st.header("Настройки")
    part_label  = st.selectbox("Часть курса", list(PARTS.keys()))
    part_filter = PARTS[part_label]
    length      = st.radio("Длина ответа", ["Коротко", "Стандартно", "Подробно"], index=1)

query = st.text_input("Вопрос", placeholder="Что такое Core Job?")

if st.button("Спросить", type="primary") and query.strip():
    st.markdown("### Ответ")

    answer_placeholder = st.empty()
    full_text = ""
    relevant = []

    try:
        for item in answer_stream(
            query=query.strip(),
            chunks=chunks,
            embeddings=embeddings,
            part_filter=part_filter,
            answer_length=length,
        ):
            if isinstance(item, tuple):
                _, relevant = item
                break
            full_text += item
            answer_placeholder.markdown(full_text + "▌")

        answer_placeholder.markdown(full_text)

    except Exception as e:
        st.error(f"Ошибка: {e}")
        st.stop()

    if relevant:
        with st.expander(f"Источники ({len(relevant)})"):
            for c in relevant[:4]:
                label = f"**{c['part_title']}** | {c.get('lecture', '')} | score: {c['score']:.3f}"
                st.markdown(label)
                st.caption(c["text"][:300] + "...")
                st.divider()
