import pandas as pd
import streamlit as st
from openai import OpenAI

# =========================
# CONFIG
# =========================

CSV_PATH = "data/liturgi_data.csv"
MAX_CSV_CHARS = 80000
SOURCE_LABEL = f"CSV file: {CSV_PATH}"

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# Load CSV
# =========================
def load_liturgi() -> pd.DataFrame:
    try:
        return pd.read_csv(CSV_PATH)
    except Exception as e:
        st.error(f"Error membaca CSV: {e}")
        st.stop()


# =========================
# Ask ChatGPT
# =========================
def ask_chatgpt(full_prompt: str) -> str:
    try:
        resp = client.responses.create(
            model="gpt-5-nano",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant that analyzes church liturgy data for GKIN. "
                        "Jawab dengan bahasa Indonesia yang jelas dan terstruktur."
                    ),
                },
                {"role": "user", "content": full_prompt},
            ],
        )

        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text

        try:
            return resp.output[0].content[0].text
        except Exception:
            return "⚠️ Model returned no usable content."

    except Exception as e:
        return f"❌ OpenAI error: {str(e)}"


@st.cache_data(show_spinner=True)
def load_liturgi_cached():
    return load_liturgi()


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Liturgi AI", layout="wide")

st.title("GKIN Den Haag Liturgy Exploration")

if st.button("Clear Cache"):
    load_liturgi_cached.clear()
    st.rerun()

if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = ""

if "qa_history" not in st.session_state:
    st.session_state["qa_history"] = []

df_liturgi = load_liturgi_cached()
df_enriched = df_liturgi.copy()

if "liturgy_date" in df_enriched.columns:
    df_enriched["liturgy_date_parsed"] = pd.to_datetime(
        df_enriched["liturgy_date"], errors="coerce"
    )
    df_enriched["liturgy_month"] = (
        df_enriched["liturgy_date_parsed"].dt.to_period("M").astype(str)
    )
else:
    df_enriched["liturgy_month"] = "Unknown"

left, mid, right = st.columns([4, 0.2, 5.8])

with left:
    st.subheader("Liturgy Overview")

    st.metric("Total Liturgies", len(df_enriched))

    month_counts = (
        df_enriched.groupby("liturgy_month")
        .size()
        .reset_index(name="liturgy_count")
        .sort_values("liturgy_month")
    )

    if not month_counts.empty:
        st.bar_chart(month_counts.set_index("liturgy_month"))

    st.subheader("Liturgy Details")
    st.dataframe(df_enriched, height=200)


with mid:
    st.markdown("<div style='border-left:1px solid #ccc;height:100vh;'></div>", unsafe_allow_html=True)


with right:
    st.subheader("Liturgy AI Assistant")

    default_instruction = (
        "Tolong analisis dataset liturgi ini:\n"
        "- Ringkas pola umum urutan liturgi.\n"
        "- Identifikasi pola lagu, ayat, tema kotbah.\n"
        "- Berikan 5–10 insight praktis untuk tim liturgi.\n"
        "- Gunakan bahasa yang mudah dimengerti."
    )

    qcol, acol = st.columns(2)

    with qcol:
        user_instruction = st.text_area(
            "Question / Instruction", value=default_instruction, height=110
        )
        ask = st.button("Ask AI")

    if ask:
        csv_text = df_enriched.to_csv(index=False)
        csv_short = csv_text[:MAX_CSV_CHARS]

        full_prompt = f"""
Berikut adalah seluruh data liturgi dari {SOURCE_LABEL} dalam format CSV.

INSTRUKSI:
{user_instruction}

DATA CSV:
```csv
{csv_short}
```
""".strip()

        with st.spinner("Meminta jawaban AI..."):
            answer = ask_chatgpt(full_prompt)

        st.session_state["last_answer"] = answer
        st.session_state["qa_history"].insert(
            0,
            {
                "id": len(st.session_state["qa_history"]) + 1,
                "asked_at": pd.Timestamp.utcnow(),
                "instruction": user_instruction,
                "answer": answer,
                "model": "gpt-5-nano",
            },
        )

    with acol:
        st.text_area("AI Answer", value=st.session_state["last_answer"], height=110)

    st.markdown("---")
    st.markdown("### AI Q&A History (Session Only)")

    hist = st.session_state["qa_history"]

    if hist:
        df_hist = pd.DataFrame(hist)
        if "asked_at" in df_hist.columns:
            df_hist["asked_at"] = pd.to_datetime(df_hist["asked_at"]).dt.tz_convert(
                "Europe/Amsterdam"
            )

        st.dataframe(
            df_hist[["id", "asked_at", "instruction", "answer", "model"]],
            height=200,
        )
    else:
        st.info("Belum ada history.")

