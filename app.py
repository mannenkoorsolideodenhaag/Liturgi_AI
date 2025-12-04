import pandas as pd
import streamlit as st
from openai import OpenAI

# =========================
# CONFIG
# =========================

# Lokasi CSV di dalam repo GitHub kamu
# Contoh: file berada di folder "data/liturgi_data.csv"
CSV_PATH = "data/liturgi_data.csv"

# Max characters of CSV sent to the model
MAX_CSV_CHARS = 80000

# Label sumber data (hanya untuk teks di prompt)
SOURCE_LABEL = f"CSV file: {CSV_PATH}"

# OpenAI client
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# Load liturgy data from CSV
# =========================
def load_liturgi() -> pd.DataFrame:
    """
    Load liturgy data from the CSV file.
    """
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        st.error(
            f"CSV file tidak ditemukan di path: {CSV_PATH}. "
            "Pastikan file sudah di-commit ke repo dan path-nya benar."
        )
        st.stop()
    except Exception as e:
        st.error(f"Terjadi error saat membaca CSV: {e}")
        st.stop()

    return df


# =========================
# Call ChatGPT (gpt-5-nano safe)
# =========================
def ask_chatgpt(full_prompt: str) -> str:
    """
    Send prompt to ChatGPT (gpt-5.1) and return its answer.
    Aman terhadap NoneType (output kosong).
    """
    try:
        resp = client.responses.create(
            #model="gpt-5-nano",
            model="gpt-5.1",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant that analyzes church liturgy data for GKIN. "
                        "You answer in clear, structured Indonesian, easy to understand "
                        "for the liturgy team and congregation."
                    ),
                },
                {"role": "user", "content": full_prompt},
            ],
        )

        answer = None

        # Prioritas 1: gunakan output_text kalau tersedia
        if hasattr(resp, "output_text") and resp.output_text:
            answer = resp.output_text
        else:
            # Prioritas 2: fallback ke struktur output standar jika ada
            try:
                answer = resp.output[0].content[0].text
            except Exception:
                answer = None

        if not answer:
            return "⚠️ Model returned no usable content."

        return answer

    except Exception as e:
        return f"❌ OpenAI error: {str(e)}"


# =========================
# Cached loader
# =========================
@st.cache_data(show_spinner=True)
def load_liturgi_cached():
    """
    Cached wrapper around load_liturgi().
    """
    df_lit = load_liturgi()
    return df_lit


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Liturgi AI", layout="wide")

# Minimal vertical padding so content fits nicely on one screen
# and ensure the Clear Cache button is not hidden by the top bar
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title + Clear Cache button on the same row
title_col, btn_col = st.columns([8, 2])
with title_col:
    st.title("GKIN Den Haag Liturgy Exploration")
with btn_col:
    if st.button("Clear Cache"):
        # Clear the cached liturgy DataFrame and rerun the app
        load_liturgi_cached.clear()
        st.rerun()

# Session state for last answer & history
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = ""

if "qa_history" not in st.session_state:
    # List of dicts: {id, asked_at, user_instruction, answer, model}
    st.session_state["qa_history"] = []


# =========================
# Load liturgy data (from cache, WHICH READS CSV)
# =========================
df_liturgi = load_liturgi_cached()

# Enrich with date & month if available
df_liturgi_enriched = df_liturgi.copy()
if "liturgy_date" in df_liturgi_enriched.columns:
    df_liturgi_enriched["liturgy_date_parsed"] = pd.to_datetime(
        df_liturgi_enriched["liturgy_date"], errors="coerce"
    )
    df_liturgi_enriched["liturgy_month"] = (
        df_liturgi_enriched["liturgy_date_parsed"].dt.to_period("M").astype(str)
    )
else:
    df_liturgi_enriched["liturgy_month"] = "Unknown"


# =========================
# Main three-column layout (40% : separator : 60%)
# =========================
left_col, sep_col, right_col = st.columns([4, 0.2, 5.8])

# ---------- LEFT: LITURGY DASHBOARD ----------
with left_col:
    st.subheader("Liturgy Overview")

    top_left, top_right = st.columns(2)

    with top_left:
        total_liturgi = len(df_liturgi_enriched)
        st.metric("Total Liturgies in CSV", total_liturgi)

    with top_right:
        # Aggregate by month
        month_counts = (
            df_liturgi_enriched.groupby("liturgy_month")
            .size()
            .reset_index(name="liturgy_count")
            .sort_values("liturgy_month")
        )
        if not month_counts.empty:
            month_counts = month_counts.set_index("liturgy_month")
            st.caption("Number of Liturgies per Month")
            # Reduced height for more compact dashboard
            st.bar_chart(month_counts, height=160, width="stretch")
        else:
            st.info("No liturgy data available by month yet.")

    st.subheader("Liturgy Details")
    # Table with compact height (~5 visible rows, rest scrollable)
    st.dataframe(
        df_liturgi_enriched,
        width="stretch",
        height=190,
    )

# ---------- MIDDLE: VERTICAL SEPARATOR ----------
with sep_col:
    # Draw a full-height vertical line inside this narrow column
    st.markdown(
        "<div style='border-left: 1px solid #dddddd; height: 100vh; margin: 0 auto;'></div>",
        unsafe_allow_html=True,
    )

# ---------- RIGHT: AI PROMPT & HISTORY ----------
with right_col:
    st.subheader("Liturgy AI Assistant")

    n_rows = len(df_liturgi_enriched)

    if n_rows <= 0:
        st.info("Tidak ada data liturgi di CSV untuk dianalisis.")
    else:
        # Default instruction in Bahasa Indonesia
        default_instruction = (
            "Tolong analisis dataset liturgi ini:\n"
            "- Ringkas pola umum urutan liturgi dan elemen-elemen pentingnya "
            "(misalnya: pembukaan, aanvangstekst, bacaan Alkitab, "
            "genadeverkondiging, prediking, dankofferande, slotlied).\n"
            "- Identifikasi pola dan variasi: misalnya lagu pembukaan yang sering dipakai, "
            "kitab/ayat yang sering muncul, tema-tema yang tampak dari bacaan dan judul khotbah.\n"
            "- Berikan 5–10 insight praktis yang dapat membantu tim liturgi dalam merencanakan ibadah ke depan "
            "(misalnya keseimbangan tema, variasi lagu, keterlibatan jemaat dalam nyanyian).\n"
            "- Jelaskan dengan bahasa yang mudah dimengerti oleh tim liturgi dan majelis."
        )

        # Row 1: two columns (Question on the left, Answer on the right)
        q_col, a_col = st.columns(2)

        with q_col:
            user_instruction = st.text_area(
                "Question / Instruction for the AI",
                value=default_instruction,
                height=110,
            )
            ask_clicked = st.button("Ask AI")

        if ask_clicked:
            # Prepare ALL liturgy data to send to AI (subject to MAX_CSV_CHARS)
            df_for_ai = df_liturgi_enriched
            csv_text = df_for_ai.to_csv(index=False)
            if len(csv_text) > MAX_CSV_CHARS:
                csv_text_short = csv_text[:MAX_CSV_CHARS]
                st.warning(
                    f"CSV has length {len(csv_text)} characters. "
                    f"Only the first {MAX_CSV_CHARS} characters are sent to the model."
                )
            else:
                csv_text_short = csv_text

            full_prompt = f"""
Berikut adalah seluruh data liturgi dari {SOURCE_LABEL}
dalam format CSV (dipotong bila terlalu panjang).

INSTRUKSI SAYA:
{user_instruction}

DATA CSV:
```csv
{csv_text_short}
```
""".strip()

            # Show temporary status while waiting for AI, then clear it
            status_placeholder = st.empty()
            status_placeholder.info("Meminta jawaban dari AI...")

            answer = ask_chatgpt(full_prompt)

            # Clear the status message once answer is received
            status_placeholder.empty()

            # Save to session state
            st.session_state["last_answer"] = answer

            # Simpel history di memory (session_state)
            st.session_state["qa_history"].insert(
                0,  # prepend supaya terbaru di atas
                {
                    "id": len(st.session_state["qa_history"]) + 1,
                    "asked_at": pd.Timestamp.utcnow(),
                    "limit_rows": None,
                    "user_instruction": user_instruction,
                    "answer": answer,
                    "model": "gpt-5-nano",
                },
            )

        with a_col:
            # Always show AI Answer text area, even before first answer
            st.text_area(
                "AI Answer",
                value=st.session_state["last_answer"],
                height=110,
            )
