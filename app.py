import pandas as pd
import streamlit as st
from databricks import sql
from openai import OpenAI

# =========================
# Secrets Configuration
# =========================
# In Streamlit Cloud, set them in: Settings -> Secrets
# Example (DO NOT hardcode in code):
# DATABRICKS_SERVER_HOSTNAME = "dbc-....cloud.databricks.com"
# DATABRICKS_HTTP_PATH = "/sql/1.0/warehouses/xxxxxxx"
# DATABRICKS_TOKEN = "dapiXXXXXXXX"
# OPENAI_API_KEY = "sk-XXXXXXXX"

DATABRICKS_SERVER_HOSTNAME = st.secrets["DATABRICKS_SERVER_HOSTNAME"]
DATABRICKS_HTTP_PATH = st.secrets["DATABRICKS_HTTP_PATH"]
DATABRICKS_TOKEN = st.secrets["DATABRICKS_TOKEN"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Max characters of CSV sent to the model
MAX_CSV_CHARS = 80000

# Source & history tables
SOURCE_TABLE = "liturgi.`01_curated`.pdf_liturgi_ai_analysis"
HISTORY_TABLE = "liturgi.`02_app`.liturgi_ai_qa_history"


# =========================
# Databricks helper
# =========================
def _get_connection():
    return sql.connect(
        server_hostname=DATABRICKS_SERVER_HOSTNAME,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN,
    )


# =========================
# Load liturgy data
# =========================
def load_liturgi() -> pd.DataFrame:
    """
    Load liturgy data from the curated table.
    """
    query = f"SELECT * FROM {SOURCE_TABLE}"

    with _get_connection() as connection:
        df = pd.read_sql(query, connection)

    return df


# =========================
# Save & load Q&A history
# =========================
def save_history(
    user_instruction: str,
    full_prompt: str,
    answer: str,
    model_name: str = "gpt-5-nano",
):
    """
    Save Q&A to the history table in Databricks.
    """
    max_len = 65000
    prompt_trimmed = full_prompt[:max_len]
    answer_trimmed = answer[:max_len]

    insert_sql = f"""
        INSERT INTO {HISTORY_TABLE} (
            asked_at,
            source_table,
            limit_rows,
            user_instruction,
            prompt_sent,
            answer,
            model
        )
        VALUES (
            current_timestamp(),
            ?,
            NULL,
            ?,
            ?,
            ?,
            ?
        )
    """

    with _get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                insert_sql,
                (
                    SOURCE_TABLE,
                    user_instruction,
                    prompt_trimmed,
                    answer_trimmed,
                    model_name,
                ),
            )


def load_history(limit_rows: int = 50) -> pd.DataFrame:
    """
    Load Q&A history from the history table.
    """
    query = f"""
        SELECT
          id,
          asked_at,
          limit_rows,
          user_instruction,
          answer,
          model
        FROM {HISTORY_TABLE}
        ORDER BY asked_at DESC
        LIMIT {limit_rows}
    """

    with _get_connection() as connection:
        df = pd.read_sql(query, connection)

    return df


# =========================
# Call ChatGPT
# =========================
def ask_chatgpt(full_prompt: str) -> str:
    """
    Send prompt to ChatGPT (gpt-5-nano) and return its answer.
    """
    resp = client.responses.create(
        model="gpt-5-nano",
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
    return resp.output[0].content[0].text


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

# Session state for last answer
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = ""

# =========================
# Load liturgy data (from cache)
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
        st.metric("Total Liturgies in Database", total_liturgi)

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
        st.info("No liturgy data in the database to analyze yet.")
    else:
        # Default instruction in Bahasa Indonesia (as requested)
        default_instruction = (
            "Tolong analisis dataset liturgi ini:\n"
            "- Ringkas pola umum urutan liturgi dan elemen-elemen pentingnya (misalnya: pembukaan, aanvangstekst, "
            "bacaan Alkitab, genadeverkondiging, prediking, dankofferande, slotlied).\n"
            "- Identifikasi pola dan variasi: misalnya lagu pembukaan yang sering dipakai, kitab/ayat yang sering muncul, "
            "tema-tema yang tampak dari bacaan dan judul khotbah.\n"
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
Berikut adalah seluruh data liturgi dari tabel {SOURCE_TABLE}
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
            status_placeholder.info("Requesting answer from AI...")

            answer = ask_chatgpt(full_prompt)

            # Clear the status message once answer is received
            status_placeholder.empty()

            # Save to session state
            st.session_state["last_answer"] = answer

            # Save to history table
            try:
                save_history(
                    user_instruction=user_instruction,
                    full_prompt=full_prompt,
                    answer=answer,
                    model_name="gpt-5.1",
                )
            except Exception as e:
                st.error(f"Failed to save Q&A history: {e}")

        with a_col:
            # Always show AI Answer text area, even before first answer
            st.text_area(
                "AI Answer",
                value=st.session_state["last_answer"],
                height=110,
            )

        st.markdown("---")
        st.markdown("### AI Q&A History")

        # Only show a small visible area (~3 rows), scroll within table for more
        history_limit = 50
        try:
            df_history = load_history(limit_rows=history_limit)
            if not df_history.empty:
                df_hist_display = df_history.copy()
                # Show full answer content without truncation
                st.dataframe(
                    df_hist_display[
                        ["id", "asked_at", "limit_rows", "user_instruction", "answer", "model"]
                    ],
                    width="stretch",
                    height=130,  # compact height ~3 visible rows, scroll inside table
                )
            else:
                st.info("No Q&A history available yet.")
        except Exception as e:
            st.error(f"Failed to load Q&A history: {e}")
