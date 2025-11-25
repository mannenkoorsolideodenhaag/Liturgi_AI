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
    query = f"""
        SELECT *
        FROM {SOURCE_TABLE}
    """

    with _get_connection() as connection:
        df = pd.read_sql(query, connection)

    return df


# =========================
# Save & load Q&A history
# =========================
def save_history(
    limit_rows: int,
    user_instruction: str,
    full_prompt: str,
    answer: str,
    model_name: str = "gpt-5.1",
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
            ?,
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
                    int(limit_rows),
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
    Send prompt to ChatGPT (gpt-5.1) and return its answer.
    """
    resp = client.responses.create(
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
    return resp.output[0].content[0].text


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Liturgi AI", layout="wide")

# Minimal vertical padding so content fits nicely on one screen
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("GKIN Den Haag Liturgy Exploration")

# Session state for last answer & slider default
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = ""
if "last_limit_rows" not in st.session_state:
    st.session_state["last_limit_rows"] = 10

# =========================
# Load liturgy data once
# =========================
@st.cache_data(show_spinner=True)
def load_liturgi_cached():
    df_lit = load_liturgi()
    return df_lit


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
# Main two-column layout
# =========================
left_col, right_col = st.columns([2, 2])

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
            st.bar_chart(month_counts, height=260, width="stretch")
        else:
            st.info("No liturgy data available by month yet.")

    st.subheader("Liturgy Details")
    # Scrollable table within fixed height
    st.dataframe(
        df_liturgi_enriched,
        width="stretch",
        height=350,
    )

# ---------- RIGHT: AI PROMPT & HISTORY ----------
with right_col:
    st.subheader("Liturgy AI Assistant")

    n_rows = len(df_liturgi_enriched)

    if n_rows <= 0:
        st.info("No liturgy data in the database to analyze yet.")
    else:
        # Slider configuration depends on number of rows
        if n_rows < 10:
            slider_min = 1
            slider_max = n_rows
            step = 1
        else:
            slider_min = 10
            slider_max = min(500, n_rows)
            step = 10

        # Ensure default value stays within bounds
        default_val = st.session_state["last_limit_rows"]
        if default_val < slider_min or default_val > slider_max:
            default_val = slider_min

        limit_rows_for_ai = st.slider(
            "Number of liturgy rows sent to AI",
            min_value=slider_min,
            max_value=slider_max,
            value=default_val,
            step=step,
        )

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

        user_instruction = st.text_area(
            "Question / Instruction for the AI",
            value=default_instruction,
            height=150,
        )

        ask_clicked = st.button("Ask AI")

        if ask_clicked:
            # Prepare subset of data to send to AI
            df_for_ai = df_liturgi_enriched.head(limit_rows_for_ai)
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
Berikut adalah data liturgi dari tabel {SOURCE_TABLE}
(dibatasi {limit_rows_for_ai} baris pertama) dalam format CSV
(dipotong bila terlalu panjang).

INSTRUKSI SAYA:
{user_instruction}

DATA CSV:
```csv
{csv_text_short}
```
""".strip()

            st.info("Requesting answer from AI...")
            answer = ask_chatgpt(full_prompt)

            # Save to session state
            st.session_state["last_answer"] = answer
            st.session_state["last_limit_rows"] = limit_rows_for_ai

            # Save to history table
            try:
                save_history(
                    limit_rows=limit_rows_for_ai,
                    user_instruction=user_instruction,
                    full_prompt=full_prompt,
                    answer=answer,
                    model_name="gpt-5.1",
                )
                st.success("Q&A history saved successfully.")
            except Exception as e:
                st.error(f"Failed to save Q&A history: {e}")

        # Show latest answer
        st.markdown("### AI Answer (Latest)")
        if st.session_state["last_answer"]:
            st.text_area(
                "AI Answer",
                value=st.session_state["last_answer"],
                height=180,
            )
        else:
            st.info("No AI answer yet. Please ask a question first.")

        st.markdown("---")
        st.markdown("### AI Q&A History")

        history_limit = 30  # keep history compact
        try:
            df_history = load_history(limit_rows=history_limit)
            if not df_history.empty:
                df_hist_display = df_history.copy()
                df_hist_display["answer_preview"] = (
                    df_hist_display["answer"].str.slice(0, 120) + "..."
                )
                st.dataframe(
                    df_hist_display[
                        ["id", "asked_at", "limit_rows", "user_instruction", "answer_preview", "model"]
                    ],
                    width="stretch",
                    height=220,
                )
            else:
                st.info("No Q&A history available yet.")
        except Exception as e:
            st.error(f"Failed to load Q&A history: {e}")
