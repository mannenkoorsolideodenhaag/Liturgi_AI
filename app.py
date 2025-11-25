import pandas as pd
import streamlit as st
from databricks import sql
from openai import OpenAI

# =========================
# Konfigurasi dari Secrets
# =========================
# Di Streamlit Cloud, set di: Settings -> Secrets
# Contoh isi secrets (JANGAN di code):
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

# Batas maksimum karakter CSV yang dikirim ke model
MAX_CSV_CHARS = 80000

# Tabel sumber & tabel riwayat
SOURCE_TABLE = "liturgi.`01_curated`.pdf_liturgi_ai_analysis"
HISTORY_TABLE = "liturgi.`02_app`.liturgi_ai_qa_history"


# =========================
# Fungsi util Databricks
# =========================
def _get_connection():
    return sql.connect(
        server_hostname=DATABRICKS_SERVER_HOSTNAME,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN,
    )


# =========================
# Fungsi ambil data liturgi
# =========================
def load_liturgi() -> pd.DataFrame:
    """
    Ambil seluruh data (atau sebagian besar) dari tabel liturgi.`01_curated`.pdf_liturgi_ai_analysis.
    Jika datanya sangat besar, bisa ditambahkan LIMIT di query.
    """
    query = f"""
        SELECT *
        FROM {SOURCE_TABLE}
    """

    with _get_connection() as connection:
        df = pd.read_sql(query, connection)

    return df


# =========================
# Fungsi simpan & baca riwayat
# =========================
def save_history(
    limit_rows: int,
    user_instruction: str,
    full_prompt: str,
    answer: str,
    model_name: str = "gpt-5.1",
):
    """
    Simpan Q&A ke tabel riwayat di Databricks.
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
    Ambil riwayat Q&A dari tabel riwayat.
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
# Fungsi panggil ChatGPT
# =========================
def ask_chatgpt(full_prompt: str) -> str:
    """
    Kirim prompt ke ChatGPT (gpt-5.1) dan balikan teks jawabannya.
    """
    resp = client.responses.create(
        model="gpt-5.1",
        input=[
            {
                "role": "system",
                "content": (
                    "Kamu adalah asisten AI yang menganalisis data liturgi ibadah GKIN "
                    "dan menjawab dalam bahasa Indonesia yang jelas, terstruktur, dan mudah dimengerti "
                    "oleh tim liturgi maupun jemaat."
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

# Sedikit CSS untuk mengurangi scroll utama dan memberi tinggi tetap pada beberapa komponen
st.markdown(
    """
    <style>
    /* Kurangi padding atas-bawah supaya konten muat di layar */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("GKIN Den Haag Liturgi Exploration")

# Inisialisasi session state untuk jawaban terakhir
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = ""
if "last_limit_rows" not in st.session_state:
    st.session_state["last_limit_rows"] = 10

# =========================
# Load data sekali di awal
# =========================
@st.cache_data(show_spinner=True)
def load_liturgi_cached():
    df_lit = load_liturgi()
    return df_lit


df_liturgi = load_liturgi_cached()

# Tambahkan kolom tanggal/bulan jika ada liturgy_date
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
# Layout utama: dua kolom
# =========================
left_col, right_col = st.columns([2, 2])

# ---------- LEFT: DASHBOARD DATA LITURGI ----------
with left_col:
    st.subheader("Liturgi Overview")

    top_left, top_right = st.columns(2)

    with top_left:
        total_liturgi = len(df_liturgi_enriched)
        st.metric("Jumlah Liturgi di Database", total_liturgi)

    with top_right:
        # Aggregasi per bulan
        month_counts = (
            df_liturgi_enriched.groupby("liturgy_month")
            .size()
            .reset_index(name="jumlah_liturgi")
            .sort_values("liturgy_month")
        )
        if not month_counts.empty:
            month_counts = month_counts.set_index("liturgy_month")
            st.caption("Jumlah liturgi per bulan")
            st.bar_chart(month_counts, height=260, width="stretch")
        else:
            st.info("Belum ada data liturgi untuk ditampilkan per bulan.")

    st.subheader("Detail Liturgi")
    # Tabel dengan scroll internal
    st.dataframe(
        df_liturgi_enriched,
        width="stretch",
        height=350,
    )

# ---------- RIGHT: AI PROMPT & HISTORY ----------
with right_col:
    st.subheader("Liturgi AI Assistant")

    n_rows = len(df_liturgi_enriched)

    if n_rows <= 0:
        st.info("Belum ada data liturgi di database untuk dianalisis.")
    else:
        # Limit rows untuk data yang dikirim ke AI (agar CSV tidak terlalu besar)
        if n_rows < 10:
            slider_min = 1
            slider_max = n_rows
            step = 1
        else:
            slider_min = 10
            slider_max = min(500, n_rows)
            step = 10

        # pastikan default ada di antara min dan max
        default_val = st.session_state["last_limit_rows"]
        if default_val < slider_min or default_val > slider_max:
            default_val = slider_min

        limit_rows_for_ai = st.slider(
            "Jumlah baris liturgi yang dikirim ke AI",
            min_value=slider_min,
            max_value=slider_max,
            value=default_val,
            step=step,
        )

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
            "Pertanyaan / Instruksi ke AI",
            value=default_instruction,
            height=150,
        )

        ask_clicked = st.button("Ask AI")

        if ask_clicked:
            # Siapkan subset data untuk AI
            df_for_ai = df_liturgi_enriched.head(limit_rows_for_ai)
            csv_text = df_for_ai.to_csv(index=False)
            if len(csv_text) > MAX_CSV_CHARS:
                csv_text_short = csv_text[:MAX_CSV_CHARS]
                st.warning(
                    f"CSV panjangnya {len(csv_text)} karakter. "
                    f"Hanya {MAX_CSV_CHARS} karakter pertama yang dikirim ke model."
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

            st.info("Meminta jawaban dari AI...")
            answer = ask_chatgpt(full_prompt)

            # Simpan ke session state
            st.session_state["last_answer"] = answer
            st.session_state["last_limit_rows"] = limit_rows_for_ai

            # Simpan ke riwayat
            try:
                save_history(
                    limit_rows=limit_rows_for_ai,
                    user_instruction=user_instruction,
                    full_prompt=full_prompt,
                    answer=answer,
                    model_name="gpt-5.1",
                )
                st.success("Riwayat pertanyaan & jawaban berhasil disimpan.")
            except Exception as e:
                st.error(f"Gagal menyimpan riwayat: {e}")

        # Tampilkan jawaban terakhir
        st.markdown("### Jawaban AI (Terakhir)")
        if st.session_state["last_answer"]:
            st.text_area(
                "Jawaban AI",
                value=st.session_state["last_answer"],
                height=180,
            )
        else:
            st.info("Belum ada jawaban. Silakan ajukan pertanyaan terlebih dahulu.")

        st.markdown("---")
        st.markdown("### Riwayat Pertanyaan & Jawaban")

        history_limit = 30  # fixed number to keep UI compact
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
                st.info("Belum ada riwayat Q&A yang tersimpan.")
        except Exception as e:
            st.error(f"Gagal mengambil riwayat Q&A: {e}")
