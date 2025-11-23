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


# =========================
# Fungsi ambil data liturgi
# =========================
def load_liturgi(limit_rows: int = 100) -> pd.DataFrame:
    """
    Ambil data dari tabel liturgi.`01_curated`.pdf_liturgi_ai_analysis
    dan kembalikan sebagai pandas DataFrame.
    """
    query = f"""
        SELECT *
        FROM liturgi.`01_curated`.pdf_liturgi_ai_analysis
        LIMIT {limit_rows}
    """

    with sql.connect(
        server_hostname=DATABRICKS_SERVER_HOSTNAME,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN,
    ) as connection:
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

st.title("GKIN Den Haag Liturgi AI Prompt")
st.write(
    "Aplikasi AI GKIN Den Haag yang menganalisis seluruh data liturgi kebaktian "
    "dari PDF liturgi mingguan, disimpan di Databricks, distrukturisasi, dan "
    "dianalisis oleh OpenAI model 5.1."
)

# Pilih jumlah baris
limit_rows = st.slider(
    "Jumlah baris yang diambil dari tabel Databricks",
    min_value=10,
    max_value=500,
    value=100,
    step=10,
)

# Prompt dari kamu (default khusus liturgi)
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
    "Instruksi / Prompt ke AI",
    value=default_instruction,
    height=260,
)

st.markdown("---")

if st.button("Kirim ke AI"):
    # 1. Ambil data dari Databricks
    with st.spinner("Mengambil data liturgi dari Databricks..."):
        df = load_liturgi(limit_rows=limit_rows)

    st.success(f"Berhasil ambil {len(df)} baris dan {len(df.columns)} kolom.")
    st.subheader("Preview Data (5 baris pertama)")
    st.dataframe(df.head())

    # 2. Convert ke CSV dan batasi panjang
    csv_text = df.to_csv(index=False)
    if len(csv_text) > MAX_CSV_CHARS:
        csv_text_short = csv_text[:MAX_CSV_CHARS]
        st.warning(
            f"CSV panjangnya {len(csv_text)} karakter. "
            f"Hanya {MAX_CSV_CHARS} karakter pertama yang dikirim ke model."
        )
    else:
        csv_text_short = csv_text

    # 3. Buat prompt final
    full_prompt = f"""
Berikut adalah data liturgi dari tabel liturgi.`01_curated`.pdf_liturgi_ai_analysis
dalam format CSV (dipotong bila terlalu panjang).

INSTRUKSI SAYA:
{user_instruction}

DATA CSV:
```csv
{csv_text_short}
```
""".strip()

    # 4. Panggil ChatGPT
    st.info("Meminta jawaban dari AI...")
    answer = ask_chatgpt(full_prompt)

    # 5. Tampilkan jawaban
    st.markdown("### Jawaban AI")
    st.markdown(answer)
