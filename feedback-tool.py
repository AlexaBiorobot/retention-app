import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import altair as alt
import json
import string

SPREADSHEET_ID = "1fR8_Ay7jpzmPCAl6dWSCC7sWw5VJOaNpu5Zp8b78LRg"
SHEET_NAME = "Form Responses 1"

# ---- Авторизация через st.secrets (строка JSON) ----
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]
sa_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT"])
creds = ServiceAccountCredentials.from_json_keyfile_dict(sa_info, scope)
client = gspread.authorize(creds)

# ---- Загрузка данных (без заголовков, присваиваем буквы колонкам) ----
ws = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
values = ws.get_all_values()  # 2D-список: первая строка — заголовки/данные

if not values or len(values) < 2:
    st.warning("Недостаточно данных на листе.")
    st.stop()

num_cols = len(values[0])
letters = list(string.ascii_uppercase)[:num_cols]  # ["A","B","C",...]
df = pd.DataFrame(values[1:], columns=letters)     # игнорируем текстовые заголовки, используем буквы

# ---- Приведение типов ----
# A — дата фидбека, N — курс, S — неделя, G — оценка
df["A"] = pd.to_datetime(df["A"], errors="coerce")
df["S"] = pd.to_numeric(df["S"], errors="coerce")
df["G"] = pd.to_numeric(df["G"], errors="coerce")

# ---- Фильтры ----
st.sidebar.header("Фильтры")

# Курс (N)
courses = sorted([c for c in df["N"].dropna().unique()])
selected_courses = st.sidebar.multiselect("Курс (N)", courses, default=courses if courses else [])

# Дата фидбека (A)
min_date = df["A"].min()
max_date = df["A"].max()
if pd.isna(min_date) or pd.isna(max_date):
    date_range = st.sidebar.date_input("Дата фидбека (A)", [])
else:
    date_range = st.sidebar.date_input("Дата фидбека (A)", [min_date.date(), max_date.date()])

# Применение фильтров (аккуратно, если дата не выбрана)
df_f = df.copy()
if selected_courses:
    df_f = df_f[df_f["N"].isin(selected_courses)]

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_dt = pd.to_datetime(date_range[0])
    end_dt = pd.to_datetime(date_range[1])
    df_f = df_f[(df_f["A"] >= start_dt) & (df_f["A"] <= end_dt)]

# ---- Группировка: avg(G) по S ----
agg = (
    df_f.dropna(subset=["S", "G"])
       .groupby("S", as_index=False)["G"]
       .mean()
       .rename(columns={"G": "avg_G"})
       .sort_values("S")
)

# ---- График ----
st.title("40 week courses")
chart = (
    alt.Chart(agg)
      .mark_line(point=True)
      .encode(
          x=alt.X("S:Q", title="S"),
          y=alt.Y("avg_G:Q", title="Average G"),
          tooltip=["S", "avg_G"]
      )
)
st.altair_chart(chart, use_container_width=True)
