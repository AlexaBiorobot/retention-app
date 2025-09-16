import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import altair as alt
import json
import string
from datetime import datetime

SPREADSHEET_ID = "1fR8_Ay7jpzmPCAl6dWSCC7sWw5VJOaNpu5Zp8b78LRg"
SHEET_NAME = "Form Responses 1"

# ---- Авторизация через st.secrets (строка JSON) ----
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]
sa_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT"])
creds = ServiceAccountCredentials.from_json_keyfile_dict(sa_info, scope)
client = gspread.authorize(creds)

# ---- Загрузка данных ----
ws = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
values = ws.get_all_values()

if not values or len(values) < 2:
    st.warning("Недостаточно данных на листе.")
    st.stop()

num_cols = len(values[0])
letters = list(string.ascii_uppercase)[:num_cols]  # ["A","B","C",...]
df = pd.DataFrame(values[1:], columns=letters)

# ---- Приведение типов ----
# A — дата фидбека, N — курс, S — неделя, G — оценка
df["A"] = pd.to_datetime(df["A"], errors="coerce")
df["S"] = pd.to_numeric(df["S"], errors="coerce")
df["G"] = pd.to_numeric(df["G"], errors="coerce")

# ==================== ФИЛЬТРЫ ====================
st.sidebar.header("Фильтры")

# --- Курсы (N): дефолтный multiselect + Select all / Clear ---
courses_all = sorted([c for c in df["N"].dropna().unique()])

if "courses_selected" not in st.session_state:
    st.session_state["courses_selected"] = courses_all.copy()

b1, b2 = st.sidebar.columns(2)
if b1.button("Select all"):
    st.session_state["courses_selected"] = courses_all.copy()
    st.rerun()
if b2.button("Clear"):
    st.session_state["courses_selected"] = []
    st.rerun()

selected_courses = st.sidebar.multiselect(
    "Курсы (N)",
    options=courses_all,
    default=st.session_state["courses_selected"],
    key="courses_selected",
    help="Начни печатать для поиска. Можно выбрать несколько."
)

# --- Дата фидбека (A) ---
min_date = df["A"].min()
max_date = df["A"].max()
if pd.isna(min_date) or pd.isna(max_date):
    date_range = st.sidebar.date_input("Дата фидбека (A)", [])
else:
    date_range = st.sidebar.date_input(
        "Дата фидбека (A)",
        [min_date.date(), max_date.date()]
    )

# ==================== ПРИМЕНЕНИЕ ФИЛЬТРОВ ====================
df_f = df.copy()

# по курсам
if selected_courses:
    df_f = df_f[df_f["N"].isin(selected_courses)]
else:
    df_f = df_f.iloc[0:0]  # пусто, если ничего не выбрано

# по датам (учитываем, если пользователь выбрал одну дату)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_dt = pd.to_datetime(date_range[0])
    end_dt = pd.to_datetime(date_range[1])
    df_f = df_f[(df_f["A"] >= start_dt) & (df_f["A"] <= end_dt)]
elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
    only_dt = pd.to_datetime(date_range[0])
    df_f = df_f[df_f["A"].dt.date == only_dt.date()]

# ==================== АГРЕГАЦИЯ ====================
grp = df_f.dropna(subset=["S", "G"])
agg = (
    grp.groupby("S", as_index=False)
       .agg(avg_G=("G", "mean"),
            count=("G", "size"))
       .sort_values("S")
)

# ==================== ГРАФИК ====================
st.title("40 week courses")

if agg.empty:
    st.info("Нет данных для выбранных фильтров.")
else:
    chart = (
        alt.Chart(agg)
          .mark_line(point=True)
          .encode(
              x=alt.X("S:Q", title="S"),
              y=alt.Y("avg_G:Q", title="Average G"),  # динамическая шкала
              tooltip=[
                  alt.Tooltip("S:Q", title="S"),
                  alt.Tooltip("avg_G:Q", title="Average G", format=".2f"),
                  alt.Tooltip("count:Q", title="Кол-во ответов")
              ]
          )
    )
    st.altair_chart(chart, use_container_width=True)
