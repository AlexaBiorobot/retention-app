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

# ---- Фильтры ----
st.sidebar.header("Фильтры")

courses_all = sorted([c for c in df["N"].dropna().unique()])

# Состояния выбора
if "courses_include" not in st.session_state:
    st.session_state["courses_include"] = courses_all.copy()
if "courses_exclude" not in st.session_state:
    st.session_state["courses_exclude"] = []

# Кнопки быстрого выбора
c1, c2 = st.sidebar.columns(2)
if c1.button("Выбрать все"):
    st.session_state["courses_include"] = courses_all.copy()
    st.session_state["courses_exclude"] = []
if c2.button("Снять все"):
    st.session_state["courses_include"] = []
    st.session_state["courses_exclude"] = []

# Включить курсы
include_selected = st.sidebar.multiselect(
    "Курсы — включить",
    options=courses_all,
    default=st.session_state["courses_include"],
    key="courses_include"
)

# Исключить курсы (из доступных лучше показывать весь список, чтобы можно было вычеркнуть прямо здесь)
exclude_selected = st.sidebar.multiselect(
    "Курсы — исключить",
    options=courses_all,
    default=st.session_state["courses_exclude"],
    key="courses_exclude"
)

# Итоговый набор курсов
courses_final = sorted(list(set(include_selected) - set(exclude_selected)))

st.sidebar.caption(f"Итого выбрано курсов: {len(courses_final)}")

# Дата фидбека (A)
min_date = df["A"].min()
max_date = df["A"].max()
if pd.isna(min_date) or pd.isna(max_date):
    date_range = st.sidebar.date_input("Дата фидбека (A)", [])
else:
    date_range = st.sidebar.date_input("Дата фидбека (A)", [min_date.date(), max_date.date()])

# ---- Применение фильтров ----
df_f = df.copy()

if courses_final:
    df_f = df_f[df_f["N"].isin(courses_final)]
else:
    df_f = df_f.iloc[0:0]  # пусто, если ничего не выбрано

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_dt = pd.to_datetime(date_range[0])
    end_dt = pd.to_datetime(date_range[1])
    df_f = df_f[(df_f["A"] >= start_dt) & (df_f["A"] <= end_dt)]

# ---- Группировка: avg(G) и count по S ----
grp = df_f.dropna(subset=["S", "G"])
agg = (
    grp.groupby("S", as_index=False)
       .agg(avg_G=("G", "mean"),
            count=("G", "size"))
       .sort_values("S")
)

st.title("40 week courses")

if agg.empty:
    st.info("Нет данных для выбранных фильтров.")
else:
    # Нижняя граница оси Y = 4, если минимум >= 4
    y_scale = None
    y_min = agg["avg_G"].min()
    y_max = agg["avg_G"].max()
    if pd.notna(y_min) and y_min >= 4:
        y_scale = alt.Scale(domain=[4, float(y_max)])

    chart = (
        alt.Chart(agg)
          .mark_line(point=True)
          .encode(
              x=alt.X("S:Q", title="S"),
              y=alt.Y("avg_G:Q", title="Average G", scale=y_scale),
              tooltip=[
                  alt.Tooltip("S:Q", title="S"),
                  alt.Tooltip("avg_G:Q", title="Average G", format=".2f"),
                  alt.Tooltip("count:Q", title="Кол-во ответов")
              ]
          )
    )
    st.altair_chart(chart, use_container_width=True)
