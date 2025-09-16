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

# ==================== ФИЛЬТРЫ ====================
st.sidebar.header("Фильтры")

# --- Курс (N): один выпадающий список с поиском и Select all внутри ---
courses_all = sorted([c for c in df["N"].dropna().unique()])

if "courses_selected" not in st.session_state:
    st.session_state["courses_selected"] = courses_all.copy()

label_text = f"Курсы (N): {len(st.session_state['courses_selected'])}/{len(courses_all)} выбрано"
with st.sidebar.popover(label_text):
    # Поиск по значениям
    search = st.text_input("Search value", placeholder="Начните вводить название курса...")
    if search:
        filtered = [c for c in courses_all if search.lower() in str(c).lower()]
    else:
        filtered = courses_all

    # Чекбокс Select all применим к текущему отфильтрованному списку
    all_in_selected = set(filtered).issubset(set(st.session_state["courses_selected"]))
    select_all = st.checkbox("Select all", value=all_in_selected)

    # Мультивыбор по отфильтрованному списку
    selected_in_filtered = st.multiselect(
        label="Выбор курсов",
        options=filtered,
        default=[c for c in filtered if c in st.session_state["courses_selected"]],
        label_visibility="collapsed",
        placeholder="Выберите курсы…"
    )

    # Логика обновления общего набора выбранных курсов
    current = set(st.session_state["courses_selected"])
    if select_all:
        # Добавляем все из filtered
        current = current.union(filtered)
    else:
        # Сохраняем выбор вне filtered и заменяем внутри filtered на то, что отмечено
        current = (current - set(filtered)).union(selected_in_filtered)

    st.session_state["courses_selected"] = sorted(current)

# --- Дата фидбека (A) ---
min_date = df["A"].min()
max_date = df["A"].max()
if pd.isna(min_date) or pd.isna(max_date):
    date_range = st.sidebar.date_input("Дата фидбека (A)", [])
else:
    date_range = st.sidebar.date_input("Дата фидбека (A)", [min_date.date(), max_date.date()])

# ==================== ПРИМЕНЕНИЕ ФИЛЬТРОВ ====================
df_f = df.copy()

if st.session_state["courses_selected"]:
    df_f = df_f[df_f["N"].isin(st.session_state["courses_selected"])]
else:
    df_f = df_f.iloc[0:0]  # пустой df, если ничего не выбрано

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_dt = pd.to_datetime(date_range[0])
    end_dt = pd.to_datetime(date_range[1])
    df_f = df_f[(df_f["A"] >= start_dt) & (df_f["A"] <= end_dt)]

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
    # Нижняя граница оси Y = 4, если минимум >= 4 (убираем «пустоту» снизу)
    y_scale = None
    y_min = float(agg["avg_G"].min())
    y_max = float(agg["avg_G"].max())
    if y_min >= 4:
        y_scale = alt.Scale(domain=[4, max(4.0, y_max)])

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
