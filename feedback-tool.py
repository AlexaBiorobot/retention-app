import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import altair as alt
import json
import string

st.set_page_config(layout="wide", page_title="40 week courses")

SPREADSHEET_ID = "1fR8_Ay7jpzmPCAl6dWSCC7sWw5VJOaNpu5Zp8b78LRg"

# ---- Авторизация через st.secrets (строка JSON) ----
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]
sa_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT"])
creds = ServiceAccountCredentials.from_json_keyfile_dict(sa_info, scope)
client = gspread.authorize(creds)

def load_sheet_as_letter_df(sheet_name: str) -> pd.DataFrame:
    ws = client.open_by_key(SPREADSHEET_ID).worksheet(sheet_name)
    values = ws.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame()
    num_cols = len(values[0])
    letters = list(string.ascii_uppercase)[:num_cols]  # A..Z
    return pd.DataFrame(values[1:], columns=letters)

# ---------- ДАННЫЕ ----------
df1 = load_sheet_as_letter_df("Form Responses 1")   # A=date, N=course, S=x, G=y
df2 = load_sheet_as_letter_df("Form Responses 2")   # A=date, M=course, R=x, I=y

# Приведение типов
for df, date_col, x_col, y_col in [
    (df1, "A", "S", "G"),
    (df2, "A", "R", "I"),
]:
    if df.empty:
        continue
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

# ---------- ЕДИНЫЕ ФИЛЬТРЫ ----------
st.sidebar.header("Фильтры")

# Курсы: объединяем множества из обеих таблиц
courses_union = sorted(list(set(
    ([] if df1.empty else df1["N"].dropna().unique().tolist()) +
    ([] if df2.empty else df2["M"].dropna().unique().tolist())
)))

if "courses_selected" not in st.session_state:
    st.session_state["courses_selected"] = courses_union.copy()

b1, b2 = st.sidebar.columns(2)
if b1.button("Select all"):
    st.session_state["courses_selected"] = courses_union.copy()
    st.rerun()
if b2.button("Clear"):
    st.session_state["courses_selected"] = []
    st.rerun()

selected_courses = st.sidebar.multiselect(
    "Курсы",
    options=courses_union,
    default=st.session_state["courses_selected"],
    key="courses_selected",
    help="Можно выбрать несколько; поиск поддерживается."
)
st.sidebar.caption(f"Выбрано: {len(selected_courses)} из {len(courses_union)}")

# Дата: общий диапазон по двум таблицам
def safe_minmax(dt1_min, dt1_max, dt2_min, dt2_max):
    mins = [d for d in [dt1_min, dt2_min] if pd.notna(d)]
    maxs = [d for d in [dt1_max, dt2_max] if pd.notna(d)]
    return (min(mins) if mins else pd.NaT, max(maxs) if maxs else pd.NaT)

min1, max1 = (df1["A"].min(), df1["A"].max()) if not df1.empty else (pd.NaT, pd.NaT)
min2, max2 = (df2["A"].min(), df2["A"].max()) if not df2.empty else (pd.NaT, pd.NaT)
glob_min, glob_max = safe_minmax(min1, max1, min2, max2)

if pd.isna(glob_min) or pd.isna(glob_max):
    date_range = st.sidebar.date_input("Дата фидбека (A)", [])
else:
    date_range = st.sidebar.date_input("Дата фидбека (A)", [glob_min.date(), glob_max.date()])

# Гранулярность для распределений
granularity = st.sidebar.selectbox("Гранулярность для распределения", ["День", "Неделя", "Месяц"])

# ---------- ФУНКЦИИ ФИЛЬТРАЦИИ/АГРЕГАЦИИ ----------
def filter_df(df: pd.DataFrame, course_col: str, date_col: str,
              selected_courses, date_range):
    if df.empty:
        return df
    dff = df.copy()
    if selected_courses:
        dff = dff[dff[course_col].isin(selected_courses)]
    else:
        return dff.iloc[0:0]  # пусто
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1])
        dff = dff[(dff[date_col] >= start_dt) & (dff[date_col] <= end_dt)]
    elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
        only_dt = pd.to_datetime(date_range[0])
        dff = dff[dff[date_col].dt.date == only_dt.date()]
    return dff

def apply_filters_and_aggregate(df: pd.DataFrame, course_col: str, date_col: str,
                                x_col: str, y_col: str):
    dff = filter_df(df, course_col, date_col, selected_courses, date_range)
    grp = dff.dropna(subset=[x_col, y_col])
    if grp.empty:
        return pd.DataFrame(columns=[x_col, "avg_y", "count"])
    agg = (grp.groupby(x_col, as_index=False)
             .agg(avg_y=(y_col, "mean"),
                  count=(y_col, "size"))
             .sort_values(x_col))
    return agg

def add_bucket(dff: pd.DataFrame, date_col: str, granularity: str) -> pd.DataFrame:
    out = dff.copy()
    if granularity == "День":
        out["bucket"] = out[date_col].dt.floor("D")
    elif granularity == "Неделя":
        # понедельник недели
        out["bucket"] = out[date_col].dt.to_period("W-MON").dt.start_time
    else:  # "Месяц"
        out["bucket"] = out[date_col].dt.to_period("M").dt.to_timestamp()
    return out

# ---------- ПРИМЕНЕНИЕ ----------
agg1 = apply_filters_and_aggregate(df1, course_col="N", date_col="A", x_col="S", y_col="G")
agg2 = apply_filters_and_aggregate(df2, course_col="M", date_col="A", x_col="R", y_col="I")

# для распределений нужны отфильтрованные «сырые» значения
df1_f = filter_df(df1, course_col="N", date_col="A", selected_courses=selected_courses, date_range=date_range)
df2_f = filter_df(df2, course_col="M", date_col="A", selected_courses=selected_courses, date_range=date_range)

if not df1_f.empty:
    df1_f = df1_f.dropna(subset=["A", "G"])
    df1_f = add_bucket(df1_f, "A", granularity)
if not df2_f.empty:
    df2_f = df2_f.dropna(subset=["A", "I"])
    df2_f = add_bucket(df2_f, "A", granularity)

# ---------- ВЕРХНИЙ РЯД: СРЕДНИЕ ----------
st.title("40 week courses")

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Form Responses 1 — Average by S")
    if agg1.empty:
        st.info("Нет данных для выбранных фильтров.")
    else:
        chart1 = (
            alt.Chart(agg1)
              .mark_line(point=True)
              .encode(
                  x=alt.X("S:Q", title="S"),
                  y=alt.Y("avg_y:Q", title="Average G"),
                  tooltip=[
                      alt.Tooltip("S:Q", title="S"),
                      alt.Tooltip("avg_y:Q", title="Average G", format=".2f"),
                      alt.Tooltip("count:Q", title="Кол-во ответов")
                  ]
              )
              .properties(height=380)
        )
        st.altair_chart(chart1, use_container_width=True)

with col2:
    st.subheader("Form Responses 2 — Average by R")
    if agg2.empty:
        st.info("Нет данных для выбранных фильтров.")
    else:
        chart2 = (
            alt.Chart(agg2)
              .mark_line(point=True)
              .encode(
                  x=alt.X("R:Q", title="R"),
                  y=alt.Y("avg_y:Q", title="Average I"),
                  tooltip=[
                      alt.Tooltip("R:Q", title="R"),
                      alt.Tooltip("avg_y:Q", title="Average I", format=".2f"),
                      alt.Tooltip("count:Q", title="Кол-во ответов")
                  ]
              )
              .properties(height=380)
        )
        st.altair_chart(chart2, use_container_width=True)

# ---------- НИЖНИЙ РЯД: РАСПРЕДЕЛЕНИЯ (stacked bars, без бинов) ----------
st.markdown("---")
st.subheader(f"Распределение значений (гранулярность: {granularity.lower()})")

# Подписи периода для толще-столбцов (делаем ось категориальной)
def add_bucket_label(dff: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if dff.empty:
        return dff
    if granularity == "День":
        fmt = "%Y-%m-%d"
    elif granularity == "Неделя":
        # показываем старт недели
        fmt = "W%W (%Y-%m-%d)"
    else:  # "Месяц"
        fmt = "%Y-%m"
    dff["bucket_label"] = dff["bucket"].dt.strftime(fmt)
    return dff

df1_f = add_bucket_label(df1_f, granularity) if not df1_f.empty else df1_f
df2_f = add_bucket_label(df2_f, granularity) if not df2_f.empty else df2_f

col3, col4 = st.columns([1, 1])

with col3:
    st.markdown("**Form Responses 1 — распределение G (1–5)**")
    if df1_f.empty:
        st.info("Нет данных (FR1).")
    else:
        fr1_vals = [1, 2, 3, 4, 5]
        df1_counts = df1_f[df1_f["G"].isin(fr1_vals)].copy()
        df1_counts["G_str"] = df1_counts["G"].astype(int).astype(str)
        sort_values = [str(v) for v in fr1_vals]

        chart1_dist = (
            alt.Chart(df1_counts)
              .mark_bar(size=28)  # <-- толще столбцы
              .encode(
                  x=alt.X("bucket_label:N",
                          title="Период",
                          sort=alt.SortField(field="bucket", order="ascending")),
                  y=alt.Y("count():Q", title="Кол-во ответов"),
                  color=alt.Color("G_str:N", title="G", sort=sort_values),
                  order=alt.Order("G_str:N", sort="ascending"),
                  tooltip=[
                      alt.Tooltip("bucket_label:N", title="Период"),
                      alt.Tooltip("G_str:N", title="G"),
                      alt.Tooltip("count():Q", title="Кол-во ответов")
                  ]
              )
              .properties(height=380)
        )
        st.altair_chart(chart1_dist, use_container_width=True)

with col4:
    st.markdown("**Form Responses 2 — распределение I (1–10)**")
    if df2_f.empty:
        st.info("Нет данных (FR2).")
    else:
        fr2_vals = list(range(1, 11))  # 1..10
        df2_counts = df2_f[df2_f["I"].isin(fr2_vals)].copy()
        df2_counts["I_str"] = df2_counts["I"].astype(int).astype(str)
        sort_values2 = [str(v) for v in fr2_vals]

        chart2_dist = (
            alt.Chart(df2_counts)
              .mark_bar(size=28)  # <-- толще столбцы
              .encode(
                  x=alt.X("bucket_label:N",
                          title="Период",
                          sort=alt.SortField(field="bucket", order="ascending")),
                  y=alt.Y("count():Q", title="Кол-во ответов"),
                  color=alt.Color("I_str:N", title="I", sort=sort_values2),
                  order=alt.Order("I_str:N", sort="ascending"),
                  tooltip=[
                      alt.Tooltip("bucket_label:N", title="Период"),
                      alt.Tooltip("I_str:N", title="I"),
                      alt.Tooltip("count():Q", title="Кол-во ответов")
                  ]
              )
              .properties(height=380)
        )
        st.altair_chart(chart2_dist, use_container_width=True)

