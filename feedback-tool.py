import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import altair as alt
import json
import string

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
    letters = list(string.ascii_uppercase)[:num_cols]  # A..Z (хватает для A..Z)
    return pd.DataFrame(values[1:], columns=letters)

# ---------- ДАННЫЕ ----------
df1 = load_sheet_as_letter_df("Form Responses 1")   # A=date, N=course, S=x, G=y
df2 = load_sheet_as_letter_df("Form Responses 2")   # A=date, M=course, R=x, I=y

# приведение типов
for df, date_col, x_col, y_col in [
    (df1, "A", "S", "G"),
    (df2, "A", "R", "I"),
]:
    if df.empty:
        continue
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

# ---------- ФИЛЬТРЫ (в сайдбаре, раздельно) ----------
st.sidebar.header("Фильтры")

# --- FR1 ---
st.sidebar.subheader("Form Responses 1")
courses_all_1 = sorted([c for c in df1["N"].dropna().unique()]) if not df1.empty else []
if "courses_selected_fr1" not in st.session_state:
    st.session_state["courses_selected_fr1"] = courses_all_1.copy()
c1a, c1b = st.sidebar.columns(2)
if c1a.button("Select all (FR1)"):
    st.session_state["courses_selected_fr1"] = courses_all_1.copy()
    st.rerun()
if c1b.button("Clear (FR1)"):
    st.session_state["courses_selected_fr1"] = []
    st.rerun()
selected_courses_1 = st.sidebar.multiselect(
    "Курсы (N)",
    options=courses_all_1,
    default=st.session_state["courses_selected_fr1"],
    key="courses_selected_fr1"
)
if not df1.empty:
    min_date_1, max_date_1 = df1["A"].min(), df1["A"].max()
else:
    min_date_1 = max_date_1 = None
date_range_1 = st.sidebar.date_input(
    "Дата фидбека (A) — FR1",
    [] if (pd.isna(min_date_1) if min_date_1 is not None else True) else [min_date_1.date(), max_date_1.date()]
)

# --- FR2 ---
st.sidebar.subheader("Form Responses 2")
courses_all_2 = sorted([c for c in df2["M"].dropna().unique()]) if not df2.empty else []
if "courses_selected_fr2" not in st.session_state:
    st.session_state["courses_selected_fr2"] = courses_all_2.copy()
c2a, c2b = st.sidebar.columns(2)
if c2a.button("Select all (FR2)"):
    st.session_state["courses_selected_fr2"] = courses_all_2.copy()
    st.rerun()
if c2b.button("Clear (FR2)"):
    st.session_state["courses_selected_fr2"] = []
    st.rerun()
selected_courses_2 = st.sidebar.multiselect(
    "Курсы (M)",
    options=courses_all_2,
    default=st.session_state["courses_selected_fr2"],
    key="courses_selected_fr2"
)
if not df2.empty:
    min_date_2, max_date_2 = df2["A"].min(), df2["A"].max()
else:
    min_date_2 = max_date_2 = None
date_range_2 = st.sidebar.date_input(
    "Дата фидбека (A) — FR2",
    [] if (pd.isna(min_date_2) if min_date_2 is not None else True) else [min_date_2.date(), max_date_2.date()]
)

# ---------- ФУНКЦИИ ФИЛЬТРАЦИИ/АГРЕГАЦИИ ----------
def apply_filters_and_aggregate(df: pd.DataFrame, course_col: str, date_col: str,
                                x_col: str, y_col: str,
                                selected_courses, date_range):
    if df.empty:
        return pd.DataFrame(columns=[x_col, "avg_y", "count"])
    dff = df.copy()

    # по курсам
    if selected_courses:
        dff = dff[dff[course_col].isin(selected_courses)]
    else:
        return pd.DataFrame(columns=[x_col, "avg_y", "count"])

    # по датам
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1])
        dff = dff[(dff[date_col] >= start_dt) & (dff[date_col] <= end_dt)]
    elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
        only_dt = pd.to_datetime(date_range[0])
        dff = dff[dff[date_col].dt.date == only_dt.date()]

    grp = dff.dropna(subset=[x_col, y_col])
    if grp.empty:
        return pd.DataFrame(columns=[x_col, "avg_y", "count"])

    agg = (grp.groupby(x_col, as_index=False)
             .agg(avg_y=(y_col, "mean"),
                  count=(y_col, "size"))
             .sort_values(x_col))
    return agg

agg1 = apply_filters_and_aggregate(df1, course_col="N", date_col="A",
                                   x_col="S", y_col="G",
                                   selected_courses=selected_courses_1,
                                   date_range=date_range_1)

agg2 = apply_filters_and_aggregate(df2, course_col="M", date_col="A",
                                   x_col="R", y_col="I",
                                   selected_courses=selected_courses_2,
                                   date_range=date_range_2)

# ---------- ЛЕЙАУТ: ДВА ГРАФИКА СПРАВА/СЛЕВА ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("40 week courses — FR1")
    if agg1.empty:
        st.info("Нет данных для выбранных фильтров (FR1).")
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
        )
        st.altair_chart(chart1, use_container_width=True)

with col2:
    st.subheader("40 week courses — FR2")
    if agg2.empty:
        st.info("Нет данных для выбранных фильтров (FR2).")
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
        )
        st.altair_chart(chart2, use_container_width=True)
