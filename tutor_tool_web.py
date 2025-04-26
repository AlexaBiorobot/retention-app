import streamlit as st
import json
st.set_page_config(
    page_title="Retention Tool",
    layout="wide",
    initial_sidebar_state="expanded")
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import timedelta, datetime

# === Constants ===
MAIN_SS_ID           = "1sZeAqY6dwdnEwPBg5vhOJDVXODyMuoV8OKMzBOviYQA"
MAIN_SHEET           = "auto"
LEADS_SS_ID          = "1SudB1YkPD0Tt7xkEiNJypRv0vb62BSdsCLrcrGqALAI"
LEADS_SHEET          = "Tutors"

@st.cache_data
def load_data_from_gsheet():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    # берём JSON сервис-аккаунта из секрета
    sa_json = st.secrets["GCP_SERVICE_ACCOUNT"]
    sa_info = json.loads(sa_json)
    creds   = ServiceAccountCredentials.from_json_keyfile_dict(sa_info, scope)

    client = gspread.authorize(creds)

    # 1) читаем основной лист
    ws1 = client.open_by_key(MAIN_SS_ID).worksheet(MAIN_SHEET)
    df  = pd.DataFrame(ws1.get_all_records())

    # 2) нормализуем имена колонок
    df.columns = (df.columns.astype(str)
                    .str.strip()
                    .str.lower()
                    .str.replace(r"\s+","_", regex=True))

    # 3) очищаем и парсим первую дату
    df["first_lesson_date_teach"] = (
        df["first_lesson_date_teach"]
          .astype(str)          # на всякий случай приводим к str
          .str.strip()          # убираем лишние пробелы
          .replace({"^\s*$": None}, regex=True)  # пустые строки → None
    )
    df["first_lesson_date_dt"] = pd.to_datetime(
        df["first_lesson_date_teach"],
        dayfirst=True,
        errors="coerce"          # некорректные превратятся в NaT
    )
    # --- отсечём строки, где парсинг не сработал ---
    df = df[~df["first_lesson_date_dt"].isna()]

    # 4) приводим к числу
    df["course_duration"] = pd.to_numeric(df["course_duration"], errors="coerce")
    df["first_lesson"]    = pd.to_numeric(df["first_lesson"],    errors="coerce")

    # === 5) рассчитываем границы периодов точно по ARRAYFORMULA ===
    # шаг в днях: 7 для 32/40, иначе 3.5
    df["step"] = df["course_duration"].apply(lambda x: 7 if x in (32, 40) else 3.5)

    # 1-й период начинается в день first_lesson_date_dt + (first_lesson−1)*step
    df["1st_period_start"] = (
        df["first_lesson_date_dt"]
        + pd.to_timedelta((df["first_lesson"] - 1) * df["step"], unit="D")
    )
    # 1-й период длится 10 шагов минус один (т.е. exactly как в IF(F=32, H+(10*7)-7…))
    df["1st_period_end"] = (
        df["first_lesson_date_dt"]
        + pd.to_timedelta((df["first_lesson"] - 1 + 10) * df["step"] - df["step"], unit="D")
    )

    # 2-й период стартует сразу после 1-го: first + 10*step
    df["2nd_period_start"] = (
        df["first_lesson_date_dt"]
        + pd.to_timedelta(10 * df["step"], unit="D")
    )
    # 2-й период длится 8 шагов минус один
    df["2nd_period_end"] = (
        df["first_lesson_date_dt"]
        + pd.to_timedelta(10 * df["step"] + 8 * df["step"] - df["step"], unit="D")
    )

    # 3-й период стартует сразу после 2-го: first + (10+8)*step
    df["3rd_period_start"] = (
        df["first_lesson_date_dt"]
        + pd.to_timedelta((10 + 8) * df["step"], unit="D")
    )
    # 3-й период длится 13 шагов минус один
    df["3rd_period_end"] = (
        df["first_lesson_date_dt"]
        + pd.to_timedelta(10 * df["step"] + 8 * df["step"] + 13 * df["step"] - df["step"], unit="D")
    )

    # уберём вспомогательный столбец
    df = df.drop(columns="step")

    # 6) форматируем концы периодов и первую дату в dd/mm/YYYY
    for c in ["1st_period_end","2nd_period_end","3rd_period_end"]:
        df[c] = pd.to_datetime(df[c]).dt.strftime("%d/%m/%Y")
    df["first_lesson_date_teach"] = df["first_lesson_date_dt"].dt.strftime("%d/%m/%Y")

    # 7) подтягиваем team_lead
    ws2 = client.open_by_key(LEADS_SS_ID).worksheet(LEADS_SHEET)
    leads = {row[0]: row[3] for row in ws2.get_all_values()[1:] if row[0]}
    df["team_lead"] = df["teacher_id"].astype(str).map(leads).fillna("")

    # 8) отфильтровываем one_time_replacement
    df = df[df["one_time_replacement"] == 0]

    return df

df = load_data_from_gsheet()

# === Sidebar filters ===
with st.sidebar:
    st.header("Filters")

    # text filters
    bo_ids       = st.multiselect("bo_id", df["bo_id"].unique(), default=None)
    teachers     = st.multiselect("teacher_name", df["teacher_name"].unique(), default=None)
    leads        = st.multiselect("team_lead", df["team_lead"].unique(), default=None)
    courses      = st.multiselect("course", df["course"].unique(), default=None)
    groups       = st.multiselect("group_title", df["group_title"].unique(), default=None)

    # period selections
    p1 = st.multiselect("Period 1", df["period_1"].unique(), default=None)
    p2 = st.multiselect("Period 2", df["period_2"].unique(), default=None)
    p3 = st.multiselect("Period 3", df["period_3"].unique(), default=None)

    # date ranges
    st.markdown("### Date ranges (end of period)")
    d1 = st.date_input("1st_period_end between", [])
    d2 = st.date_input("2nd_period_end between", [])
    d3 = st.date_input("3rd_period_end between", [])

# === Apply filters ===
dff = df.copy()
def apply_multiselect(col, sel):
    if sel:
        return dff[dff[col].isin(sel)]
    return dff

# apply one by one
if bo_ids:  dff = dff[dff["bo_id"].isin(bo_ids)]
if teachers: dff = dff[dff["teacher_name"].isin(teachers)]
if leads:   dff = dff[dff["team_lead"].isin(leads)]
if courses: dff = dff[dff["course"].isin(courses)]
if groups:  dff = dff[dff["group_title"].isin(groups)]

# periods OR logic
mask = pd.Series(False, index=dff.index)
for col, sel in [("period_1",p1),("period_2",p2),("period_3",p3)]:
    if sel:
        mask |= dff[col].isin(sel)
if mask.any():
    dff = dff[mask]

# date ranges
def apply_date_range(df_, col, dr):
    if len(dr)==2:
        start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
        return df_[(pd.to_datetime(df_[col], dayfirst=True) >= start) &
                   (pd.to_datetime(df_[col], dayfirst=True) <= end)]
    return df_

dff = apply_date_range(dff, "1st_period_end", d1)
dff = apply_date_range(dff, "2nd_period_end", d2)
dff = apply_date_range(dff, "3rd_period_end", d3)

# === Main table ===
for col in dff.select_dtypes("number").columns:
    dff[col] = dff[col].round(2)

hide_cols = [
    "1st_period_start","2nd_period_start","3rd_period_start",
    "first_lesson_date_teach","dropp","one_time_replacement"
]
dff = dff.drop(columns=[c for c in hide_cols if c in dff.columns])

# И выводим без .style, чтобы не было лимита Styler:
st.dataframe(
    dff,
    use_container_width=True
)

# === Export button ===
@st.cache_data
def to_excel(data: pd.DataFrame):
    return data.to_excel(index=False)

if st.button("Export to Excel"):
    tmp = to_excel(dff)
    st.download_button("Download XLSX", tmp, file_name="report.xlsx")
