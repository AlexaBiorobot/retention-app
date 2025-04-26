import streamlit as st
import json
st.set_page_config(
    page_title="Retention Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import timedelta, datetime

# === Constants ===
MAIN_SS_ID   = "1sZeAqY6dwdnEwPBg5vhOJDVXODyMuoV8OKMzBOviYQA"
MAIN_SHEET   = "auto"
LEADS_SS_ID  = "1SudB1YkPD0Tt7xkEiNJypRv0vb62BSdsCLrcrGqALAI"
LEADS_SHEET  = "Tutors"

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
    df.columns = (
        df.columns.astype(str)
          .str.strip()
          .str.lower()
          .str.replace(r"\s+", "_", regex=True)
    )

    # 3) очищаем и парсим первую дату
    df["first_lesson_date_teach"] = (
        df["first_lesson_date_teach"]
          .astype(str)
          .str.strip()
          .replace({"^\s*$": None}, regex=True)
    )
    df["first_lesson_date_dt"] = pd.to_datetime(
        df["first_lesson_date_teach"],
        format="%Y-%m-%d",
        errors="coerce"
    )
    df = df[~df["first_lesson_date_dt"].isna()]

    # 4) приводим к числу
    df["course_duration"] = pd.to_numeric(df["course_duration"], errors="coerce")
    df["first_lesson"]    = pd.to_numeric(df["first_lesson"],    errors="coerce")

    # === 5) рассчитываем периоды по вашей ARRAYFORMULA-логике ===
    lesson_counts = {
        32: (10,  8, 13),
        40: (13, 12, 14),
        64: (20, 20, 23),
        80: (26, 34, 19),
    }

    # 5.1) шаг в днях
    df["step"] = df["course_duration"].apply(
        lambda dur: 3.5 if dur in (64, 80) else 7
    )

    # 5.2) первый день первого периода
    df["1st_period_start"] = (
        df["first_lesson_date_dt"]
        - pd.to_timedelta(df["first_lesson"] * df["step"], unit="D")
        + pd.to_timedelta(df["step"], unit="D")
    )

    # 5.3) концы и начала периодов «от H»
    def compute_periods(row):
        dur = row["course_duration"]
        s   = row["step"]
        h   = row["1st_period_start"]

        # 1-й период
        if   dur == 32:
            end1 = h + pd.to_timedelta(10 * 7   - 7,    unit="D")
        elif dur == 40:
            end1 = h + pd.to_timedelta(13 * 7   - 7,    unit="D")
        elif dur == 64:
            end1 = h + pd.to_timedelta(20 * 3.5 - 3.5,  unit="D")
        elif dur == 80:
            end1 = h + pd.to_timedelta(26 * 3.5 - 3.5,  unit="D")
        else:
            end1 = pd.NaT

        # 2-й период
        lc1, lc2, lc3 = lesson_counts[dur]
        start2 = h + pd.to_timedelta(lc1 * s,         unit="D")
        end2   = start2 + pd.to_timedelta(lc2 * s - s, unit="D")

        # 3-й период
        start3 = h + pd.to_timedelta((lc1 + lc2) * s,  unit="D")
        if   dur == 32:
            end3 = start3 + pd.to_timedelta(13 * 7,     unit="D")
        elif dur == 40:
            end3 = start3 + pd.to_timedelta(14 * 7,     unit="D")
        elif dur == 64:
            end3 = start3 + pd.to_timedelta(23 * 3.5,   unit="D")
        elif dur == 80:
            end3 = start3 + pd.to_timedelta(19 * 3.5,   unit="D")
        else:
            end3 = pd.NaT

        return pd.Series(
            [end1, start2, end2, start3, end3],
            index=[
                "1st_period_end",
                "2nd_period_start", "2nd_period_end",
                "3rd_period_start", "3rd_period_end"
            ]
        )

    # 5.3.1) применяем compute_periods ко всем строкам
    df[
        [
            "1st_period_end",
            "2nd_period_start", "2nd_period_end",
            "3rd_period_start", "3rd_period_end"
        ]
    ] = df.apply(compute_periods, axis=1)
    
    # 5.3.2) приводим эти пять колонок в настоящий datetime64
    for c in [
        "1st_period_end","2nd_period_start","2nd_period_end",
        "3rd_period_start","3rd_period_end"
    ]:
        df[c] = pd.to_datetime(
            df[c],
            format="%d/%m/%Y",
            dayfirst=True,
            errors="coerce"
        )

    # 5.5) убираем временный столбец
    df.drop(columns="step", inplace=True)

    # 7) подтягиваем team_lead
    ws2 = client.open_by_key(LEADS_SS_ID).worksheet(LEADS_SHEET)
    leads = {
        row[0]: row[3]
        for row in ws2.get_all_values()[1:]
        if row[0]
    }
    df["team_lead"] = df["teacher_id"].astype(str).map(leads).fillna("")

    # 8) отфильтровываем one_time_replacement
    df = df[df["one_time_replacement"] == 0]

    return df

df = load_data_from_gsheet()

# === Sidebar filters ===
with st.sidebar:
    st.header("Filters")

    bo_ids   = st.multiselect("bo_id", df["bo_id"].unique(), default=None)
    teachers = st.multiselect("teacher_name", df["teacher_name"].unique(), default=None)
    leads    = st.multiselect("team_lead", df["team_lead"].unique(), default=None)
    courses  = st.multiselect("course", df["course"].unique(), default=None)
    groups   = st.multiselect("group_title", df["group_title"].unique(), default=None)

    p1 = st.multiselect("Period 1", df["period_1"].unique(), default=None)
    p2 = st.multiselect("Period 2", df["period_2"].unique(), default=None)
    p3 = st.multiselect("Period 3", df["period_3"].unique(), default=None)

    st.markdown("### Date ranges (end of period)")
    d1 = st.date_input("1st_period_end between", [])
    d2 = st.date_input("2nd_period_end between", [])
    d3 = st.date_input("3rd_period_end between", [])

# === Apply filters ===
dff = df.copy()

def apply_multiselect(col, sel):
    return dff[dff[col].isin(sel)] if sel else dff

if bo_ids:   dff = apply_multiselect("bo_id", bo_ids)
if teachers: dff = apply_multiselect("teacher_name", teachers)
if leads:    dff = apply_multiselect("team_lead", leads)
if courses:  dff = apply_multiselect("course", courses)
if groups:   dff = apply_multiselect("group_title", groups)

mask = pd.Series(False, index=dff.index)
for col, sel in [("period_1", p1), ("period_2", p2), ("period_3", p3)]:
    if sel:
        mask |= dff[col].isin(sel)
if mask.any():
    dff = dff[mask]

def apply_date_range(df_, col, dr):
    if len(dr) == 2:
        start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
        return df_[
            (pd.to_datetime(df_[col], dayfirst=True) >= start) &
            (pd.to_datetime(df_[col], dayfirst=True) <= end)
        ]
    return df_

dff = apply_date_range(dff, "1st_period_end", d1)
dff = apply_date_range(dff, "2nd_period_end", d2)
dff = apply_date_range(dff, "3rd_period_end", d3)

for col in dff.select_dtypes("number").columns:
    dff[col] = dff[col].round(2)

hide_cols = ["first_lesson_date_teach", "dropp", "one_time_replacement", "1st_period_start", "2nd_period_start", "3rd_period_start"]
dff = dff.drop(columns=[c for c in hide_cols if c in dff.columns])

# Форматируем колонки-дат в строки только для отображения
dff_display = dff.copy()
for c in [
    "1st_period_end","2nd_period_start","2nd_period_end",
    "3rd_period_start","3rd_period_end"
]:
    dff_display[c] = dff_display[c].dt.strftime("%d/%m/%Y")

# Выводим уже отформатированную копию
st.dataframe(dff_display, use_container_width=True)

@st.cache_data
def to_excel(data: pd.DataFrame):
    return data.to_excel(index=False)

if st.button("Export to Excel"):
    tmp = to_excel(dff)
    st.download_button("Download XLSX", tmp, file_name="report.xlsx")
