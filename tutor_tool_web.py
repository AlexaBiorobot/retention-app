import os
import io
import json
import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(
    page_title="Retention Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Constants ===
MAIN_SS_ID = os.getenv("MAIN_SS_ID") or st.secrets["MAIN_SS_ID"]
MAIN_SHEET = "auto"

LEADS_SS_ID = "1SudB1YkPD0Tt7xkEiNJypRv0vb62BSdsCLrcrGqALAI"
LEADS_SHEET = "Tutors"


@st.cache_data
def load_data_from_gsheet():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]

    sa_json = os.getenv("GCP_SERVICE_ACCOUNT") or st.secrets["GCP_SERVICE_ACCOUNT"]
    sa_info = json.loads(sa_json)

    creds = ServiceAccountCredentials.from_json_keyfile_dict(sa_info, scope)
    client = gspread.authorize(creds)

    # 1) читаем таб auto целиком (чтобы гарантированно забрать R/S/T/U)
    ws = client.open_by_key(MAIN_SS_ID).worksheet(MAIN_SHEET)
    all_vals = ws.get_all_values()
    if not all_vals or len(all_vals) < 2:
        return pd.DataFrame()

    headers = all_vals[0]
    max_len = max(len(r) for r in all_vals)

    # добиваем заголовки, если вдруг короче
    if len(headers) < max_len:
        headers = headers + [f"col_{i+1}" for i in range(len(headers), max_len)]

    rows = [r + [""] * (max_len - len(r)) for r in all_vals[1:]]
    df = pd.DataFrame(rows, columns=headers)

    # 2) нормализуем имена колонок
    df.columns = (
        df.columns.astype(str)
          .str.strip()
          .str.lower()
          .str.replace(r"\s+", "_", regex=True)
    )

    # 3) гарантируем, что нужные колонки существуют (fallback по позициям R/S/T/U)
    # R=18, S=19, T=20, U=21 (1-based) => 17/18/19/20 (0-based)
    pos_map = {
        "last_lesson_date": 17,     # col R
        "1st_period_end": 18,       # col S
        "period2_end_date": 19,     # col T
        "period3_end_date": 20,     # col U
    }
    for col_name, idx in pos_map.items():
        if col_name not in df.columns:
            if df.shape[1] > idx:
                df[col_name] = df.iloc[:, idx]
            else:
                df[col_name] = ""

    # 4) парсим даты (last_lesson_date тоже, просто для нормального отображения)
    for c in ["last_lesson_date", "1st_period_end", "period2_end_date", "period3_end_date"]:
        df[c] = (
            df[c].astype(str).str.strip()
              .replace({"^\s*$": None}, regex=True)
        )
        df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)

    # 5) team_lead
    ws2 = client.open_by_key(LEADS_SS_ID).worksheet(LEADS_SHEET)
    leads = {
        row[0]: row[3]
        for row in ws2.get_all_values()[1:]
        if row and row[0]
    }
    if "teacher_id" in df.columns:
        df["teacher_id"] = df["teacher_id"].astype(str).str.strip()
        df["team_lead"] = df["teacher_id"].map(leads).fillna("")
    else:
        df["team_lead"] = ""

    # 6) one_time_replacement = 0
    if "one_time_replacement" in df.columns:
        df["one_time_replacement"] = pd.to_numeric(df["one_time_replacement"], errors="coerce").fillna(0)
        df = df[df["one_time_replacement"] == 0]

    return df


df = load_data_from_gsheet()

if df.empty:
    st.warning("Данных нет (лист пустой или не удалось прочитать).")
    st.stop()

# === Sidebar filters ===
with st.sidebar:
    st.header("Filters")

    def ms(label, col):
        return st.multiselect(label, df[col].dropna().unique() if col in df.columns else [], default=None)

    bo_ids = ms("bo_id", "bo_id")
    teachers = ms("teacher_name", "teacher_name")
    teacher_ids = ms("teacher_id", "teacher_id")
    leads = ms("team_lead", "team_lead")
    courses = ms("course", "course")
    groups = ms("group_title", "group_title")

    p1 = ms("Period 1", "period_1")
    p2 = ms("Period 2", "period_2")
    p3 = ms("Period 3", "period_3")

    st.markdown("### Date ranges (end of period)")
    d1 = st.date_input("1st_period_end between", [])
    d2 = st.date_input("period2_end_date between", [])
    d3 = st.date_input("period3_end_date between", [])

# === Apply filters ===
dff = df.copy()

def apply_multiselect(df_, col, sel):
    if sel and col in df_.columns:
        return df_[df_[col].isin(sel)]
    return df_

dff = apply_multiselect(dff, "bo_id", bo_ids)
dff = apply_multiselect(dff, "teacher_name", teachers)
dff = apply_multiselect(dff, "teacher_id", teacher_ids)
dff = apply_multiselect(dff, "team_lead", leads)
dff = apply_multiselect(dff, "course", courses)
dff = apply_multiselect(dff, "group_title", groups)

# period_1/2/3 — OR логика (как было)
mask = pd.Series(False, index=dff.index)
for col, sel in [("period_1", p1), ("period_2", p2), ("period_3", p3)]:
    if sel and col in dff.columns:
        mask |= dff[col].isin(sel)
if mask.any():
    dff = dff[mask]

def apply_date_range(df_, col, dr):
    if col not in df_.columns:
        return df_
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        start = pd.to_datetime(dr[0])
        end = pd.to_datetime(dr[1])
        col_dt = pd.to_datetime(df_[col], errors="coerce")
        return df_[(col_dt >= start) & (col_dt <= end)]
    return df_

dff = apply_date_range(dff, "1st_period_end", d1)
dff = apply_date_range(dff, "period2_end_date", d2)
dff = apply_date_range(dff, "period3_end_date", d3)

# округление чисел (если есть)
for col in dff.select_dtypes("number").columns:
    dff[col] = dff[col].round(2)

# скрываем колонки, которые больше не нужны (если они есть в ресурсе)
hide_cols = [
    "first_lesson_date_teach",
    "first_lesson_date_dt",
    "course_duration",
    "first_lesson",
    "step",
    "1st_period_start",
    "2nd_period_start",
    "3rd_period_start",
    "dropp",
    "one_time_replacement",
]
dff = dff.drop(columns=[c for c in hide_cols if c in dff.columns], errors="ignore")

st.dataframe(dff, use_container_width=True)

@st.cache_data
def to_excel(data: pd.DataFrame):
    output = io.BytesIO()
    data.to_excel(output, index=False)
    output.seek(0)
    return output

if st.button("Export to Excel"):
    tmp = to_excel(dff)
    st.download_button("Download XLSX", tmp, file_name="report.xlsx")
