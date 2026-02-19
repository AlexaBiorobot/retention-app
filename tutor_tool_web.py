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

    ws = client.open_by_key(MAIN_SS_ID).worksheet(MAIN_SHEET)

    # ВАЖНО: UNFORMATTED_VALUE => даты придут числами (serial), а не строками
    vals = ws.get("A1:U", value_render_option="UNFORMATTED_VALUE")
    if not vals or len(vals) < 2:
        return pd.DataFrame()

    WIDTH = 21  # A..U
    vals = [row + [""] * (WIDTH - len(row)) for row in vals]

    headers = vals[0]
    if len(headers) < WIDTH:
        headers = headers + [f"col_{i+1}" for i in range(len(headers), WIDTH)]

    df = pd.DataFrame(vals[1:], columns=headers[:WIDTH])

    # нормализуем имена колонок (для фильтров/привычного вида)
    df.columns = (
        df.columns.astype(str)
          .str.strip()
          .str.lower()
          .str.replace(r"\s+", "_", regex=True)
    )

    # --- R/S/T/U по позициям внутри A..U ---
    # A=0 ... R=17, S=18, T=19, U=20
    def normalize_serial(x):
        """Оставляем как число/строку. Пусто => ''.
        Если 45000.0 => 45000 (int), чтобы красивее выглядело.
        """
        if x is None or x == "":
            return ""
        if isinstance(x, float) and x.is_integer():
            return int(x)
        return x

    df["last_lesson_date"] = df.iloc[:, 17].apply(normalize_serial)  # R — ЧИСЛОМ, БЕЗ ПАРСИНГА
    # serial number -> datetime (Google Sheets / Excel)
    def serial_to_datetime(x):
        if x is None or x == "":
            return pd.NaT
        if isinstance(x, (int, float)):
            return pd.to_datetime(x, unit="D", origin="1899-12-30", errors="coerce")
        return pd.to_datetime(str(x).strip(), errors="coerce", dayfirst=True)
    
    # A=0 ... R=17, S=18, T=19, U=20
    df["last_lesson_date"] = df.iloc[:, 17].apply(serial_to_datetime)   # R -> datetime
    df["period1_end_date"] = df.iloc[:, 18].apply(serial_to_datetime)   # S
    df["period2_end_date"] = df.iloc[:, 19].apply(serial_to_datetime)   # T
    df["period3_end_date"] = df.iloc[:, 20].apply(serial_to_datetime)   # U

    # first_lesson_date_teach -> datetime (если колонка есть в загруженном диапазоне)
    if "first_lesson_date_teach" in df.columns:
        df["first_lesson_date_teach"] = df["first_lesson_date_teach"].apply(serial_to_datetime)

        # оставить только дату (обнулить время)
    date_cols = [
        "last_lesson_date",
        "period1_end_date",
        "period2_end_date",
        "period3_end_date",
        "first_lesson_date_teach",
    ]
    
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.normalize()


    # убираем дубли, если в листе есть другие колонки с похожими названиями
    df.drop(columns=[c for c in ["1st_period_end", "period2_end_date_date", "period3_end_date_date"] if c in df.columns],
            inplace=True, errors="ignore")

    # team_lead
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

    # one_time_replacement = 0
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
    d1 = st.date_input("period1_end_date between", [])
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

# period_1/2/3 — OR логика
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

dff = apply_date_range(dff, "period1_end_date", d1)
dff = apply_date_range(dff, "period2_end_date", d2)
dff = apply_date_range(dff, "period3_end_date", d3)

# скрываем мусорные/технические колонки
hide_cols = ["dropp", "one_time_replacement"]
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
