import os
import streamlit as st
import json
st.set_page_config(
    page_title="Retention Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)
import pandas as pd
import gspread
import io
from oauth2client.service_account import ServiceAccountCredentials
from datetime import timedelta, datetime

# === Constants ===
MAIN_SS_ID = os.getenv("MAIN_SS_ID") or st.secrets["MAIN_SS_ID"]
MAIN_SHEET   = "auto"
LEADS_SS_ID  = "1SudB1YkPD0Tt7xkEiNJypRv0vb62BSdsCLrcrGqALAI"
LEADS_SHEET  = "Tutors"

@st.cache_data
def load_data_from_gsheet():
    import os, json
    from oauth2client.service_account import ServiceAccountCredentials
    import gspread

    # === 1) авторизационный скоуп ===
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]

    # === 2) читаем JSON ключ ===
    # сначала из переменной окружения GCP_SERVICE_ACCOUNT
    sa_json = os.getenv("GCP_SERVICE_ACCOUNT")
    if not sa_json:
        # fallback на локальный streamlit secret
        sa_json = st.secrets["GCP_SERVICE_ACCOUNT"]
    sa_info = json.loads(sa_json)

    # === 3) собираем credentials и клиент ===
    creds  = ServiceAccountCredentials.from_json_keyfile_dict(sa_info, scope)
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

    # 5.5) убираем временный столбец — убедитесь, что этот вызов НЕ внутри цикла!
    df.drop(columns="step", inplace=True)

    # === 6) подхватываем оверрайды из листа "tech" ===
    import numpy as np
    
    TECH_SS_ID = "1wJvMIf62izX10-r_-B1QtfKWRzobZHCH8dufVsCCVko"
    TECH_SHEET = "tech"
    ws3 = client.open_by_key(TECH_SS_ID).worksheet(TECH_SHEET)
    
    # 1) загружаем исходную тех.таблицу
    all_vals = ws3.get_all_values()
    headers  = all_vals[0]
    tech_df  = pd.DataFrame(all_vals[1:], columns=headers).fillna("")
    
    # 2) разбиваем на три блока и ставим флаги наличия записи
    ov1 = tech_df.iloc[:, 0:3].copy(); ov1.columns = ["teacher_id","bo_id","ov1"]; ov1["has_ov1"] = True
    ov2 = tech_df.iloc[:, 3:6].copy(); ov2.columns = ["teacher_id","bo_id","ov2"]; ov2["has_ov2"] = True
    ov3 = tech_df.iloc[:, 6:9].copy(); ov3.columns = ["teacher_id","bo_id","ov3"]; ov3["has_ov3"] = True
    
    # 3) приводим override-колонки к числам и чистим ключи
    for df_ov, col, flag in [(ov1,"ov1","has_ov1"), (ov2,"ov2","has_ov2"), (ov3,"ov3","has_ov3")]:
        df_ov[col] = pd.to_numeric(df_ov[col], errors="coerce")
        df_ov["teacher_id"] = df_ov["teacher_id"].astype(str).str.strip()
        df_ov["bo_id"]      = df_ov["bo_id"].astype(str).str.strip()
    
    # 4) чистим ключи в основном df (до merge)
    df["teacher_id"] = df["teacher_id"].astype(str).str.strip()
    df["bo_id"]      = df["bo_id"].astype(str).str.strip()
    
    # 5) первый период
    df = df.merge(ov1[["teacher_id","bo_id","ov1","has_ov1"]],
                  on=["teacher_id","bo_id"], how="left")
    # если запись есть, но ov1 пустой — очищаем
    df.loc[df["has_ov1"] & df["ov1"].isna(), "period_1"] = ""
    # если ov1 число — подставляем
    df.loc[df["ov1"].notna(),        "period_1"] = df.loc[df["ov1"].notna(), "ov1"]
    
    # 6) второй период
    df = df.merge(ov2[["teacher_id","bo_id","ov2","has_ov2"]],
                  on=["teacher_id","bo_id"], how="left")
    df.loc[df["has_ov2"] & df["ov2"].isna(), "period_2"] = ""
    df.loc[df["ov2"].notna(),        "period_2"] = df.loc[df["ov2"].notna(), "ov2"]
    
    # 7) третий период
    df = df.merge(ov3[["teacher_id","bo_id","ov3","has_ov3"]],
                  on=["teacher_id","bo_id"], how="left")
    df.loc[df["has_ov3"] & df["ov3"].isna(), "period_3"] = ""
    df.loc[df["ov3"].notna(),        "period_3"] = df.loc[df["ov3"].notna(), "ov3"]
    
    # 8) удаляем вспомогательные столбцы
    df.drop(columns=[
        "ov1","has_ov1",
        "ov2","has_ov2",
        "ov3","has_ov3"
    ], inplace=True)

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
    teacher_ids = st.multiselect("teacher_id", df["teacher_id"].unique(), default=None)
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
if teacher_ids: dff = apply_multiselect("teacher_id", teacher_ids)
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

st.dataframe(
    dff,
    use_container_width=True
)

@st.cache_data
def to_excel(data: pd.DataFrame):
    output = io.BytesIO()
    data.to_excel(output, index=False)
    output.seek(0)
    return output

if st.button("Export to Excel"):
    tmp = to_excel(dff)
    st.download_button("Download XLSX", tmp, file_name="report.xlsx")
