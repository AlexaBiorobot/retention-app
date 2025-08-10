import os
import json
import io
import re
import numpy as np
import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ==== Page / UX ====  (ДОЛЖНО быть самым первым вызовом Streamlit)
st.set_page_config(
    page_title="Disbanding | Initial Export",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.write("Streamlit version:", st.__version__)

# ==== Constants ====
DEFAULT_SHEET_ID = "1Jbb4p1cZCo67ZRiW5cmFUq-c9ijo5VMH_hFFMVYeJk4"
DEFAULT_WS_NAME  = "data"

EXT_GROUPS_SS_ID = "1u_NwMt3CVVgozm04JGmccyTsNZnZGiHjG5y0Ko3YdaY"
EXT_GROUPS_WS    = "Groups & Teachers"

RATING_SS_ID = "1HItT2-PtZWoldYKL210hCQOLg3rh6U1Qj6NWkBjDjzk"
RATING_WS    = "Rating"

SHEET_ID = os.getenv("GSHEET_ID") or st.secrets.get("GSHEET_ID", DEFAULT_SHEET_ID)
WS_NAME  = os.getenv("GSHEET_WS") or st.secrets.get("GSHEET_WS", DEFAULT_WS_NAME)

SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]


def _authorize_client():
    sa_json = os.getenv("GCP_SERVICE_ACCOUNT") or st.secrets.get("GCP_SERVICE_ACCOUNT")
    if not sa_json:
        st.error("Не найден сервисный ключ. Добавь GCP_SERVICE_ACCOUNT в Secrets или ENV.")
        st.stop()
    try:
        sa_info = json.loads(sa_json)
    except Exception:
        st.error("GCP_SERVICE_ACCOUNT должен быть JSON-строкой (а не объектом).")
        st.stop()
    creds  = ServiceAccountCredentials.from_json_keyfile_dict(sa_info, SCOPE)
    client = gspread.authorize(creds)
    return client


@st.cache_data(show_spinner=False, ttl=300)
def load_sheet_df(sheet_id: str, worksheet_name: str = "data") -> pd.DataFrame:
    client = _authorize_client()
    sh = client.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet_name)
    values = ws.get(
        "A:R",
        value_render_option="UNFORMATTED_VALUE",
        date_time_render_option="FORMATTED_STRING",
    )
    if not values:
        return pd.DataFrame()
    header = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=header)
    df = df.replace({"": pd.NA})
    return df


def adjust_local_time_minus_3(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    col = None
    for c in df.columns:
        name = str(c).strip().lower().replace("_", " ")
        if name == "local time":
            col = c
            break
    if col is None:
        if len(df.columns) >= 9:
            col = df.columns[8]
        else:
            return df

    s = df[col].astype(str).str.strip()
    time_only_mask = s.str.match(r"^\d{1,2}:\d{2}(:\d{2})?$", na=False)

    def _shift_time_str(v: str) -> str:
        parts = v.split(":")
        h = int(parts[0]); m = int(parts[1]); sec = int(parts[2]) if len(parts) > 2 else None
        h = (h - 3) % 24
        return f"{h:02d}:{m:02d}" + (f":{sec:02d}" if sec is not None else "")

    out = pd.Series(pd.NA, index=df.index, dtype="object")
    out.loc[time_only_mask] = s.loc[time_only_mask].apply(_shift_time_str)

    dt_mask = (~time_only_mask) & s.ne("")
    if dt_mask.any():
        dt = pd.to_datetime(s[dt_mask], errors="coerce", dayfirst=False)
        miss = dt.isna()
        if miss.any():
            dt2 = pd.to_datetime(s[dt_mask][miss], errors="coerce", dayfirst=True)
            dt.loc[miss] = dt2
        dt = dt - pd.Timedelta(hours=3)
        out.loc[dt_mask] = dt.dt.strftime("%Y-%m-%d %H:%M")

    df = df.copy()
    df[col] = out.where(out.notna(), df[col])
    return df


@st.cache_data(show_spinner=False, ttl=300)
def load_group_age_map(sheet_id: str = EXT_GROUPS_SS_ID, worksheet_name: str = EXT_GROUPS_WS) -> dict:
    client = _authorize_client()
    ws = client.open_by_key(sheet_id).worksheet(worksheet_name)
    vals = ws.get("A:E")
    if not vals or len(vals) < 2:
        return {}
    rows = vals[1:]
    mapping = {}
    for r in rows:
        if len(r) >= 5:
            key = str(r[0]).strip()
            val = r[4]
            if key:
                mapping[key] = val
    return mapping


def replace_group_age_from_map(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    if df.empty or not mapping:
        return df.copy()
    dff = df.copy()
    colB = None
    for c in dff.columns:
        if str(c).strip().lower().replace("_", " ") in ("b","group id","group","group title","group_name","group name"):
            colB = c; break
    if colB is None:
        colB = dff.columns[1] if len(dff.columns) >= 2 else None
    colG = None
    for c in dff.columns:
        if str(c).strip().lower().replace("_", " ") == "group age":
            colG = c; break
    if colG is None:
        colG = dff.columns[6] if len(dff.columns) >= 7 else None
    if colB is None or colG is None:
        return dff
    keys = dff[colB].astype(str).str.strip()
    new_vals = keys.map(lambda k: mapping.get(k, pd.NA))
    dff[colG] = new_vals.where(new_vals.notna() & (new_vals.astype(str).str.strip() != ""), dff[colG])
    return dff


@st.cache_data(show_spinner=False, ttl=300)
def load_rating_bp_map(sheet_id: str = RATING_SS_ID, worksheet_name: str = RATING_WS) -> dict:
    client = _authorize_client()
    ws = client.open_by_key(sheet_id).worksheet(worksheet_name)
    vals = ws.get(
        "A:BP",
        value_render_option="UNFORMATTED_VALUE",
        date_time_render_option="FORMATTED_STRING",
    )
    if not vals or len(vals) < 2:
        return {}
    mapping = {}
    for r in vals[1:]:
        a  = str(r[0]).strip() if len(r) >= 1  else ""
        bp = r[67]              if len(r) >= 68 else None  # BP = 68-я колонка
        if a:
            mapping[a] = bp
    return mapping


def add_rating_bp_by_O(df: pd.DataFrame, mapping: dict, new_col_name: str = "Rating_BP") -> pd.DataFrame:
    if df.empty or not mapping:
        return df.copy()
    if len(df.columns) < 15:
        return df.copy()
    colO = df.columns[14]  # O
    keys = df[colO].astype(str).str.strip()
    out = df.copy()
    name = new_col_name
    while name in out.columns:
        name += "_x"
    out[name] = keys.map(lambda k: mapping.get(k, pd.NA))
    return out


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if len(df.columns) < 18:
        st.error("Ожидалось минимум 18 колонок (до R). Проверь диапазон A:R и заголовки.")
        st.stop()

    colD, colK = df.columns[3], df.columns[10]
    colL, colM = df.columns[11], df.columns[12]
    colP, colQ, colR = df.columns[15], df.columns[16], df.columns[17]

    d_active = df[colD].astype(str).str.strip().str.lower() == "active"

    k_num = pd.to_numeric(df[colK], errors="coerce")
    k_ok = k_num.notna() & (k_num > 3) & (k_num < 32)

    r_blank = df[colR].isna() | (df[colR].astype(str).str.strip() == "")

    p_true = (df[colP] == True) | (df[colP].astype(str).str.strip().str.lower() == "true")
    q_true = (df[colQ] == True) | (df[colQ].astype(str).str.strip().str.lower() == "true")

    l_num = pd.to_numeric(df[colL], errors="coerce")
    m_num = pd.to_numeric(df[colM], errors="coerce")

    exclude_m = (m_num > 0)
    exclude_l = (l_num > 2)

    mask = d_active & k_ok & r_blank & ~p_true & ~q_true & ~exclude_m & ~exclude_l

    # доп. фильтр Paid/Capacity >= 50%
    def _norm(s: str) -> str:
        return str(s).strip().lower().replace("_", " ").replace("-", " ")
    paid_aliases = {"paid students", "paid student", "paid"}
    cap_aliases  = {"capacity", "cap"}
    colPaid = colCap = None
    for c in df.columns:
        n = _norm(c)
        if (colPaid is None) and (n in paid_aliases): colPaid = c
        if (colCap  is None) and (n in cap_aliases):  colCap  = c
        if colPaid is not None and colCap is not None: break
    if colPaid is not None and colCap is not None:
        paid_num = pd.to_numeric(df[colPaid], errors="coerce")
        cap_num  = pd.to_numeric(df[colCap],  errors="coerce")
        ratio_ge_50 = (cap_num > 0) & ((paid_num / cap_num) >= 0.5)
        mask = mask & ~ratio_ge_50

    out = df.loc[mask].copy()
    out[colK] = k_num.loc[out.index]
    return out


def to_excel_bytes(data: pd.DataFrame) -> io.BytesIO | None:
    try:
        import xlsxwriter  # noqa: F401
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            data.to_excel(writer, index=False)
        buf.seek(0)
        return buf
    except Exception:
        return None


def _time_to_minutes(v: str) -> float:
    if v is None or str(v).strip() == "" or pd.isna(v):
        return np.nan
    s = str(v).strip()
    if pd.Series([s]).str.match(r"^\d{1,2}:\d{2}(:\d{2})?$", na=False).iloc[0]:
        parts = s.split(":")
        h = int(parts[0]); m = int(parts[1])
        return h * 60 + m
    dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return np.nan
    return dt.hour * 60 + dt.minute


def _find_rating_col(df: pd.DataFrame) -> str | None:
    if "Rating_BP" in df.columns:
        return "Rating_BP"
    for c in df.columns:
        n = str(c).strip().lower()
        if "rating" in n:
            return c
    return None


def add_matches_column(df: pd.DataFrame,
                       good_set=("Good","Amazing","New tutor (Good)"),
                       bad_set=("Bad","New tutor (Bad)"),
                       new_col_name="Matches") -> pd.DataFrame:
    if df.empty:
        return df

    colB, colE, colF, colG, colI, colJ, colK = df.columns[1], df.columns[4], df.columns[5], df.columns[6], df.columns[8], df.columns[9], df.columns[10]
    rating_col = _find_rating_col(df)

    f_vals = df[colF].astype(str).str.strip()
    g_vals = df[colG].astype(str).str.strip()
    i_vals = df[colI].astype(str).str.strip()
    i_mins = i_vals.apply(_time_to_minutes)
    k_num  = pd.to_numeric(df[colK], errors="coerce")
    r_vals = df[rating_col].astype(str).str.strip() if rating_col else pd.Series("", index=df.index)

    b_vals   = df[colB].astype(str).str.upper().fillna("")
    b_is_prm = b_vals.str.contains("PRM", na=False)

    good_l = {x.lower() for x in good_set}
    bad_l  = {x.lower() for x in bad_set}
    r_low  = r_vals.str.lower()

    lines, counts = [], []
    n = len(df)
    for i in range(n):
        mF = (f_vals == f_vals.iloc[i])
        mG = (g_vals == g_vals.iloc[i])

        base_t = i_mins.iloc[i]
        mI = pd.Series(False, index=df.index)
        if not pd.isna(base_t):
            mI = (i_mins.sub(base_t).abs() <= 120)

        base_k = k_num.iloc[i]
        mK = pd.Series(False, index=df.index)
        if not pd.isna(base_k):
            mK = (k_num.sub(base_k).abs() <= 1)

        ri = r_low.iloc[i] if rating_col else ""
        mR = (~r_low.isin(bad_l)) & ((r_low == ri) | (r_low.isin(good_l)))

        mPRM = (b_is_prm == b_is_prm.iloc[i])

        mask = mF & mG & mI & mK & mR & mPRM
        mask.iloc[i] = False

        if mask.any():
            sub = df.loc[mask, [colB, colI, colJ, colK, colE]]
            if rating_col and rating_col in df.columns:
                sub = sub.assign(_rating=df.loc[mask, rating_col].values)
            else:
                sub = sub.assign(_rating="")
            lst = [f"{row[colB]}, {row[colI]}, {row[colJ]}, K: {row[colK]}, E: {row[colE]}, Rating: {row['_rating']}" for _, row in sub.iterrows()]
            lines.append("\n".join(lst)); counts.append(len(lst))
        else:
            lines.append(""); counts.append(0)

    out = df.copy()
    name = new_col_name
    while name in out.columns:
        name += "_x"
    count_name = f"{new_col_name}_count"
    while count_name in out.columns:
        count_name += "_x"
    out[name] = lines
    out[count_name] = counts
    return out


def _b_suffix3(s: str) -> str:
    """
    Возвращает 3 буквы:
      - если в B есть >= 2 подчёркиваний — берём 3 буквы после второго '_'
      - если ровно 1 подчёркивание — берём 3 буквы после первого '_'
      - иначе ""
    """
    if s is None or pd.isna(s):
        return ""
    s = str(s).upper()
    parts = s.split("_")
    if len(parts) >= 3:
        tail = parts[2]
    elif len(parts) == 2:
        tail = parts[1]
    else:
        return ""
    letters = "".join(ch for ch in tail if ch.isalpha())
    return letters[:3] if len(letters) >= 3 else ""


def add_alt_matches_column(df: pd.DataFrame,
                           good_set=("Good","Amazing","New tutor (Good)"),
                           bad_set=("Bad","New tutor (Bad)"),
                           new_col_name="AltMatches") -> pd.DataFrame:
    if df.empty:
        return df

    colB, colE, colF, colG, colI, colJ, colK = df.columns[1], df.columns[4], df.columns[5], df.columns[6], df.columns[8], df.columns[9], df.columns[10]
    rating_col = _find_rating_col(df)

    f_vals = df[colF].astype(str).str.strip()
    g_vals = df[colG].astype(str).str.strip()
    k_num  = pd.to_numeric(df[colK], errors="coerce")
    r_vals = df[rating_col].astype(str).str.strip() if rating_col else pd.Series("", index=df.index)

    b_vals    = df[colB].astype(str).fillna("").str.upper()
    b_is_prm  = b_vals.str.contains("PRM", na=False)
    b_suffix3 = b_vals.apply(_b_suffix3)

    good_l = {x.lower() for x in good_set}
    bad_l  = {x.lower() for x in bad_set}
    r_low  = r_vals.str.lower()

    i_vals = df[colI].astype(str).str.strip()
    i_mins = i_vals.apply(_time_to_minutes)

    lines_alt, counts_alt = [], []
    n = len(df)

    for i in range(n):
        mF = (f_vals == f_vals.iloc[i])
        mG = (g_vals == g_vals.iloc[i])

        base_k = k_num.iloc[i]
        mK = pd.Series(False, index=df.index)
        if not pd.isna(base_k):
            mK = (k_num.sub(base_k).abs() <= 1)

        ri = r_low.iloc[i] if rating_col else ""
        mR = (~r_low.isin(bad_l)) & ((r_low == ri) | (r_low.isin(good_l)))

        mPRM = (b_is_prm == b_is_prm.iloc[i])
        mSUF = (b_suffix3 == b_suffix3.iloc[i])

        base_t = i_mins.iloc[i]
        mI = pd.Series(False, index=df.index)
        if not pd.isna(base_t):
            mI = (i_mins.sub(base_t).abs() <= 120)
        mask_regular = mF & mG & mI & mK & mR & mPRM

        mask_alt = mF & mG & mK & mR & mPRM & mSUF

        mask_regular.iloc[i] = False
        mask_alt.iloc[i]     = False

        mask_alt = mask_alt & ~mask_regular

        if mask_alt.any():
            sub = df.loc[mask_alt, [colB, colI, colJ, colK, colE]]
            if rating_col and rating_col in df.columns:
                sub = sub.assign(_rating=df.loc[mask_alt, rating_col].values)
            else:
                sub = sub.assign(_rating="")
            lst = [f"{row[colB]}, {row[colI]}, {row[colJ]}, K: {row[colK]}, E: {row[colE]}, Rating: {row['_rating']}" for _, row in sub.iterrows()]
            lines_alt.append("\n".join(lst)); counts_alt.append(len(lst))
        else:
            lines_alt.append(""); counts_alt.append(0)

    out = df.copy()
    name = new_col_name
    while name in out.columns:
        name += "_x"
    count_name = f"{new_col_name}_count"
    while count_name in out.columns:
        count_name += "_x"
    out[name] = lines_alt
    out[count_name] = counts_alt
    return out


def add_wide_matches_column(df: pd.DataFrame,
                            good_set=("Good","Amazing","New tutor (Good)"),
                            bad_set=("Bad","New tutor (Bad)"),
                            new_col_name="WideMatches") -> pd.DataFrame:
    if df.empty:
        return df

    colB, colE, colF, colG, colI, colJ, colK = df.columns[1], df.columns[4], df.columns[5], df.columns[6], df.columns[8], df.columns[9], df.columns[10]
    rating_col = _find_rating_col(df)

    f_vals = df[colF].astype(str).str.strip()
    g_vals = df[colG].astype(str).str.strip()
    k_num  = pd.to_numeric(df[colK], errors="coerce")
    r_vals = df[rating_col].astype(str).str.strip() if rating_col else pd.Series("", index=df.index)

    b_vals   = df[colB].astype(str).fillna("").str.upper()
    b_is_prm = b_vals.str.contains("PRM", na=False)

    i_vals = df[colI].astype(str).str.strip()
    i_mins = i_vals.apply(_time_to_minutes)
    b_suf3 = b_vals.apply(_b_suffix3)

    good_l = {x.lower() for x in good_set}
    bad_l  = {x.lower() for x in bad_set}
    r_low  = r_vals.str.lower()

    pos = pd.Series(range(len(df)), index=df.index)

    lines, counts = [], []
    n = len(df)
    for i in range(n):
        mF = (f_vals == f_vals.iloc[i])
        mG = (g_vals == g_vals.iloc[i])

        base_k = k_num.iloc[i]
        mK = pd.Series(False, index=df.index)
        if not pd.isna(base_k):
            mK = (k_num.sub(base_k).abs() <= 1)

        ri = r_low.iloc[i] if rating_col else ""
        mR = (~r_low.isin(bad_l)) & ((r_low == ri) | (r_low.isin(good_l)))

        mPRM = (b_is_prm == b_is_prm.iloc[i])

        base_t = i_mins.iloc[i]
        mI = pd.Series(False, index=df.index)
        if not pd.isna(base_t):
            mI = (i_mins.sub(base_t).abs() <= 120)
        mask_regular = mF & mG & mI & mK & mR & mPRM

        mask_alt = mF & mG & mK & mR & mPRM & (b_suf3 == b_suf3.iloc[i])

        mask_wide = mF & mG & mK & mR & mPRM

        mask_regular.iloc[i] = False
        mask_alt.iloc[i]     = False
        mask_wide.iloc[i]    = False
        mask_wide = mask_wide & (pos > pos.iloc[i])

        mask_final = mask_wide & ~mask_regular & ~mask_alt

        if mask_final.any():
            sub = df.loc[mask_final, [colB, colI, colJ, colK, colE]]
            if rating_col and rating_col in df.columns:
                sub = sub.assign(_rating=df.loc[mask_final, rating_col].values)
            else:
                sub = sub.assign(_rating="")
            lst = [f"{row[colB]}, {row[colI]}, {row[colJ]}, K: {row[colK]}, E: {row[colE]}, Rating: {row['_rating']}" for _, row in sub.iterrows()]
            lines.append("\n".join(lst)); counts.append(len(lst))
        else:
            lines.append(""); counts.append(0)

    out = df.copy()
    name = new_col_name
    while name in out.columns:
        name += "_x"
    cnt = f"{new_col_name}_count"
    while cnt in out.columns:
        cnt += "_x"
    out[name] = lines
    out[cnt]  = counts
    return out

import re

def _norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def _pick_col(df: pd.DataFrame, candidates: set[str], fallback_idx: int | None = None) -> str | None:
    """Находит колонку по набору синонимов (без регистра/лишних пробелов). Иначе — fallback по индексу."""
    norm = {_norm_name(c): c for c in df.columns}
    for key in candidates:
        if key in norm:
            return norm[key]
    if fallback_idx is not None and fallback_idx < len(df.columns):
        return df.columns[fallback_idx]
    return None


def main():
    st.title("Initial export (A:R, D='active', K < 32, R empty, P/Q != TRUE)")

    with st.sidebar:
        st.header("Source")
        sheet_id = st.text_input("Google Sheet ID", value=SHEET_ID)
        ws_name  = st.text_input("Worksheet", value=WS_NAME)
        try:
            client = _authorize_client()
            sh = client.open_by_key(sheet_id)
            ws_names = [ws.title for ws in sh.worksheets()]
            if ws_name in ws_names:
                ws_name = st.selectbox("Select worksheet", ws_names, index=ws_names.index(ws_name))
            else:
                ws_name = st.selectbox("Select worksheet", ws_names, index=0)
            selected_ws = sh.worksheet(ws_name)
            gid = getattr(selected_ws, "id", None) or selected_ws._properties.get("sheetId")
            sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"
            tab_url   = f"{sheet_url}#gid={gid}"
            st.markdown(f"[Open Google Sheet]({sheet_url})")
            st.markdown(f"[Open tab **{ws_name}**]({tab_url})")
            st.caption(f"Worksheets found: {len(ws_names)}")
        except Exception as e:
            st.warning(f"Не удалось получить список вкладок/ссылки: {e}")

    with st.spinner("Loading data from Google Sheets…"):
        df = load_sheet_df(sheet_id, ws_name)

    if df.empty:
        st.warning(f"Пусто: проверь вкладку '{ws_name}' и доступ сервисного аккаунта (Viewer/Editor).")
        st.stop()

    df = adjust_local_time_minus_3(df)

    mapping = load_group_age_map()
    df = replace_group_age_from_map(df, mapping)

    rating_map = load_rating_bp_map()
    df = add_rating_bp_by_O(df, rating_map, new_col_name="Rating_BP")

    filtered = filter_df(df)

    # матчи
    filtered = add_matches_column(filtered, new_col_name="Matches")
    filtered = add_alt_matches_column(filtered, new_col_name="AltMatches")
    filtered = add_wide_matches_column(filtered, new_col_name="WideMatches")

    c1, c2 = st.columns(2)
    c1.caption(f"Rows total: {len(df)}")
    c2.success(f"Filtered rows: {len(filtered)}")

        # --- Sidebar Filters ---
    dff = filtered.copy()

    # маппим нужные поля по именам, с fallback на позиции A:R (0-based)
    col_group      = _pick_col(dff, {"group", "group title", "group name", "group id"}, fallback_idx=1)   # B
    col_tutor      = _pick_col(dff, {"tutor", "teacher", "tutor name", "teacher name"}, fallback_idx=4)   # E
    col_tutor_id   = _pick_col(dff, {"tutor id", "teacher id", "id"}, fallback_idx=14)                    # O
    col_course     = _pick_col(dff, {"course"}, fallback_idx=5)                                           # F
    col_group_age  = _pick_col(dff, {"group age", "age"}, fallback_idx=6)                                 # G
    col_local_time = _pick_col(dff, {"local time", "localtime", "time (local)"}, fallback_idx=8)          # I
    col_module     = _pick_col(dff, {"module"}, fallback_idx=2)                                           # C
    col_lesson_num = _pick_col(dff, {"lesson number", "lesson", "lesson_num"}, fallback_idx=13)           # N
    col_capacity   = _pick_col(dff, {"capacity", "cap"})                                                  # по имени
    col_paid       = _pick_col(dff, {"paid students", "paid student", "paid"})                            # по имени
    col_transfer1  = _pick_col(dff, {"students transferred 1 time", "transferred 1 time"}, fallback_idx=11) # L

    # колонки matches
    col_m_text  = "Matches"           if "Matches"           in dff.columns else None
    col_m_cnt   = "Matches_count"     if "Matches_count"     in dff.columns else None
    col_a_text  = "AltMatches"        if "AltMatches"        in dff.columns else None
    col_a_cnt   = "AltMatches_count"  if "AltMatches_count"  in dff.columns else None
    col_w_text  = "WideMatches"       if "WideMatches"       in dff.columns else None
    col_w_cnt   = "WideMatches_count" if "WideMatches_count" in dff.columns else None

    with st.sidebar.expander("Filters", expanded=True):
        # — текстовые мультиселекты —
        def _multi(df, label, col):
            if not col: return df
            vals = sorted(df[col].dropna().astype(str).unique())
            sel = st.multiselect(label, vals)
            if sel:
                df = df[df[col].astype(str).isin(sel)]
            return df

        dff = _multi(dff, "Group",      col_group)
        dff = _multi(dff, "Tutor",      col_tutor)
        dff = _multi(dff, "Tutor ID",   col_tutor_id)
        dff = _multi(dff, "Course",     col_course)
        dff = _multi(dff, "Group age",  col_group_age)
        dff = _multi(dff, "Local time", col_local_time)
        dff = _multi(dff, "Module",     col_module)

        # — числовые диапазоны —
        def _range(df, label, col):
            if not col: return df
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                vmin = int(s.min())
                vmax = int(s.max())
                lo, hi = st.slider(label, vmin, vmax, (vmin, vmax))
                df = df[(pd.to_numeric(df[col], errors="coerce") >= lo) &
                        (pd.to_numeric(df[col], errors="coerce") <= hi)]
            return df

        dff = _range(dff, "Lesson number",          col_lesson_num)
        dff = _range(dff, "Capacity",               col_capacity)
        dff = _range(dff, "Paid students",          col_paid)
        dff = _range(dff, "Students transferred 1 time", col_transfer1)

        # — фильтры по matches —
        def _count_slider(df, label, col_cnt):
            if not col_cnt: return df
            s = pd.to_numeric(df[col_cnt], errors="coerce").fillna(0).astype(int)
            mx = int(s.max()) if len(s) else 0
            v  = st.slider(label, 0, max(1, mx), 0)
            return df[s >= v]

        dff = _count_slider(dff, "Min Matches",      col_m_cnt)
        dff = _count_slider(dff, "Min Alt matches",  col_a_cnt)
        dff = _count_slider(dff, "Min Wide matches", col_w_cnt)

        q = st.text_input("Search in matches text")
        if q:
            qrx = re.escape(q)
            mask = pd.Series(False, index=dff.index)
            for col in [col_m_text, col_a_text, col_w_text]:
                if col:
                    mask |= dff[col].astype(str).str.contains(qrx, case=False, na=False)
            dff = dff[mask]

        only_any = st.checkbox("Only rows with any matches", value=False)
        if only_any:
            cnt = pd.Series(0, index=dff.index)
            for col in [col_m_cnt, col_a_cnt, col_w_cnt]:
                if col:
                    cnt = cnt.add(pd.to_numeric(dff[col], errors="coerce").fillna(0), fill_value=0)
            dff = dff[cnt > 0]

    # --- Причесанный вывод в заданном порядке ---
    cols_all = list(dff.columns)
    def col(idx): 
        return cols_all[idx] if idx < len(cols_all) else None
    colA, colB, colC = col(0), col(1), col(2)
    colE, colF, colG = col(4), col(5), col(6)
    colI, colJ, colK = col(8), col(9), col(10)
    colL, colN, colO = col(11), col(13), col(14)

    desired = [
        colA, colB, colE, colO, colF, colG, colI, colJ, colK, colC, colN, colL,
        "Matches_count", "Matches",
        "AltMatches_count", "AltMatches",
        "WideMatches_count", "WideMatches",
    ]
    display_cols = [c for c in desired if (c is not None and c in dff.columns)]
    curated = dff.loc[:, display_cols].copy()

    st.dataframe(curated, use_container_width=True, height=700)
    st.download_button(
        "⬇️ Download CSV (curated)",
        curated.to_csv(index=False).encode("utf-8"),
        file_name="curated_view.csv",
        mime="text/csv",
    )


    # --- Причесанный вывод в заданном порядке ---
    # Берём имена колонок по позициям A..R (как в исходном листе)
    cols = list(filtered.columns)
    # защитимся, если колонок меньше ожидаемого
    def col(idx): 
        return cols[idx] if idx < len(cols) else None

    colA = col(0)   # A
    colB = col(1)   # B
    colC = col(2)   # C
    # D = col(3)
    colE = col(4)   # E
    colF = col(5)   # F
    colG = col(6)   # G
    # H = col(7)
    colI = col(8)   # I
    colJ = col(9)   # J
    colK = col(10)  # K
    colL = col(11)  # L
    # M = col(12)
    colN = col(13)  # N
    colO = col(14)  # O

    desired = [
        colA, colB, colE, colO, colF, colG, colI, colJ, colK, colC, colN, colL,
        "Matches_count", "Matches",
        "AltMatches_count", "AltMatches",
        "WideMatches_count", "WideMatches",
    ]

    # убираем None и оставляем только реально существующие колонki
    display_cols = [c for c in desired if (c is not None and c in filtered.columns)]

    curated = filtered.loc[:, display_cols].copy()

    st.dataframe(curated, use_container_width=True, height=700)

    st.download_button(
        "⬇️ Download CSV (curated)",
        curated.to_csv(index=False).encode("utf-8"),
        file_name="curated_view.csv",
        mime="text/csv",
    )

    if st.button("Refresh"):
        load_sheet_df.clear()
        load_group_age_map.clear()
        load_rating_bp_map.clear()
        st.rerun()


if __name__ == "__main__":
    main()
