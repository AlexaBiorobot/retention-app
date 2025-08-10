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

    # --- рейтинг ---
    if rating_col:
        r_vals = df[rating_col].astype(str).str.strip()
        r_low  = r_vals.str.lower()
        good_l = {x.lower() for x in good_set}
        bad_l  = {x.lower() for x in bad_set}
        cand_is_bad = r_low.isin(bad_l)
        cand_is_good = r_low.isin(good_l)
        cand_is_new  = r_low.str.contains("new tutor", na=False)
    else:
        r_vals = pd.Series("", index=df.index)
        r_low  = r_vals
        cand_is_bad  = pd.Series(False, index=df.index)
        cand_is_good = pd.Series(False, index=df.index)
        cand_is_new  = pd.Series(False, index=df.index)

    # PRM по B
    b_vals   = df[colB].astype(str).str.upper().fillna("")
    b_is_prm = b_vals.str.contains("PRM", na=False)

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

        # рейтинг: не bad И (равен моему ИЛИ кандидат из good ИЛИ
        # если у меня "new tutor", то и у кандидата должен быть "new tutor")
        if rating_col:
            my_r  = r_low.iloc[i]
            my_is_new = "new tutor" in my_r
            mR = (~cand_is_bad) & ((r_low == my_r) | cand_is_good | (my_is_new & cand_is_new))
        else:
            mR = pd.Series(True, index=df.index)

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

    # рейтинг
    if rating_col:
        r_vals = df[rating_col].astype(str).str.strip()
        r_low  = r_vals.str.lower()
        good_l = {x.lower() for x in good_set}
        bad_l  = {x.lower() for x in bad_set}
        cand_is_bad = r_low.isin(bad_l)
        cand_is_good = r_low.isin(good_l)
        cand_is_new  = r_low.str.contains("new tutor", na=False)
    else:
        r_low = pd.Series("", index=df.index)
        cand_is_bad = pd.Series(False, index=df.index)
        cand_is_good = pd.Series(False, index=df.index)
        cand_is_new  = pd.Series(False, index=df.index)

    # PRM и суффикс
    b_vals    = df[colB].astype(str).fillna("").str.upper()
    b_is_prm  = b_vals.str.contains("PRM", na=False)
    b_suffix3 = b_vals.apply(_b_suffix3)

    # для исключения обычных матчей
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

        if rating_col:
            my_r  = r_low.iloc[i]
            my_is_new = "new tutor" in my_r
            mR = (~cand_is_bad) & ((r_low == my_r) | cand_is_good | (my_is_new & cand_is_new))
        else:
            mR = pd.Series(True, index=df.index)

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

    # рейтинг
    if rating_col:
        r_vals = df[rating_col].astype(str).str.strip()
        r_low  = r_vals.str.lower()
        good_l = {x.lower() for x in good_set}
        bad_l  = {x.lower() for x in bad_set}
        cand_is_bad = r_low.isin(bad_l)
        cand_is_good = r_low.isin(good_l)
        cand_is_new  = r_low.str.contains("new tutor", na=False)
    else:
        r_low = pd.Series("", index=df.index)
        cand_is_bad = pd.Series(False, index=df.index)
        cand_is_good = pd.Series(False, index=df.index)
        cand_is_new  = pd.Series(False, index=df.index)

    b_vals   = df[colB].astype(str).fillna("").str.upper()
    b_is_prm = b_vals.str.contains("PRM", na=False)

    i_vals = df[colI].astype(str).str.strip()
    i_mins = i_vals.apply(_time_to_minutes)
    b_suf3 = b_vals.apply(_b_suffix3)

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

        if rating_col:
            my_r  = r_low.iloc[i]
            my_is_new = "new tutor" in my_r
            mR = (~cand_is_bad) & ((r_low == my_r) | cand_is_good | (my_is_new & cand_is_new))
        else:
            mR = pd.Series(True, index=df.index)

        mPRM = (b_is_prm == b_is_prm.iloc[i])

        # исключаем «обычные» и «альтернативные»
        base_t = i_mins.iloc[i]
        mI = pd.Series(False, index=df.index)
        if not pd.isna(base_t):
            mI = (i_mins.sub(base_t).abs() <= 120)
        mask_regular = mF & mG & mI & mK & mR & mPRM
        mask_alt     = mF & mG & mK & mR & mPRM & (b_suf3 == b_suf3.iloc[i])

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

# ---- Нормализатор имён и выбор колонок по синонимам ----
def _norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def _pick_col(df: pd.DataFrame, candidates: set[str], fallback_idx: int | None = None) -> str | None:
    norm = {_norm_name(c): c for c in df.columns}
    for key in candidates:
        if key in norm:
            return norm[key]
    if fallback_idx is not None and fallback_idx < len(df.columns):
        return df.columns[fallback_idx]
    return None


def main():
    st.title("Initial export (A:R, D='active', K < 32, R empty, P/Q != TRUE)")

    # --- Source (hidden, no UI) ---
    sheet_id = SHEET_ID
    ws_name  = WS_NAME
    
    # Подстрахуемся: если указанной вкладки нет — возьмём первую, но ничего не показываем в UI
    try:
        client = _authorize_client()
        sh = client.open_by_key(sheet_id)
        ws_names = [ws.title for ws in sh.worksheets()]
        if ws_name not in ws_names and ws_names:
            ws_name = ws_names[0]
    except Exception:
        # молча продолжаем — ниже всё равно отработает обработка ошибок
        pass


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

    # --- Sidebar Filters (multiselect-only, без слайдеров по тексту) ---
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

    def _ms_options(df_, col):
        if not col or col not in df_.columns:
            return []
        return sorted(df_[col].dropna().astype(str).unique())

    def _apply_ms(df_, col, sel):
        if not col or not sel:
            return df_
        return df_[df_[col].astype(str).isin(sel)]

    with st.sidebar.expander("Filters", expanded=True):
        # текстовые мультиселекты
        dff = _apply_ms(dff, col_group,      st.multiselect("Group",      _ms_options(dff, col_group)))
        dff = _apply_ms(dff, col_tutor,      st.multiselect("Tutor",      _ms_options(dff, col_tutor)))
        dff = _apply_ms(dff, col_tutor_id,   st.multiselect("Tutor ID",   _ms_options(dff, col_tutor_id)))
        dff = _apply_ms(dff, col_course,     st.multiselect("Course",     _ms_options(dff, col_course)))
        dff = _apply_ms(dff, col_group_age,  st.multiselect("Group age",  _ms_options(dff, col_group_age)))
        dff = _apply_ms(dff, col_local_time, st.multiselect("Local time", _ms_options(dff, col_local_time)))
        dff = _apply_ms(dff, col_module,     st.multiselect("Module",     _ms_options(dff, col_module)))

        # тоже как текст — никаких слайдеров
        dff = _apply_ms(dff, col_lesson_num, st.multiselect("Lesson number",             _ms_options(dff, col_lesson_num)))
        dff = _apply_ms(dff, col_capacity,   st.multiselect("Capacity",                  _ms_options(dff, col_capacity)))
        dff = _apply_ms(dff, col_paid,       st.multiselect("Paid students",             _ms_options(dff, col_paid)))
        dff = _apply_ms(dff, col_transfer1,  st.multiselect("Students transferred 1 time", _ms_options(dff, col_transfer1)))

        # пороги по количеству матчей
        if col_m_cnt: 
            min_m = st.number_input("Min Matches",      min_value=0, value=0, step=1)
            dff = dff[pd.to_numeric(dff[col_m_cnt], errors="coerce").fillna(0).astype(int) >= min_m]
        if col_a_cnt:
            min_a = st.number_input("Min Alt matches",  min_value=0, value=0, step=1)
            dff = dff[pd.to_numeric(dff[col_a_cnt], errors="coerce").fillna(0).astype(int) >= min_a]
        if col_w_cnt:
            min_w = st.number_input("Min Wide matches", min_value=0, value=0, step=1)
            dff = dff[pd.to_numeric(dff[col_w_cnt], errors="coerce").fillna(0).astype(int) >= min_w]

        # поиск по тексту матчей
        q = st.text_input("Search in matches text")
        if q:
            qrx = re.escape(q)
            mask = pd.Series(False, index=dff.index)
            for col in [col_m_text, col_a_text, col_w_text]:
                if col:
                    mask |= dff[col].astype(str).str.contains(qrx, case=False, na=False)
            dff = dff[mask]

        # оставить только строки, где есть любые матчи
        only_any = st.checkbox("Only rows with any matches", value=False)
        if only_any:
            cnt = pd.Series(0, index=dff.index)
            for col in [col_m_cnt, col_a_cnt, col_w_cnt]:
                if col:
                    cnt = cnt.add(pd.to_numeric(dff[col], errors="coerce").fillna(0).astype(int), fill_value=0)
            dff = dff[cnt > 0]

    # --- Причесанный вывод в заданном порядке ---
    cols_all = list(dff.columns)
    def col(idx): 
        return cols_all[idx] if idx < len(cols_all) else None
    
    colA, colB, colC = col(0), col(1), col(2)
    colE, colF, colG = col(4), col(5), col(6)
    colI, colJ, colK = col(8), col(9), col(10)
    colL, colN, colO = col(11), col(13), col(14)
    
    # найдём колонку рейтинга и поставим её сразу после Tutor ID
    rating_colname = _find_rating_col(dff)  # например "Rating_BP" или "Rating"
    desired = [
        colA, colB, colE, colO, rating_colname,  # <- Rating сразу после Tutor ID
        colF, colG, colI, colJ, colK, colC, colN, colL,
        "Matches_count", "Matches",
        "AltMatches_count", "AltMatches",
        "WideMatches_count", "WideMatches",
    ]
    display_cols = [c for c in desired if (c is not None and c in dff.columns)]
    curated = dff.loc[:, display_cols].copy()
    
    # переименуем заголовок рейтинга для красоты
    if rating_colname and rating_colname in curated.columns and "Rating" not in curated.columns:
        curated = curated.rename(columns={rating_colname: "Rating"})
    
    # убрать полностью пустые строки ('' тоже считаем пустым)
    curated = curated.replace(r"^\s*$", pd.NA, regex=True).dropna(how="all")


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
