import os
import json
import io
import re
import numpy as np
import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread.exceptions import SpreadsheetNotFound, WorksheetNotFound

# ==== Page / UX ====  (ДОЛЖНО быть самым первым вызовом Streamlit)
st.set_page_config(
    page_title="Disbanding Brazil",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==== Constants ====
DEFAULT_SHEET_ID = "1Jbb4p1cZCo67ZRiW5cmFUq-c9ijo5VMH_hFFMVYeJk4"
DEFAULT_WS_NAME  = "data"

# правильный ID без replace
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
    try:
        client = _authorize_client()
        sh = client.open_by_key(sheet_id)
        ws = sh.worksheet(worksheet_name)
        vals = ws.get("A:E")
    except SpreadsheetNotFound:
        st.warning("Не могу открыть таблицу EXT_GROUPS_SS_ID. Проверь ID и доступ сервисного аккаунта.")
        return {}
    except WorksheetNotFound:
        st.warning(f"Не найден лист '{worksheet_name}' в EXT_GROUPS_SS_ID.")
        return {}
    except Exception as e:
        st.warning(f"Ошибка при чтении EXT_GROUPS_SS_ID: {e}")
        return {}
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


# >>> ВОТ ЭТА ФУНКЦИЯ БЫЛА ОТСУТСТВУЮЩЕЙ <<<
def replace_group_age_from_map(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Подставляем Group age из внешней карты по колонке B (Group/ID/Title).
       Если в mapping нет значения — оставляем исходное.
    """
    if df.empty or not mapping:
        return df.copy()

    dff = df.copy()

    # ищем колонку B (group id/title) по синонимам
    colB = None
    for c in dff.columns:
        if str(c).strip().lower().replace("_", " ") in (
            "b", "group id", "group", "group title", "group_name", "group name"
        ):
            colB = c
            break
    if colB is None:
        colB = dff.columns[1] if len(dff.columns) >= 2 else None

    # ищем колонку Group age
    colG = None
    for c in dff.columns:
        if str(c).strip().lower().replace("_", " ") == "group age":
            colG = c
            break
    if colG is None:
        colG = dff.columns[6] if len(dff.columns) >= 7 else None

    if colB is None or colG is None:
        return dff

    keys = dff[colB].astype(str).str.strip()
    new_vals = keys.map(lambda k: mapping.get(k, pd.NA))

    # подставляем только непустые значения из mapping
    dff[colG] = new_vals.where(new_vals.notna() & (new_vals.astype(str).str.strip() != ""), dff[colG])
    return dff
# <<< КОНЕЦ ДОБАВЛЕННОГО БЛОКА >>>


@st.cache_data(show_spinner=False, ttl=300)
def load_rating_bp_map(sheet_id: str = RATING_SS_ID, worksheet_name: str = RATING_WS) -> dict:
    try:
        client = _authorize_client()
        ws = client.open_by_key(sheet_id).worksheet(worksheet_name)
        vals = ws.get(
            "A:BP",
            value_render_option="UNFORMATTED_VALUE",
            date_time_render_option="FORMATTED_STRING",
        )
    except SpreadsheetNotFound:
        st.warning("Не могу открыть таблицу RATING_SS_ID. Проверь ID и доступ сервисного аккаунта.")
        return {}
    except WorksheetNotFound:
        st.warning(f"Не найден лист '{worksheet_name}' в RATING_SS_ID.")
        return {}
    except Exception as e:
        st.warning(f"Ошибка при чтении RATING_SS_ID: {e}")
        return {}
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
    """Базовые фильтры + дополнительный: оставляем только строки, где (Capacity - Paid) >= 1.
       По R берём, если пусто/NaN/0/"0".
    """
    if df.empty:
        return df
    if len(df.columns) < 18:
        st.error("Ожидалось минимум 18 колонок (до R). Проверь диапазон A:R и заголовки.")
        st.stop()

    colD, colK = df.columns[3], df.columns[10]
    colL, colM = df.columns[11], df.columns[12]
    colP, colQ, colR = df.columns[15], df.columns[16], df.columns[17]

    # Базовые поля
    d_active = df[colD].astype(str).str.strip().str.lower() == "active"

    k_num = pd.to_numeric(df[colK], errors="coerce")
    k_ok  = k_num.notna() & (k_num > 3) & (k_num < 32)

    # R: ок, если пусто/NaN/0/"0"
    r_str = df[colR].astype(str).str.strip().str.lower()
    r_num = pd.to_numeric(df[colR], errors="coerce")
    r_ok  = df[colR].isna() | (r_str == "") | (r_str == "0") | (r_num == 0)

    # P/Q (флаги), L/M (исключения)
    p_true = (df[colP] == True) | (df[colP].astype(str).str.strip().str.lower() == "true")
    q_true = (df[colQ] == True) | (df[colQ].astype(str).str.strip().str.lower() == "true")

    l_num = pd.to_numeric(df[colL], errors="coerce")
    m_num = pd.to_numeric(df[colM], errors="coerce")

    exclude_m = (m_num > 0)
    exclude_l = (l_num > 2)

    # Итоговая маска
    mask = d_active & k_ok & r_ok & ~p_true & ~q_true & ~exclude_m & ~exclude_l

    # --- ДОП. ФИЛЬТР: есть свободные места (Capacity - Paid >= 1) ---
    def _norm(s: str) -> str:
        return str(s).strip().lower().replace("_", " ").replace("-", " ")

    paid_aliases = {"paid students", "paid student", "paid"}
    cap_aliases  = {"capacity", "cap"}
    colPaid = colCap = None
    for c in df.columns:
        n = _norm(c)
        if colPaid is None and n in paid_aliases:
            colPaid = c
        if colCap  is None and n in cap_aliases:
            colCap  = c
        if colPaid is not None and colCap is not None:
            break

    free_slots = None
    paid_pct_series = None
    if colPaid is not None and colCap is not None:
        paid_num = pd.to_numeric(df[colPaid], errors="coerce")
        cap_num  = pd.to_numeric(df[colCap],  errors="coerce")
        free_slots = (cap_num - paid_num)
        have_free  = (cap_num.notna() & paid_num.notna()) & (free_slots >= 1)
        mask = mask & have_free

        # Paid % = round(Paid/Capacity*100), формат строкой "NN%"
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(cap_num > 0, (paid_num / cap_num) * 100.0, np.nan)
        paid_pct_series = pd.Series(pct, index=df.index)
        paid_pct_series = paid_pct_series.apply(lambda x: f"{int(round(x))}%"
                                                if pd.notna(x) else pd.NA)

    out = df.loc[mask].copy()
    out[colK] = k_num.loc[out.index]

    # Добавим колонки Free slots и Paid % (если нашли Paid/Capacity)
    if (colPaid is not None) and (colCap is not None) and (free_slots is not None):
        out["Free slots"] = free_slots.loc[out.index]
        if paid_pct_series is not None:
            out["Paid %"] = paid_pct_series.loc[out.index]

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


# --- утилиты матчинга ---

def _time_to_minutes(v: str) -> float:
    """I всегда HH:MM."""
    if v is None or pd.isna(v): return np.nan
    s = str(v).strip()
    if not re.fullmatch(r"\d{1,2}:\d{2}", s):
        return np.nan
    h, m = s.split(":")
    return int(h) * 60 + int(m)

def _find_rating_col(df: pd.DataFrame) -> str | None:
    if "Rating_BP" in df.columns:
        return "Rating_BP"
    for c in df.columns:
        n = str(c).strip().lower()
        if "rating" in n:
            return c
    return None

def _find_free_slots_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c).strip().lower() == "free slots":
            return c
    return None

def _norm_rating(x: str) -> str:
    if x is None or pd.isna(x): return ""
    s = str(x).strip().lower()
    repl = {
        "amazing": "amazing",
        "good": "good",
        "ok": "ok",
        "new tutor (good)": "new_tutor_good",
        "new tutor (ok)": "new_tutor_ok",
        "new tutor (bad)": "new_tutor_bad",
        "new tutor": "new_tutor",
        "bad": "bad",
    }
    for k in sorted(repl.keys(), key=len, reverse=True):
        if s == k:
            return repl[k]
    return s

def can_pair(my_rating_raw: str, cand_rating_raw: str) -> bool:
    """Правило пар по рейтингам из ТЗ."""
    my   = _norm_rating(my_rating_raw)
    cand = _norm_rating(cand_rating_raw)

    NEVER = {"bad", "new_tutor_bad"}
    if cand in NEVER:
        return False

    HIGH  = {"amazing", "good", "new_tutor_good"}
    OKISH = {"ok", "new_tutor_ok"}

    if cand in OKISH:
        return my not in (HIGH | {"new_tutor"})  # не к high и не к 'new_tutor'
    if cand == "new_tutor":
        return my not in HIGH                    # не к high
    if cand in HIGH:
        return True

    return True  # неизвестные ярлыки — разрешаем


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


# --- объединённые Matches (time±120 ИЛИ suffix3 равен), + базовые условия ---
def add_matches_combined(df: pd.DataFrame, new_col_name="Matches") -> pd.DataFrame:
    if df.empty:
        return df

    colB, colE, colF, colG = df.columns[1], df.columns[4], df.columns[5], df.columns[6]
    colI, colK             = df.columns[8], df.columns[10]
    rating_col             = _find_rating_col(df)

    f_vals = df[colF].astype(str).str.strip()
    g_vals = df[colG].astype(str).str.strip()
    b_vals = df[colB].astype(str).fillna("").str.upper()
    k_num  = pd.to_numeric(df[colK], errors="coerce")
    i_mins = df[colI].astype(str).str.strip().apply(_time_to_minutes)
    suf3   = b_vals.apply(_b_suffix3)
    b_is_prm = b_vals.str.contains("PRM", na=False)

    r_vals = df[rating_col].astype(str) if rating_col else pd.Series("", index=df.index)
    slots_col = _find_free_slots_col(df)

    lines, counts = [], []
    for i in range(len(df)):
        same_course = (f_vals == f_vals.iloc[i])
        same_age    = (g_vals == g_vals.iloc[i])

        base_k = k_num.iloc[i]
        close_k = pd.Series(False, index=df.index)
        if not pd.isna(base_k):
            close_k = (k_num.sub(base_k).abs() <= 1)

        base_t = i_mins.iloc[i]
        close_time = pd.Series(False, index=df.index)
        if not pd.isna(base_t):
            close_time = (i_mins.sub(base_t).abs() <= 120)

        same_suf = (suf3 == suf3.iloc[i])
        same_prm = (b_is_prm == b_is_prm.iloc[i])

        my_r = r_vals.iloc[i]
        ok_by_rating = r_vals.apply(lambda rr: can_pair(my_r, rr))

        mask = same_course & same_age & close_k & same_prm & ok_by_rating & (close_time | same_suf)
        mask.iloc[i] = False

        if mask.any():
            # Для вывода берём только B, E, K (+ Rating и Free slots)
            cols_take = [colB, colE, colK]
            cols_take = [c for c in cols_take if c in df.columns]
            sub = df.loc[mask, cols_take].copy()

            sub["_rating"] = r_vals.loc[sub.index].values
            sub["_slots"]  = df.loc[sub.index, slots_col].values if slots_col else ""

            lst = [
                f"- {row[colB]}, "
                f"Tutor: {row[colE]}, "
                f"Rating: {row['_rating']}, "
                f"lesson: {row[colK]}, "
                f"slots: {row['_slots']}"
                for _, row in sub.iterrows()
            ]
            lines.append("\n".join(lst)); counts.append(len(lst))
        else:
            lines.append(""); counts.append(0)

    out = df.copy()
    name = new_col_name
    while name in out.columns: name += "_x"
    cnt = f"{new_col_name}_count"
    while cnt in out.columns: cnt += "_x"
    out[name] = lines
    out[cnt]  = counts
    return out


# --- WideMatches: «широкая сетка» без времени/суффикса, только вперед и не дублируем Matches ---
def add_wide_matches_column(df: pd.DataFrame, new_col_name="WideMatches", exclude_col="Matches") -> pd.DataFrame:
    """
    WideMatches: без времени и без суффикса, теперь смотрим и вверх, и вниз;
    не исключаем те строки, у которых уже есть Matches.
    Условия: same course & same age & |K-K'| <= 1 & одинаковый PRM-статус & рейтинги совместимы.
    """
    if df.empty:
        return df

    colB, colE, colF, colG = df.columns[1], df.columns[4], df.columns[5], df.columns[6]
    colI, colK             = df.columns[8], df.columns[10]  # colI не используем, оставлен для совместимости
    rating_col             = _find_rating_col(df)

    f_vals = df[colF].astype(str).str.strip()
    g_vals = df[colG].astype(str).str.strip()
    b_vals = df[colB].astype(str).fillna("").str.upper()
    k_num  = pd.to_numeric(df[colK], errors="coerce")
    b_is_prm = b_vals.str.contains("PRM", na=False)

    r_vals = df[rating_col].astype(str) if rating_col else pd.Series("", index=df.index)
    slots_col = _find_free_slots_col(df)

    # Раньше мы исключали те, у кого уже есть Matches; теперь — нет.
    # already = df[exclude_col].astype(str).str.strip().ne("") if exclude_col in df.columns else pd.Series(False, index=df.index)
    # Теперь просто:
    already = pd.Series(False, index=df.index)

    lines, counts = [], []
    for i in range(len(df)):
        same_course = (f_vals == f_vals.iloc[i])
        same_age    = (g_vals == g_vals.iloc[i])

        base_k = k_num.iloc[i]
        if pd.isna(base_k):
            close_k = pd.Series(False, index=df.index)
        else:
            close_k = (k_num.sub(base_k).abs() <= 1)

        same_prm = (b_is_prm == b_is_prm.iloc[i])

        my_r = r_vals.iloc[i]
        ok_by_rating = r_vals.apply(lambda rr: can_pair(my_r, rr))

        # Больше не ограничиваемся pos > pos[i] и не вычёркиваем already
        mask = same_course & same_age & close_k & same_prm & ok_by_rating
        mask.iloc[i] = False  # не матчим сами на себя
        mask = mask & ~already

        if mask.any():
            cols_take = [colB, colE, colK]
            cols_take = [c for c in cols_take if c in df.columns]
            sub = df.loc[mask, cols_take].copy()

            sub["_rating"] = r_vals.loc[sub.index].values
            sub["_slots"]  = df.loc[sub.index, slots_col].values if slots_col else ""

            lst = [
                f"- {row[colB]}, "
                f"Tutor: {row[colE]}, "
                f"Rating: {row['_rating']}, "
                f"lesson: {row[colK]}, "
                f"slots: {row['_slots']}"
                for _, row in sub.iterrows()
            ]
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

        st.markdown("""
    ### Legend
    
    **Which rows are included**
    - Status: **Active**
    - **Lesson number**: 4–31 (inclusive)
    - **Alter**: blank/empty/NaN or **0**
    - **Flag P** and **Flag Q**: **not TRUE**
    - **Students transferred 1 time** ≤ 2; **Students transferred 2+ times** ≤ 0
    - **Free slots** ≥ 1 (Capacity − Paid), when both **Capacity** & **Paid** exist
    
    **Matches (strict)**
    - Same **Course** and same **Group age**
    - **Lesson number** within **±1**
    - Same **PRM** marker (both PRM or both not)
    - **Either** same local start time within **±120 minutes** **or** the same 3-letter suffix in **Group ID**
    - Rating pairing allowed:
      - **Bad** / **New tutor (Bad)** → never
      - **OK** / **New tutor (OK)** → not with **Amazing/Good/New tutor (Good)** and not with **New tutor**
      - **New tutor** → not with **Amazing/Good/New tutor (Good)**
      - **Amazing/Good/New tutor (Good)** → allowed with anyone
    - Excludes the current row itself
    
    **WideMatches (broad)**
    - Same **Course** and **Group age**
    - **Lesson number** within **±1**
    - Same **PRM**
    - Rating pairing allowed (same rules as above)
    - **No** time/suffix requirement
    - Looks **both directions** and **can include** rows already listed in **Matches**
    """)
    st.divider()


    # --- Source (hidden, no UI) ---
    sheet_id = SHEET_ID
    ws_name  = WS_NAME
    
    try:
        client = _authorize_client()
        sh = client.open_by_key(sheet_id)
        ws_names = [ws.title for ws in sh.worksheets()]
        if ws_name not in ws_names and ws_names:
            ws_name = ws_names[0]
    except Exception:
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
    filtered = add_matches_combined(filtered, new_col_name="Matches")
    filtered = add_wide_matches_column(filtered, new_col_name="WideMatches", exclude_col="Matches")

    c1, c2 = st.columns(2)
    c1.caption(f"Rows total: {len(df)}")

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
    col_paid_pct   = _pick_col(dff, {"paid %", "paid percent", "paid percentage", "paid pct"})

    # колонки matches
    col_m_text  = "Matches"           if "Matches"           in dff.columns else None
    col_m_cnt   = "Matches_count"     if "Matches_count"     in dff.columns else None
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
        dff = _apply_ms(dff, col_paid_pct,   st.multiselect("Paid %",                    _ms_options(dff, col_paid_pct)))
        dff = _apply_ms(dff, col_transfer1,  st.multiselect("Students transferred 1 time", _ms_options(dff, col_transfer1)))

        # пороги по количеству матчей
        if col_m_cnt: 
            min_m = st.number_input("Min Matches",      min_value=0, value=0, step=1)
            dff = dff[pd.to_numeric(dff[col_m_cnt], errors="coerce").fillna(0).astype(int) >= min_m]
        if col_w_cnt:
            min_w = st.number_input("Min Wide matches", min_value=0, value=0, step=1)
            dff = dff[pd.to_numeric(dff[col_w_cnt], errors="coerce").fillna(0).astype(int) >= min_w]

        # поиск по тексту матчей
        q = st.text_input("Search in matches text")
        if q:
            qrx = re.escape(q)
            mask = pd.Series(False, index=dff.index)
            for col in [col_m_text, col_w_text]:
                if col:
                    mask |= dff[col].astype(str).str.contains(qrx, case=False, na=False)
            dff = dff[mask]

        # оставить только строки, где есть любые матчи
        only_any = st.checkbox("Only rows with any matches", value=False)
        if only_any:
            cnt = pd.Series(0, index=dff.index)
            for col in [col_m_cnt, col_w_cnt]:
                if col:
                    cnt = cnt.add(pd.to_numeric(dff[col], errors="coerce").fillna(0).astype(int), fill_value=0)
            dff = dff[cnt > 0]

    st.success(f"Filtered rows: {len(dff)}")

    # --- Причесанный вывод в заданном порядке ---
    cols_all = list(dff.columns)
    def col(idx):
        return cols_all[idx] if idx < len(cols_all) else None
    
    colA, colB, colC = col(0), col(1), col(2)
    colE, colF, colG = col(4), col(5), col(6)
    colI, colJ, colK = col(8), col(9), col(10)
    colL, colN, colO = col(11), col(13), col(14)
    
    # найдём колонку рейтинга и поставим её сразу после Tutor ID
    rating_colname = _find_rating_col(dff)  # "Rating_BP" или "Rating"
    # добавим Free slots и Paid % (они появились в dff, если нашлись Paid/Capacity)
    desired = [
        colA, colB, colE, colO, rating_colname,
        colF, colG, colI,             # базовые поля
        col_capacity, col_paid, "Free slots", "Paid %",  # Capacity / Paid / Free / Paid %
        colK, colC, colN, colL,       # прочее
        "Matches_count", "Matches",
        "WideMatches_count", "WideMatches",
    ]
    display_cols = [c for c in desired if (c is not None and c in dff.columns)]
    
    # убираем дубликаты имён, сохраняя порядок
    seen = set()
    display_cols = [c for c in display_cols if not (c in seen or seen.add(c))]
    
    curated = dff.loc[:, display_cols].copy()
    
    # дополнительная страховка: если всё же где-то повторилось имя — дропаем дубликаты
    curated = curated.loc[:, ~curated.columns.duplicated()]

    # жёстко переименуем рейтинг
    if rating_colname and rating_colname in curated.columns:
        curated.rename(columns={rating_colname: "Rating"}, inplace=True)
    
    # --- чистим невидимые символы и скрытые "пустоты", затем убираем пустые строки ---
    def _to_na(v):
        if v is None or pd.isna(v):
            return pd.NA
        if isinstance(v, str):
            s = (v.replace("\u00A0", " ")
                   .replace("\u200B", "")
                   .replace("\u200C", "")
                   .replace("\u200D", "")
                   .replace("\uFEFF", "")
                   .strip())
            if s == "" or s.lower() in {"nan", "none", "null", "na"}:
                return pd.NA
            return s
        return v
    
    curated = curated.applymap(_to_na)
    
    # если все видимые значения пустые — строку удаляем (ноль в счётчиках не считается «данными»)
    count_cols = [c for c in ["Matches_count","WideMatches_count"] if c in curated.columns]
    text_cols  = [c for c in ["Matches","WideMatches"] if c in curated.columns]
    base_cols  = [c for c in curated.columns if c not in (count_cols + text_cols)]
    
    has_base   = curated[base_cols].notna().any(axis=1) if base_cols else False
    has_text   = curated[text_cols].notna().any(axis=1) if text_cols else False
    has_counts = (sum(pd.to_numeric(curated[c], errors="coerce").fillna(0).astype(int) for c in count_cols) > 0) if count_cols else False
    
    curated = curated[ has_base | has_text | has_counts ].reset_index(drop=True)

    # --- Рендер: широкая таблица без «висячих» пустых рядов ---
    ROW, HEADER, PAD = 34, 39, 8
    table_h = min(700, HEADER + ROW * max(1, len(curated)))
    
    # Настроим ширины колонок
    cfg = {}
    for c in ["BO", "Group", "Tutor", "Course", "Matches", "WideMatches"]:
        if c in curated.columns:
            cfg[c] = st.column_config.TextColumn(label=c, width="large")
    for c in ["Lesson number", "Capacity", "Paid students", "Free slots", "Paid %", "Students transferred 1 time",
              "Module", "Group age", "Local time",
              "Matches_count", "WideMatches_count"]:
        if c in curated.columns:
            cfg[c] = st.column_config.TextColumn(label=c, width="small")
    
    st.dataframe(
        curated,
        use_container_width=True,
        height=table_h,
        column_config=cfg,
    )
    
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
