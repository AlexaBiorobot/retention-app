import os
import json
import io
import re
import numpy as np
import streamlit as st
import pandas as pd
import gspread
import textwrap
from oauth2client.service_account import ServiceAccountCredentials
from gspread.exceptions import SpreadsheetNotFound, WorksheetNotFound

# ==== Page / UX ====  (ДОЛЖНО быть самым первым вызовом Streamlit)
st.set_page_config(
    page_title="Disbanding Brazil/Latam",
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

# --- NEW: источник для второй вкладки ---
EXTERNAL_SHEET_ID = "1XwyahhHC7uVzwfoErrvwrcruEjwewqIUp2u-6nvdSR0"
EXTERNAL_WS_NAME  = "data"

# --- NEW: рейтинг для новой вкладки (лист "Rating Col BU") ---
RATING2_SS_ID = "16QrbLtzLTV6GqyT8HYwzcwYIsXewzjUbM0Jy5i1fENE"
RATING2_WS    = "Rating"

# --- LATAM: источник Group age ---
LATAM_GROUPS_SS_ID = "16QrbLtzLTV6GqyT8HYwzcwYIsXewzjUbM0Jy5i1fENE"
LATAM_GROUPS_WS    = "Groups"


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


# --- Универсальный сдвиг "Local time" на заданное число часов ---
def adjust_local_time_offset(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    """
    Сдвигает колонку Local time на заданное число часов (hours может быть отрицательным).
    Ищет колонку 'Local time' по имени или берёт 9-ю колонку (I).
    """
    if df.empty:
        return df

    # Найдём колонку
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
    # время формата HH:MM(:SS)
    time_only_mask = s.str.match(r"^\d{1,2}:\d{2}(:\d{2})?$", na=False)

    def _shift_time_str(v: str) -> str:
        parts = v.split(":")
        h = int(parts[0])
        m = int(parts[1])
        sec = int(parts[2]) if len(parts) > 2 else None
        h = (h - hours) % 24
        return f"{h:02d}:{m:02d}" + (f":{sec:02d}" if sec is not None else "")

    out = pd.Series(pd.NA, index=df.index, dtype="object")

    # Сдвигаем чистое время
    out.loc[time_only_mask] = s.loc[time_only_mask].apply(_shift_time_str)

    # Сдвигаем дату-время
    dt_mask = (~time_only_mask) & s.ne("")
    if dt_mask.any():
        dt = pd.to_datetime(s[dt_mask], errors="coerce", dayfirst=False)
        miss = dt.isna()
        if miss.any():
            dt2 = pd.to_datetime(s[dt_mask][miss], errors="coerce", dayfirst=True)
            dt.loc[miss] = dt2
        dt = dt - pd.Timedelta(hours=hours)
        out.loc[dt_mask] = dt.dt.strftime("%Y-%m-%d %H:%M")

    out_df = df.copy()
    out_df[col] = out.where(out.notna(), df[col])
    return out_df

def adjust_local_time_minus_3(df: pd.DataFrame) -> pd.DataFrame:
    return adjust_local_time_offset(df, hours=3)

@st.cache_data(show_spinner=False, ttl=300)
def load_group_age_map(
    sheet_id: str = EXT_GROUPS_SS_ID,
    worksheet_name: str = EXT_GROUPS_WS
) -> dict:
    """
    Карта для Brazil: ключ = колонка A (Group/ID), значение = колонка E (Group age)
    из листа EXT_GROUPS_WS таблицы EXT_GROUPS_SS_ID.
    """
    try:
        client = _authorize_client()
        ws = client.open_by_key(sheet_id).worksheet(worksheet_name)
        # Берём только A:E, чтобы не тащить лишнее
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

    mapping: dict[str, str] = {}
    for r in vals[1:]:
        if len(r) >= 5:
            key = str(r[0]).strip()              # A — Group/ID
            val = (r[4] if r[4] is not None else "")  # E — Group age
            if key:
                mapping[key] = val
    return mapping


def replace_group_age_from_map(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Подставляем/обновляем 'Group age' по соответствию ключей из mapping с колонкой группы (из B или по синонимам).
    Если 'Group age' отсутствует — создадим её.
    Если mapping пустой или df пуст — вернём копию df без изменений.
    """
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    if df.empty or not mapping:
        return df.copy()

    dff = df.copy()

    # Нормализация названий столбцов
    def _norm(s: str) -> str:
        return str(s).strip().lower().replace("_", " ")

    # Пытаемся найти колонку группы по синонимам
    group_synonyms = {
        "group id", "group", "group title", "group name", "group_name", "b"
    }
    colB = None
    for c in dff.columns:
        if _norm(c) in group_synonyms:
            colB = c
            break
    if colB is None and len(dff.columns) >= 2:
        colB = dff.columns[1]  # fallback: вторая колонка (как B)

    # Ищем/создаём колонку Group age
    colG = None
    for c in dff.columns:
        if _norm(c) == "group age":
            colG = c
            break
    if colG is None:
        colG = "Group age"
        if colG not in dff.columns:
            dff[colG] = pd.NA

    if colB is None:
        # Нет ключевой колонки — ничего не подставляем
        return dff

    keys = dff[colB].astype(str).str.strip()
    new_vals = keys.map(lambda k: mapping.get(k, pd.NA))

    # Подставляем только непустые значения из mapping
    dff[colG] = new_vals.where(new_vals.notna() & (new_vals.astype(str).str.strip() != ""), dff[colG])
    return dff


@st.cache_data(show_spinner=False, ttl=300)
def load_group_age_map_latam(
    sheet_id: str = LATAM_GROUPS_SS_ID,
    worksheet_name: str = LATAM_GROUPS_WS
) -> dict:
    """
    LATAM: ключ = колонка A (Group/ID), значение = колонка D (Group age) с листа 'Groups'
    таблицы LATAM_GROUPS_SS_ID.
    """
    try:
        client = _authorize_client()
        ws = client.open_by_key(sheet_id).worksheet(worksheet_name)
        vals = ws.get("A:D")
    except SpreadsheetNotFound:
        st.warning("Не могу открыть LATAM_GROUPS_SS_ID. Проверь ID и доступ.")
        return {}
    except WorksheetNotFound:
        st.warning(f"Не найден лист '{worksheet_name}' в LATAM_GROUPS_SS_ID.")
        return {}
    except Exception as e:
        st.warning(f"Ошибка при чтении LATAM_GROUPS_SS_ID: {e}")
        return {}

    if not vals or len(vals) < 2:
        return {}

    mapping: dict[str, str] = {}
    for r in vals[1:]:
        if len(r) >= 4:
            key = str(r[0]).strip()             # A — Group/ID
            val = (r[3] if r[3] is not None else "")  # D — Group age
            if key:
                mapping[key] = val
    return mapping


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

@st.cache_data(show_spinner=False, ttl=300)
def load_rating_bu_map(sheet_id: str = RATING2_SS_ID, worksheet_name: str = RATING2_WS) -> dict:
    """
    Читает рейтинг из листа 'Rating': ключ = колонка A, значение = колонка BU.
    """
    try:
        client = _authorize_client()
        ws = client.open_by_key(sheet_id).worksheet(worksheet_name)
        vals = ws.get(
            "A:BU",  # BU = 73-я колонка (index 72)
            value_render_option="UNFORMATTED_VALUE",
            date_time_render_option="FORMATTED_STRING",
        )
    except SpreadsheetNotFound:
        st.warning("Не могу открыть таблицу RATING2_SS_ID. Проверь ID и доступ сервисного аккаунта.")
        return {}
    except WorksheetNotFound:
        st.warning(f"Не найден лист '{worksheet_name}' в RATING2_SS_ID.")
        return {}
    except Exception as e:
        st.warning(f"Ошибка при чтении RATING2_SS_ID: {e}")
        return {}

    if not vals or len(vals) < 2:
        return {}

    mapping = {}
    for r in vals[1:]:
        a  = str(r[0]).strip() if len(r) >= 1  else ""
        bu = r[72]             if len(r) >= 73 else None  # BU
        if a:
            mapping[a] = bu
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
    """Базовые фильтры + доп.: оставляем только строки, где (Capacity - Paid) >= 1.
       По R берём только те строки, где R == 0.
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

    # --- R: strictly == 0 ---
    r_num = pd.to_numeric(df[colR], errors="coerce")
    r_ok  = r_num == 0

    # P/Q (флаги), L/M (исключения)
    p_true = (df[colP] == True) | (df[colP].astype(str).str.strip().str.lower() == "true")
    q_true = (df[colQ] == True) | (df[colQ].astype(str).str.strip().str.lower() == "true")

    l_num = pd.to_numeric(df[colL], errors="coerce")
    m_num = pd.to_numeric(df[colM], errors="coerce")

    # Явные "ok"-маски для L и M
    l_ok = l_num.fillna(0) <= 2
    m_ok = m_num.fillna(0) == 0

    # Итоговая маска БЕЗ проверки свободных мест (добавим ниже)
    mask = d_active & k_ok & r_ok & ~p_true & ~q_true & l_ok & m_ok

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

    free_slots_values = None
    paid_pct_series = None
    if colPaid is not None and colCap is not None:
        paid_num = pd.to_numeric(df[colPaid], errors="coerce")
        cap_num  = pd.to_numeric(df[colCap],  errors="coerce")
        free_slots_values = (cap_num - paid_num)
        have_free  = (cap_num.notna() & paid_num.notna()) & (free_slots_values >= 1)
        mask = mask & have_free

        # Paid % = round(Paid/Capacity*100), как строка "NN%"
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(cap_num > 0, (paid_num / cap_num) * 100.0, np.nan)
        paid_pct_series = pd.Series(pct, index=df.index)
        paid_pct_series = paid_pct_series.apply(lambda x: f"{int(round(x))}%"
                                                if pd.notna(x) else pd.NA)

    out = df.loc[mask].copy()
    out[colK] = k_num.loc[out.index]

    # Добавим колонки Free slots и Paid % (если нашли Paid/Capacity)
    if (colPaid is not None) and (colCap is not None) and (free_slots_values is not None):
        out["Free slots"] = free_slots_values.loc[out.index]
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

def _minutes_to_hhmm(x) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        m = int(x)
    except Exception:
        return None
    h = (m // 60) % 24
    mm = m % 60
    return f"{h:02d}:{mm:02d}"

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
    Возвращает дневной суффикс из колонки B.
    Поддерживает и BR ('..._SAB-10' -> 'SAB'), и LATAM ('..._DO-10' -> 'DO').
    Берём часть после последнего '_' и из неё первые 2–3 буквы.
    """
    if s is None or pd.isna(s):
        return ""
    s = str(s).upper()
    parts = s.split("_")
    tail = parts[-1] if len(parts) >= 2 else ""
    if not tail:
        return ""
    letters = "".join(ch for ch in tail if ch.isalpha())
    # допускаем 2- и 3-буквенные коды
    return letters[:3]  # вернёт 'DO' (2) или 'SAB' (3); если букв нет — ""

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

        suf_i = suf3.iloc[i]
        if isinstance(suf_i, str) and len(suf_i) > 0:
            same_suf = (suf3 == suf_i) & (suf3.str.len() > 0)
        else:
            same_suf = pd.Series(False, index=df.index)
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


# --- WideMatches: без времени/суффикса, смотрим и вверх, и вниз; не исключаем тех, у кого уже есть Matches ---
def add_wide_matches_column(df: pd.DataFrame, new_col_name="WideMatches", exclude_col="Matches") -> pd.DataFrame:
    """
    WideMatches: без времени и без суффикса, смотрим и вверх, и вниз.
    НЕ включаем те кандидаты, которые уже присутствуют в Matches этой же строки.
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

        # Базовая широкая маска
        mask = same_course & same_age & close_k & same_prm & ok_by_rating
        mask.iloc[i] = False  # не матчим сами на себя

        # --- НОВОЕ: исключаем то, что уже есть в Matches этой строки ---
        if exclude_col in df.columns:
            ex_text = df.iloc[i][exclude_col]
            ex_set = set()
            if pd.notna(ex_text):
                for line in str(ex_text).splitlines():
                    # строки вида: "- <GroupID>, Tutor: ..., Rating: ..., lesson: ..., slots: ..."
                    m = re.match(r"^\s*-\s*(.*?),", line)
                    if m:
                        ex_set.add(m.group(1).strip())
            if ex_set:
                mask = mask & ~df[colB].astype(str).isin(ex_set)
        # --- конец нового блока ---

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

def exclude_c6_h_before_14d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Исключить строки, где C == 6 и H < (сегодня - 14 дней).
    C — 3-я колонка (index 2), H — 8-я (index 7) при исходном диапазоне A:R.
    """
    if df.empty or len(df.columns) < 8:
        return df

    col_c = df.columns[2]  # C
    col_h = df.columns[7]  # H

    c_num = pd.to_numeric(df[col_c], errors="coerce")

    # сначала пробуем ISO/US, затем fallback на dayfirst
    h_dt = pd.to_datetime(df[col_h], errors="coerce", dayfirst=False, infer_datetime_format=True)
    miss = h_dt.isna()
    if miss.any():
        h_dt.loc[miss] = pd.to_datetime(df.loc[miss, col_h], errors="coerce", dayfirst=True)

    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=14)
    mask_exclude = (c_num == 6) & h_dt.notna() & (h_dt < cutoff)
    return df.loc[~mask_exclude].copy()

def debug_filter_sequence(df, lesson_min=4, lesson_max=31):
    """Коротко показывает, сколько строк остаётся после каждого условия фильтра."""
    if df.empty:
        st.write("df is empty")
        return
    if len(df.columns) < 18:
        st.write(f"Expected ≥18 columns, got {len(df.columns)}")
        return

    colD, colK = df.columns[3], df.columns[10]
    colL, colM = df.columns[11], df.columns[12]
    colP, colQ, colR = df.columns[15], df.columns[16], df.columns[17]

    k_num = pd.to_numeric(df[colK], errors="coerce")
    r_num = pd.to_numeric(df[colR], errors="coerce")

    m_active = df[colD].astype(str).str.strip().str.lower() == "active"
    m_k      = k_num.notna() & (k_num >= lesson_min) & (k_num <= lesson_max)

    # --- R: strictly == 0 ---
    m_r      = r_num == 0

    m_p      = ~((df[colP] == True) | (df[colP].astype(str).str.strip().str.lower() == "true"))
    m_q      = ~((df[colQ] == True) | (df[colQ].astype(str).str.strip().str.lower() == "true"))
    m_l      = pd.to_numeric(df[colL], errors="coerce").fillna(0) <= 2
    m_m      = pd.to_numeric(df[colM], errors="coerce").fillna(0) == 0

    def _norm(s): return str(s).strip().lower().replace("_"," ").replace("-"," ")
    paid_aliases, cap_aliases = {"paid students","paid student","paid"}, {"capacity","cap"}
    colPaid = colCap = None
    for c in df.columns:
        n = _norm(c)
        if colPaid is None and n in paid_aliases: colPaid = c
        if colCap  is None and n in cap_aliases:  colCap  = c
    if colPaid is not None and colCap is not None:
        paid = pd.to_numeric(df[colPaid], errors="coerce")
        cap  = pd.to_numeric(df[colCap],  errors="coerce")
        m_free = (cap.notna() & paid.notna()) & ((cap - paid) >= 1)
    else:
        m_free = pd.Series(True, index=df.index)

    steps = [
        ("Active", m_active),
        (f"Lessons from {lesson_min}..{lesson_max}", m_k),
        ("Students with 0-2 balance", m_r),
        ("No disband", m_p),
        ("No merge", m_q),
        ("Students transferred 1 time ≤ 2", m_l),
        ("Students transferred 2+ times = 0", m_m),
        ("Free slots ≥ 1", m_free),
    ]
    m = pd.Series(True, index=df.index)
    st.markdown("### 🔗 Stepwise filter (intersection)")
    st.write("Start:", int(m.sum()))
    for name, mask in steps:
        prev_cnt = int(m.sum())
        m = m & mask
        st.write(f"after {name}: {int(m.sum())}  (−{prev_cnt - int(m.sum())})")
    st.write("Final:", int(m.sum()))

def _col_by_index(df: pd.DataFrame, idx: int) -> str | None:
    return df.columns[idx] if idx < len(df.columns) else None

def _series_bool(name, s):
    # для аккуратного отображения NaN → False
    return pd.Series(s.fillna(False).astype(bool), name=name)

def debug_matches_sequence(
    df: pd.DataFrame,
    strict: bool = True,
    sample_row: int | None = 0,
    exclude_col_for_wide: str | None = "Matches",
):
    if df.empty:
        st.write("df is empty")
        return

    colB = _col_by_index(df, 1)   # Group/ID
    colE = _col_by_index(df, 4)   # Tutor (только для вывода)
    colF = _col_by_index(df, 5)   # Course
    colG = _col_by_index(df, 6)   # Group age
    colI = _col_by_index(df, 8)   # Local time
    colK = _col_by_index(df, 10)  # Lesson number

    if any(c is None for c in [colB, colF, colG, colK]):
        st.warning("Ожидались колонки B,F,G,K по индексам (1,5,6,10). Проверь порядок столбцов.")
        return

    b_vals = df[colB].astype(str).fillna("").str.upper()
    f_vals = df[colF].astype(str).str.strip()
    g_vals = df[colG].astype(str).str.strip()
    k_num  = pd.to_numeric(df[colK], errors="coerce")
    i_mins = df[colI].astype(str).str.strip().apply(_time_to_minutes) if (colI in df.columns) else pd.Series(np.nan, index=df.index)
    b_is_prm = b_vals.str.contains("PRM", na=False)
    rating_col = _find_rating_col(df)
    r_vals = df[rating_col].astype(str) if rating_col else pd.Series("", index=df.index)
    suf3 = b_vals.apply(_b_suffix3)

    i = 0 if sample_row is None else int(sample_row)
    i = max(0, min(i, len(df) - 1))

    # Заголовок + quick context
    st.markdown(f"#### Debug for {'Matches (strict)' if strict else 'WideMatches'} — sample row: {i}")
    st.write({
        "Group": df.iloc[i][colB],
        "Course": df.iloc[i][colF] if colF in df.columns else None,
        "Age": df.iloc[i][colG] if colG in df.columns else None,
        "Lesson": df.iloc[i][colK] if colK in df.columns else None,
        "Local time (HH:MM)": _minutes_to_hhmm(i_mins.iloc[i]),
        "Suffix(_b_suffix3)": suf3.iloc[i],
        "PRM?": bool(b_is_prm.iloc[i]),
        "Rating": r_vals.iloc[i] if len(r_vals) else None,
    })

    # Базовые маски
    same_course = (f_vals == f_vals.iloc[i])
    same_age    = (g_vals == g_vals.iloc[i])
    base_k = k_num.iloc[i]
    close_k = (k_num.sub(base_k).abs() <= 1) if not pd.isna(base_k) else pd.Series(False, index=df.index)
    same_prm = (b_is_prm == b_is_prm.iloc[i])
    my_r = r_vals.iloc[i] if len(r_vals) else ""
    ok_by_rating = r_vals.apply(lambda rr: can_pair(my_r, rr)) if len(r_vals) else pd.Series(True, index=df.index)

    # Строгая «калитка»
    if strict:
        base_t = i_mins.iloc[i]
        close_time = (i_mins.sub(base_t).abs() <= 120) if not pd.isna(base_t) else pd.Series(False, index=df.index)
        suf_i = suf3.iloc[i]
        same_suf = (suf3 == suf_i) & (suf3.str.len() > 0) if isinstance(suf_i, str) and len(suf_i) > 0 else pd.Series(False, index=df.index)
        final_gate = (close_time | same_suf)
    else:
        final_gate = pd.Series(True, index=df.index)

    # Пошаговая стыковка
    steps = [
        ("Same course",       _series_bool("same_course", same_course)),
        ("Same group age",    _series_bool("same_age",    same_age)),
        ("Lesson ±1",         _series_bool("close_k",     close_k)),
        ("Same PRM flag",     _series_bool("same_prm",    same_prm)),
        ("Rating-compatible", _series_bool("ok_by_rating",ok_by_rating)),
    ]
    if strict:
        steps.append(("Time±120 OR same suffix", _series_bool("final_gate", final_gate)))
    else:
        steps.append(("Wide gate (no time/suffix)", _series_bool("final_gate", final_gate)))

    m = pd.Series(True, index=df.index)
    st.markdown("##### Stepwise intersection")
    st.write("Start:", int(m.sum()))
    for name, mask in steps:
        prev = int(m.sum())
        m = m & mask
        if i < len(m):
            m.iloc[i] = False  # не матчим сами на себя
        st.write(f"after {name}: {int(m.sum())}  (−{prev - int(m.sum())})")

    # ВАЖНО: исключаем то, что уже попало в Matches (как в add_wide_matches_column)
    if not strict and exclude_col_for_wide and (exclude_col_for_wide in df.columns):
        ex_text = df.iloc[i][exclude_col_for_wide]
        ex_set = set()
        if pd.notna(ex_text):
            for line in str(ex_text).splitlines():
                m_line = re.match(r"^\s*-\s*(.*?),", line)
                if m_line:
                    ex_set.add(m_line.group(1).strip())
        if ex_set:
            prev = int(m.sum())
            m = m & ~df[colB].astype(str).isin(ex_set)
            st.write(f"after Exclude already in '{exclude_col_for_wide}': {int(m.sum())}  (−{prev - int(m.sum())})")

    st.write("Final:", int(m.sum()))

    # Вывод кандидатов
    if int(m.sum()) > 0:
        cols_for_view = [c for c in [colB, colE, colF, colG, colK] if c in df.columns]
        sub = df.loc[m, cols_for_view].copy()
        if colI in df.columns:
            sub["Local time"] = i_mins.loc[sub.index].apply(_minutes_to_hhmm)
        if len(sub) > 20:
            st.write(sub.head(20))
            st.caption(f"... and {len(sub)-20} more")
        else:
            st.write(sub)
    else:
        st.info("Нет кандидатов на выходе для выбранной строки.")


def main():
    st.title("Disbanding Brazil/Latam")

    st.markdown(textwrap.dedent("""\
    ### Legend

    **Which rows are included**
    - Status: **Active**
    - **Lesson number**: 4–31 (inclusive)
    - **No 0-2 lessons left**
    - **Not flagged as "Do not disband" or "Do not merge"**
    - **Students transferred 1 time** ≤ 2; **Students transferred 2+ times** ≤ 0
    - **Free slots** ≥ 1 (Capacity − Paid), when both **Capacity** & **Paid** exist

    **Matches (strict)**
    - Same **Course** and same **Group age**
    - **Lesson number** within **±1**
    - Same **PRM** marker (both PRM or both not)
    - **Either** same local start time within **±120 minutes** **or** the same **day**
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
    - **No** time/day requirement
    """))
    st.divider()

    # === ДВЕ ВКЛАДКИ: основная и внешняя ===
    tabs = st.tabs(["Brazil groups", "Latam groups"])

    # ---------- TAB 1: ОСНОВНАЯ (как было) ----------
    with tabs[0]:
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

        with st.spinner("Loading Brazil data…"):
            df = load_sheet_df(sheet_id, ws_name)

        if df.empty:
            st.warning(f"Пусто: проверь вкладку '{ws_name}' и доступ сервисного аккаунта (Viewer/Editor).")
        else:
            df = adjust_local_time_minus_3(df)
            mapping = load_group_age_map()
            df = replace_group_age_from_map(df, mapping)

            rating_map = load_rating_bp_map()  # старый источник рейтинга
            df = add_rating_bp_by_O(df, rating_map, new_col_name="Rating_BP")

            # --- Debug: пошаговый разбор фильтра (Main) ---
            with st.expander("Show filter breakdown", expanded=False):
                debug_filter_sequence(df, lesson_min=4, lesson_max=31)


            filtered = filter_df(df)
            filtered = add_matches_combined(filtered, new_col_name="Matches")
            filtered = add_wide_matches_column(filtered, new_col_name="WideMatches", exclude_col="Matches")

            # ⬇️ DEBUG блоки для пошагового логирования матчей
            if len(filtered) > 0:
                with st.expander("🧭 Debug Matches (strict)", expanded=False):
                    row_idx_strict = st.number_input(
                        "Sample row (0-based)",
                        min_value=0,
                        max_value=len(filtered) - 1,
                        value=0,
                        step=1,
                        key="dbg_row_strict_main",
                    )
                    debug_matches_sequence(filtered, strict=True, sample_row=row_idx_strict)
            
                with st.expander("🧭 Debug WideMatches", expanded=False):
                    row_idx_wide = st.number_input(
                        "Sample row (0-based)",
                        min_value=0,
                        max_value=len(filtered) - 1,
                        value=0,
                        step=1,
                        key="dbg_row_wide_main",
                    )
                    debug_matches_sequence(
                        filtered, strict=False, sample_row=row_idx_wide,
                        exclude_col_for_wide="Matches"
                    )
            else:
                st.info("Нет строк после фильтра — лог матчей скрыт.")

            
            c1, c2 = st.columns(2)
            c1.caption(f"Rows total: {len(df)}")


            # --- Sidebar Filters (MAIN) ---
            dff = filtered.copy()

            col_group      = _pick_col(dff, {"group", "group title", "group name", "group id"}, fallback_idx=1)
            col_tutor      = _pick_col(dff, {"tutor", "teacher", "tutor name", "teacher name"}, fallback_idx=4)
            col_tutor_id   = _pick_col(dff, {"tutor id", "teacher id", "id"}, fallback_idx=14)
            col_course     = _pick_col(dff, {"course"}, fallback_idx=5)
            col_group_age  = _pick_col(dff, {"group age", "age"}, fallback_idx=6)
            col_local_time = _pick_col(dff, {"local time", "localtime", "time (local)"}, fallback_idx=8)
            col_module     = _pick_col(dff, {"module"}, fallback_idx=2)
            col_lesson_num = _pick_col(dff, {"lesson number", "lesson", "lesson_num"}, fallback_idx=13)
            col_capacity   = _pick_col(dff, {"capacity", "cap"})
            col_paid       = _pick_col(dff, {"paid students", "paid student", "paid"})
            col_transfer1  = _pick_col(dff, {"students transferred 1 time", "transferred 1 time"}, fallback_idx=11)
            col_paid_pct   = _pick_col(dff, {"paid %", "paid percent", "paid percentage", "paid pct"})

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

            with st.sidebar.expander("Filters (Brazil)", expanded=True):
                dff = _apply_ms(dff, col_group,      st.multiselect("Group",      _ms_options(dff, col_group), key="ms_group_main"))
                dff = _apply_ms(dff, col_tutor,      st.multiselect("Tutor",      _ms_options(dff, col_tutor), key="ms_tutor_main"))
                dff = _apply_ms(dff, col_tutor_id,   st.multiselect("Tutor ID",   _ms_options(dff, col_tutor_id), key="ms_tid_main"))
                dff = _apply_ms(dff, col_course,     st.multiselect("Course",     _ms_options(dff, col_course), key="ms_course_main"))
                dff = _apply_ms(dff, col_group_age,  st.multiselect("Group age",  _ms_options(dff, col_group_age), key="ms_age_main"))
                dff = _apply_ms(dff, col_local_time, st.multiselect("Local time", _ms_options(dff, col_local_time), key="ms_time_main"))
                dff = _apply_ms(dff, col_module,     st.multiselect("Module",     _ms_options(dff, col_module), key="ms_module_main"))

                dff = _apply_ms(dff, col_lesson_num, st.multiselect("Lesson number", _ms_options(dff, col_lesson_num), key="ms_lesson_main"))
                dff = _apply_ms(dff, col_capacity,   st.multiselect("Capacity",      _ms_options(dff, col_capacity), key="ms_cap_main"))
                dff = _apply_ms(dff, col_paid,       st.multiselect("Paid students", _ms_options(dff, col_paid), key="ms_paid_main"))
                dff = _apply_ms(dff, col_paid_pct,   st.multiselect("Paid %",        _ms_options(dff, col_paid_pct), key="ms_paidpct_main"))
                dff = _apply_ms(dff, col_transfer1,  st.multiselect("Students transferred 1 time", _ms_options(dff, col_transfer1), key="ms_tr1_main"))

                if col_m_cnt:
                    min_m = st.number_input("Min Matches",      min_value=0, value=0, step=1, key="min_m_main")
                    dff = dff[pd.to_numeric(dff[col_m_cnt], errors="coerce").fillna(0).astype(int) >= min_m]
                if col_w_cnt:
                    min_w = st.number_input("Min Wide matches", min_value=0, value=0, step=1, key="min_w_main")
                    dff = dff[pd.to_numeric(dff[col_w_cnt], errors="coerce").fillna(0).astype(int) >= min_w]

                q = st.text_input("Search in matches text", key="q_main")
                if q:
                    qrx = re.escape(q)
                    mask = pd.Series(False, index=dff.index)
                    for col in [col_m_text, col_w_text]:
                        if col:
                            mask |= dff[col].astype(str).str.contains(qrx, case=False, na=False)
                    dff = dff[mask]

                only_any = st.checkbox("Only rows with any matches", value=False, key="only_any_main")
                if only_any:
                    cnt = pd.Series(0, index=dff.index)
                    for col in [col_m_cnt, col_w_cnt]:
                        if col:
                            cnt = cnt.add(pd.to_numeric(dff[col], errors="coerce").fillna(0).astype(int), fill_value=0)
                    dff = dff[cnt > 0]

            st.success(f"Filtered rows: {len(dff)}")

            cols_all = list(dff.columns)
            def col(idx):
                return cols_all[idx] if idx < len(cols_all) else None

            colA, colB, colC = col(0), col(1), col(2)
            colE, colF, colG = col(4), col(5), col(6)
            colI, colJ, colK = col(8), col(9), col(10)
            colL, colN, colO = col(11), col(13), col(14)

            rating_colname = _find_rating_col(dff)
            desired = [
                colA, colB, colE, colO, rating_colname,
                colF, colG, colI,
                _pick_col(dff, {"capacity","cap"}), _pick_col(dff, {"paid students","paid student","paid"}), "Free slots", "Paid %",
                colK, colC, colN, colL,
                "Matches_count", "Matches",
                "WideMatches_count", "WideMatches",
            ]
            display_cols = [c for c in desired if (c is not None and c in dff.columns)]
            seen = set()
            display_cols = [c for c in display_cols if not (c in seen or seen.add(c))]
            curated = dff.loc[:, display_cols].copy()
            curated = curated.loc[:, ~curated.columns.duplicated()]
            if rating_colname and rating_colname in curated.columns:
                curated.rename(columns={rating_colname: "Rating"}, inplace=True)

            def _to_na(v):
                if v is None or pd.isna(v): return pd.NA
                if isinstance(v, str):
                    s = (v.replace("\u00A0"," ").replace("\u200B","").replace("\u200C","").replace("\u200D","").replace("\uFEFF","").strip())
                    if s == "" or s.lower() in {"nan","none","null","na"}: return pd.NA
                    return s
                return v
            curated = curated.applymap(_to_na)

            count_cols = [c for c in ["Matches_count","WideMatches_count"] if c in curated.columns]
            text_cols  = [c for c in ["Matches","WideMatches"] if c in curated.columns]
            base_cols  = [c for c in curated.columns if c not in (count_cols + text_cols)]
            has_base   = curated[base_cols].notna().any(axis=1) if base_cols else False
            has_text   = curated[text_cols].notna().any(axis=1) if text_cols else False
            has_counts = (sum(pd.to_numeric(curated[c], errors="coerce").fillna(0).astype(int) for c in count_cols) > 0) if count_cols else False
            curated = curated[ has_base | has_text | has_counts ].reset_index(drop=True)

            ROW, HEADER, PAD = 34, 39, 8
            table_h = min(700, HEADER + ROW * max(1, len(curated)))

            cfg = {}
            for c in ["BO","Group","Tutor","Course","Matches","WideMatches"]:
                if c in curated.columns: cfg[c] = st.column_config.TextColumn(label=c, width="large")
            for c in ["Lesson number","Capacity","Paid students","Free slots","Paid %","Students transferred 1 time",
                      "Module","Group age","Local time","Matches_count","WideMatches_count"]:
                if c in curated.columns: cfg[c] = st.column_config.TextColumn(label=c, width="small")

            st.dataframe(curated, use_container_width=True, height=table_h, column_config=cfg)
            st.download_button("⬇️ Download CSV", curated.to_csv(index=False).encode("utf-8"),
                               file_name="curated_view.csv", mime="text/csv")

    # ---------- TAB 2: ВНЕШНИЙ ФАЙЛ + правило C/H + рейтинг из BU ----------
    with tabs[1]:
        with st.spinner("Loading Latam data…"):
            df_ext = load_sheet_df(EXTERNAL_SHEET_ID, EXTERNAL_WS_NAME)
            
    
        if df_ext.empty:
            st.warning(f"Пусто: проверь файл '{EXTERNAL_SHEET_ID}', вкладку '{EXTERNAL_WS_NAME}' и доступ.")
        else:
            # правило C/H
            df_ext = exclude_c6_h_before_14d(df_ext)
    
            # остальной пайплайн
            df_ext = adjust_local_time_offset(df_ext, hours=5)
    
            mapping = load_group_age_map_latam()
            df_ext = replace_group_age_from_map(df_ext, mapping)

    
            rating_map2 = load_rating_bu_map()   # <--- рейтинг из BU (лист Rating)
            df_ext = add_rating_bp_by_O(df_ext, rating_map2, new_col_name="Rating_BP")
    
            # диагностируем, найдены ли Capacity/Paid
            def _norm(s): 
                return str(s).strip().lower().replace("_"," ").replace("-"," ")
            colPaid = colCap = None
            for c in df_ext.columns:
                n = _norm(c)
                if colPaid is None and n in {"paid students","paid student","paid"}:
                    colPaid = c
                if colCap is None and n in {"capacity","cap"}:
                    colCap = c

            with st.expander("Show filter breakdown", expanded=False):
                debug_filter_sequence(df_ext, lesson_min=4, lesson_max=31)
            
            filtered = filter_df(df_ext)
    
            filtered = add_matches_combined(filtered, new_col_name="Matches")
    
            filtered = add_wide_matches_column(filtered, new_col_name="WideMatches", exclude_col="Matches")

            # ⬇️ DEBUG блоки для пошагового логирования матчей
            if len(filtered) > 0:
                with st.expander("🧭 Debug Matches (strict)", expanded=False):
                    row_idx_strict = st.number_input(
                        "Sample row (0-based)",
                        min_value=0,
                        max_value=len(filtered) - 1,
                        value=0,
                        step=1,
                        key="dbg_row_strict_ext",
                    )
                    debug_matches_sequence(filtered, strict=True, sample_row=row_idx_strict)
            
                with st.expander("🧭 Debug WideMatches", expanded=False):
                    row_idx_wide = st.number_input(
                        "Sample row (0-based)",
                        min_value=0,
                        max_value=len(filtered) - 1,
                        value=0,
                        step=1,
                        key="dbg_row_wide_ext",
                    )
                    debug_matches_sequence(
                        filtered, strict=False, sample_row=row_idx_wide,
                        exclude_col_for_wide="Matches"
                    )
            else:
                st.info("Нет строк после фильтра — лог матчей скрыт.")

            
            c1, c2 = st.columns(2)
            c1.caption(f"Rows total: {len(df_ext)}")


            # --- Sidebar Filters (EXTERNAL) — те же, но с другими key ---
            dff = filtered.copy()

            col_group      = _pick_col(dff, {"group","group title","group name","group id"}, fallback_idx=1)
            col_tutor      = _pick_col(dff, {"tutor","teacher","tutor name","teacher name"}, fallback_idx=4)
            col_tutor_id   = _pick_col(dff, {"tutor id","teacher id","id"}, fallback_idx=14)
            col_course     = _pick_col(dff, {"course"}, fallback_idx=5)
            col_group_age  = _pick_col(dff, {"group age","age"}, fallback_idx=6)
            col_local_time = _pick_col(dff, {"local time","localtime","time (local)"}, fallback_idx=8)
            col_module     = _pick_col(dff, {"module"}, fallback_idx=2)
            col_lesson_num = _pick_col(dff, {"lesson number","lesson","lesson_num"}, fallback_idx=13)
            col_capacity   = _pick_col(dff, {"capacity","cap"})
            col_paid       = _pick_col(dff, {"paid students","paid student","paid"})
            col_transfer1  = _pick_col(dff, {"students transferred 1 time","transferred 1 time"}, fallback_idx=11)
            col_paid_pct   = _pick_col(dff, {"paid %","paid percent","paid percentage","paid pct"})

            col_m_text  = "Matches"           if "Matches"           in dff.columns else None
            col_m_cnt   = "Matches_count"     if "Matches_count"     in dff.columns else None
            col_w_text  = "WideMatches"       if "WideMatches"       in dff.columns else None
            col_w_cnt   = "WideMatches_count" if "WideMatches_count" in dff.columns else None

            def _ms_options(df_, col):
                if not col or col not in df_.columns: return []
                return sorted(df_[col].dropna().astype(str).unique())

            def _apply_ms(df_, col, sel):
                if not col or not sel: return df_
                return df_[df_[col].astype(str).isin(sel)]

            with st.sidebar.expander("Filters (Latam)", expanded=True):
                dff = _apply_ms(dff, col_group,      st.multiselect("Group",      _ms_options(dff, col_group), key="ms_group_ext"))
                dff = _apply_ms(dff, col_tutor,      st.multiselect("Tutor",      _ms_options(dff, col_tutor), key="ms_tutor_ext"))
                dff = _apply_ms(dff, col_tutor_id,   st.multiselect("Tutor ID",   _ms_options(dff, col_tutor_id), key="ms_tid_ext"))
                dff = _apply_ms(dff, col_course,     st.multiselect("Course",     _ms_options(dff, col_course), key="ms_course_ext"))
                dff = _apply_ms(dff, col_group_age,  st.multiselect("Group age",  _ms_options(dff, col_group_age), key="ms_age_ext"))
                dff = _apply_ms(dff, col_local_time, st.multiselect("Local time", _ms_options(dff, col_local_time), key="ms_time_ext"))
                dff = _apply_ms(dff, col_module,     st.multiselect("Module",     _ms_options(dff, col_module), key="ms_module_ext"))

                dff = _apply_ms(dff, col_lesson_num, st.multiselect("Lesson number", _ms_options(dff, col_lesson_num), key="ms_lesson_ext"))
                dff = _apply_ms(dff, col_capacity,   st.multiselect("Capacity",      _ms_options(dff, col_capacity), key="ms_cap_ext"))
                dff = _apply_ms(dff, col_paid,       st.multiselect("Paid students", _ms_options(dff, col_paid), key="ms_paid_ext"))
                dff = _apply_ms(dff, col_paid_pct,   st.multiselect("Paid %",        _ms_options(dff, col_paid_pct), key="ms_paidpct_ext"))
                dff = _apply_ms(dff, col_transfer1,  st.multiselect("Students transferred 1 time", _ms_options(dff, col_transfer1), key="ms_tr1_ext"))

                if col_m_cnt:
                    min_m = st.number_input("Min Matches",      min_value=0, value=0, step=1, key="min_m_ext")
                    dff = dff[pd.to_numeric(dff[col_m_cnt], errors="coerce").fillna(0).astype(int) >= min_m]
                if col_w_cnt:
                    min_w = st.number_input("Min Wide matches", min_value=0, value=0, step=1, key="min_w_ext")
                    dff = dff[pd.to_numeric(dff[col_w_cnt], errors="coerce").fillna(0).astype(int) >= min_w]

                q = st.text_input("Search in matches text", key="q_ext")
                if q:
                    qrx = re.escape(q)
                    mask = pd.Series(False, index=dff.index)
                    for col in [col_m_text, col_w_text]:
                        if col:
                            mask |= dff[col].astype(str).str.contains(qrx, case=False, na=False)
                    dff = dff[mask]

                only_any = st.checkbox("Only rows with any matches", value=False, key="only_any_ext")
                if only_any:
                    cnt = pd.Series(0, index=dff.index)
                    for col in [col_m_cnt, col_w_cnt]:
                        if col:
                            cnt = cnt.add(pd.to_numeric(dff[col], errors="coerce").fillna(0).astype(int), fill_value=0)
                    dff = dff[cnt > 0]

            st.success(f"Filtered rows: {len(dff)}")

            cols_all = list(dff.columns)
            def col(idx):
                return cols_all[idx] if idx < len(cols_all) else None

            colA, colB, colC = col(0), col(1), col(2)
            colE, colF, colG = col(4), col(5), col(6)
            colI, colJ, colK = col(8), col(9), col(10)
            colL, colN, colO = col(11), col(13), col(14)

            rating_colname = _find_rating_col(dff)
            desired = [
                colA, colB, colE, colO, rating_colname,
                colF, colG, colI,
                _pick_col(dff, {"capacity","cap"}), _pick_col(dff, {"paid students","paid student","paid"}), "Free slots", "Paid %",
                colK, colC, colN, colL,
                "Matches_count", "Matches",
                "WideMatches_count", "WideMatches",
            ]
            display_cols = [c for c in desired if (c is not None and c in dff.columns)]
            seen = set()
            display_cols = [c for c in display_cols if not (c in seen or seen.add(c))]
            curated = dff.loc[:, display_cols].copy()
            curated = curated.loc[:, ~curated.columns.duplicated()]
            if rating_colname and rating_colname in curated.columns:
                curated.rename(columns={rating_colname: "Rating"}, inplace=True)

            def _to_na(v):
                if v is None or pd.isna(v): return pd.NA
                if isinstance(v, str):
                    s = (v.replace("\u00A0"," ").replace("\u200B","").replace("\u200C","").replace("\u200D","").replace("\uFEFF","").strip())
                    if s == "" or s.lower() in {"nan","none","null","na"}: return pd.NA
                    return s
                return v
            curated = curated.applymap(_to_na)

            count_cols = [c for c in ["Matches_count","WideMatches_count"] if c in curated.columns]
            text_cols  = [c for c in ["Matches","WideMatches"] if c in curated.columns]
            base_cols  = [c for c in curated.columns if c not in (count_cols + text_cols)]
            has_base   = curated[base_cols].notna().any(axis=1) if base_cols else False
            has_text   = curated[text_cols].notna().any(axis=1) if text_cols else False
            has_counts = (sum(pd.to_numeric(curated[c], errors="coerce").fillna(0).astype(int) for c in count_cols) > 0) if count_cols else False
            curated = curated[ has_base | has_text | has_counts ].reset_index(drop=True)

            ROW, HEADER, PAD = 34, 39, 8
            table_h = min(700, HEADER + ROW * max(1, len(curated)))

            cfg = {}
            for c in ["BO","Group","Tutor","Course","Matches","WideMatches"]:
                if c in curated.columns: cfg[c] = st.column_config.TextColumn(label=c, width="large")
            for c in ["Lesson number","Capacity","Paid students","Free slots","Paid %","Students transferred 1 time",
                      "Module","Group age","Local time","Matches_count","WideMatches_count"]:
                if c in curated.columns: cfg[c] = st.column_config.TextColumn(label=c, width="small")

            st.dataframe(curated, use_container_width=True, height=table_h, column_config=cfg)
            st.download_button("⬇️ Download CSV", curated.to_csv(index=False).encode("utf-8"),
                               file_name="curated_external.csv", mime="text/csv")


    if st.button("Refresh"):
        load_sheet_df.clear()
        load_group_age_map.clear()
        load_group_age_map_latam.clear()
        load_rating_bu_map.clear()
        load_rating_bp_map.clear()
        st.rerun()


if __name__ == "__main__":
    main()
