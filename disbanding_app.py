import os
import json
import io
import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ==== Page / UX ====
st.set_page_config(
    page_title="Disbanding | Initial Export",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==== Constants (можно переопределить секретами/переменными окружения) ====
DEFAULT_SHEET_ID = "1Jbb4p1cZCo67ZRiW5cmFUq-c9ijo5VMH_hFFMVYeJk4"
DEFAULT_WS_NAME  = "data"

# внешний шит для Group age
EXT_GROUPS_SS_ID = "1u_NwMt3CVVgozm04JGmccyTsNZnZGiHjG5y0Ko3YdaY"
EXT_GROUPS_WS    = "Groups & Teachers"

# внешний шит с рейтингом (их A -> колонка BP)
RATING_SS_ID = "1HItT2-PtZWoldYKL210hCQOLg3rh6U1Qj6NWkBjDjzk"
RATING_WS    = "Rating"

SHEET_ID = os.getenv("GSHEET_ID") or st.secrets.get("GSHEET_ID", DEFAULT_SHEET_ID)
WS_NAME  = os.getenv("GSHEET_WS") or st.secrets.get("GSHEET_WS", DEFAULT_WS_NAME)

SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

def _authorize_client():
    # Ключ берём из ENV GCP_SERVICE_ACCOUNT (JSON-строка) или из Streamlit Secrets
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

    # A:R (18 колонок). Первая строка — заголовки.
    values = ws.get(
    "A:R",
    value_render_option="UNFORMATTED_VALUE",     # ← числа вернутся числами
    date_time_render_option="FORMATTED_STRING"   # даты/время пусть останутся строками
    )

    if not values:
        return pd.DataFrame()

    header = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=header)

    # Нормализация пустых
    df = df.replace({"": pd.NA})
    return df

def adjust_local_time_minus_3(df: pd.DataFrame) -> pd.DataFrame:
    """Сдвигает колонку I (Local time) на -3 часа. Если только время — HH:MM, иначе YYYY-MM-DD HH:MM."""
    if df.empty:
        return df

    # попытка найти по названию
    col = None
    for c in df.columns:
        name = str(c).strip().lower().replace("_", " ")
        if name == "local time":
            col = c
            break
    # fallback: колонка I (индекс 8)
    if col is None:
        if len(df.columns) >= 9:
            col = df.columns[8]
        else:
            st.info("Колонка I (Local time) не найдена — сдвиг времени пропущен.")
            return df

    s = df[col].astype(str).str.strip()
    # маска "только время" вида HH:MM или HH:MM:SS
    time_only_mask = s.str.match(r"^\d{1,2}:\d{2}(:\d{2})?$", na=False)

    # обработка time-only: вручную крутим часы (мод 24)
    def _shift_time_str(v: str) -> str:
        parts = v.split(":")
        h = int(parts[0]); m = int(parts[1]); sec = int(parts[2]) if len(parts) > 2 else None
        h = (h - 3) % 24
        return f"{h:02d}:{m:02d}" + (f":{sec:02d}" if sec is not None else "")

    out = pd.Series(pd.NA, index=df.index, dtype="object")
    out.loc[time_only_mask] = s.loc[time_only_mask].apply(_shift_time_str)

    # обработка дата+время
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
    df[col] = out.where(out.notna(), df[col])  # оставим исходные, если парсинг не удался
    return df

@st.cache_data(show_spinner=False, ttl=300)
def load_group_age_map(sheet_id: str = EXT_GROUPS_SS_ID, worksheet_name: str = EXT_GROUPS_WS) -> dict:
    """Грузит соответствие: их A -> их E из внешнего шита."""
    client = _authorize_client()
    ws = client.open_by_key(sheet_id).worksheet(worksheet_name)
    vals = ws.get("A:E")
    if not vals or len(vals) < 2:
        return {}
    rows = vals[1:]  # без заголовка
    mapping = {}
    for r in rows:
        if len(r) >= 5:
            key = str(r[0]).strip()
            val = r[4]
            if key:
                mapping[key] = val
    return mapping

def replace_group_age_from_map(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Заменяет нашу G (Group age) на значение из mapping, где key = наша B."""
    if df.empty or not mapping:
        return df.copy()

    dff = df.copy()

    # найдём B и G: сперва по именам, иначе по индексам (B=1, G=6)
    colB = None
    for c in dff.columns:
        if str(c).strip().lower().replace("_", " ") in ("b", "group id", "group", "group title", "group_name", "group name"):
            colB = c
            break
    if colB is None:
        colB = dff.columns[1] if len(dff.columns) >= 2 else None

    colG = None
    for c in dff.columns:
        if str(c).strip().lower().replace("_", " ") == "group age":
            colG = c
            break
    if colG is None:
        colG = dff.columns[6] if len(dff.columns) >= 7 else None

    if colB is None or colG is None:
        st.info("Не удалось определить колонки B или G — замена Group age пропущена.")
        return dff

    keys = dff[colB].astype(str).str.strip()
    new_vals = keys.map(lambda k: mapping.get(k, pd.NA))
    dff[colG] = new_vals.where(new_vals.notna() & (new_vals.astype(str).str.strip() != ""), dff[colG])
    return dff

@st.cache_data(show_spinner=False, ttl=300)
def load_rating_bp_map(sheet_id: str = RATING_SS_ID, worksheet_name: str = RATING_WS) -> dict:
    """Грузит соответствие: их A -> их BP из внешнего шита 'Rating'."""
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
    """Добавляет столбец с BP-оценкой, где ключ = наша колонка O (их A)."""
    if df.empty or not mapping:
        return df.copy()
    if len(df.columns) < 15:
        st.info("Колонка O отсутствует — Rating_BP не добавлен.")
        return df.copy()

    colO = df.columns[14]  # O
    keys = df[colO].astype(str).str.strip()

    out = df.copy()
    # добавляем В КОНЕЦ, чтобы не сдвигать P/Q/R
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

    # D == active
    d_active = df[colD].astype(str).str.strip().str.lower() == "active"

    # K > 3 и < 32
    k_num = pd.to_numeric(df[colK], errors="coerce")
    k_ok = k_num.notna() & (k_num > 3) & (k_num < 32)

    # R пусто
    r_blank = df[colR].isna() | (df[colR].astype(str).str.strip() == "")

    # P/Q не TRUE (ловим и bool, и строки "TRUE")
    p_true = (df[colP] == True) | (df[colP].astype(str).str.strip().str.lower() == "true")
    q_true = (df[colQ] == True) | (df[colQ].astype(str).str.strip().str.lower() == "true")

    # L/M как числа
    l_num = pd.to_numeric(df[colL], errors="coerce")
    m_num = pd.to_numeric(df[colM], errors="coerce")

    # исключаем отдельно: M>0 и L>2
    exclude_m = (m_num > 0)
    exclude_l = (l_num > 2)

    mask = d_active & k_ok & r_blank & ~p_true & ~q_true & ~exclude_m & ~exclude_l

    # --- Доп. фильтр: Paid students / Capacity < 50% ---
    def _norm(s: str) -> str:
        return str(s).strip().lower().replace("_", " ").replace("-", " ")

    paid_aliases = {"paid students", "paid student", "paid"}
    cap_aliases  = {"capacity", "cap"}

    colPaid = None
    colCap  = None
    for c in df.columns:
        n = _norm(c)
        if (colPaid is None) and (n in paid_aliases):
            colPaid = c
        if (colCap is None) and (n in cap_aliases):
            colCap = c
        if colPaid is not None and colCap is not None:
            break

    if colPaid is not None and colCap is not None:
        paid_num = pd.to_numeric(df[colPaid], errors="coerce")
        cap_num  = pd.to_numeric(df[colCap],  errors="coerce")
        ratio_ge_50 = (cap_num > 0) & ((paid_num / cap_num) >= 0.5)
        mask = mask & ~ratio_ge_50
    else:
        st.info("⚠️ Не нашёл колонки 'Paid students' и/или 'Capacity' — фильтр по 50% пропущен.")

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
import numpy as np

def _time_to_minutes(v: str) -> float:
    """HH:MM или YYYY-MM-DD HH:MM -> минуты от начала суток. Если не парсится — NaN."""
    if v is None or str(v).strip() == "" or pd.isna(v):
        return np.nan
    s = str(v).strip()
    # чистое время?
    if pd.Series([s]).str.match(r"^\d{1,2}:\d{2}(:\d{2})?$", na=False).iloc[0]:
        parts = s.split(":")
        h = int(parts[0]); m = int(parts[1])
        return h * 60 + m
    # дата+время
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
    """Для каждой строки собирает совпадения и пишет их в столбец new_col_name."""
    if df.empty:
        return df

    # индексы колонок по A:R
    colB, colE, colF, colG, colI, colJ, colK = df.columns[1], df.columns[4], df.columns[5], df.columns[6], df.columns[8], df.columns[9], df.columns[10]
    rating_col = _find_rating_col(df)

    # подготовим векторы
    f_vals = df[colF].astype(str).str.strip()
    g_vals = df[colG].astype(str).str.strip()
    i_vals = df[colI].astype(str).str.strip()
    i_mins = i_vals.apply(_time_to_minutes)
    k_num  = pd.to_numeric(df[colK], errors="coerce")
    r_vals = df[rating_col].astype(str).str.strip() if rating_col else pd.Series("", index=df.index)

    good_l = set(x.lower() for x in good_set)
    bad_l  = set(x.lower() for x in bad_set)
    r_low  = r_vals.str.lower()

    lines = []
    n = len(df)

    for i in range(n):
        # маски условий
        mF = f_vals == f_vals.iloc[i]
        mG = g_vals == g_vals.iloc[i]

        base_t = i_mins.iloc[i]
        mI = pd.Series(False, index=df.index)
        if not np.isnan(base_t):
            mI = (i_mins.sub(base_t).abs() <= 120)  # ±2 часа

        base_k = k_num.iloc[i]
        mK = pd.Series(False, index=df.index)
        if not np.isnan(base_k):
            mK = (k_num.sub(base_k).abs() <= 1)

        # рейтинг
        ri = r_low.iloc[i] if rating_col else ""
        mR = (~r_low.isin(bad_l)) & ( (r_low == ri) | (r_low.isin(good_l)) )

        mask = mF & mG & mI & mK & mR
        mask.iloc[i] = False  # не включаем саму строку

        if mask.any():
            sub = df.loc[mask, [colB, colI, colJ, colK, colE]]
            if rating_col and rating_col in df.columns:
                sub = sub.assign(_rating=df.loc[mask, rating_col].values)
            else:
                sub = sub.assign(_rating="")

            # формируем строки
            lst = []
            for _, row in sub.iterrows():
                lst.append(f"{row[colB]}, {row[colI]}, {row[colJ]}, K: {row[colK]}, E: {row[colE]}, Rating: {row['_rating']}")
            lines.append("\n".join(lst))
        else:
            lines.append("")

    out = df.copy()
    # безопасное имя столбца
    name = new_col_name
    while name in out.columns:
        name += "_x"
    out[name] = lines
    return out

def main():
    st.title("Initial export (A:R, D='active', K < 32, R empty, P/Q != TRUE)")

    # --- Sidebar: источник, выбор вкладки, ссылки ---
    with st.sidebar:
        st.header("Source")
        sheet_id = st.text_input("Google Sheet ID", value=SHEET_ID)
        ws_name  = st.text_input("Worksheet", value=WS_NAME)

        # Показать список вкладок и ссылки
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

    # --- Загрузка данных ---
    with st.spinner("Loading data from Google Sheets…"):
        df = load_sheet_df(sheet_id, ws_name)

    if df.empty:
        st.warning(f"Пусто: проверь вкладку '{ws_name}' и доступ сервисного аккаунта (Viewer/Editor).")
        st.stop()

    # --- Сдвиг времени I (Local time) на -3 часа ---
    df = adjust_local_time_minus_3(df)

    # --- Подмена Group age (G) из внешнего шита: их A -> наша B, берём их E ---
    mapping = load_group_age_map()
    df = replace_group_age_from_map(df, mapping)

    # --- Подтянуть BP-рейтинг по нашему O (их A -> BP) ---
    rating_map = load_rating_bp_map()
    df = add_rating_bp_by_O(df, rating_map, new_col_name="Rating_BP")

    # --- Фильтр по условиям задачи ---
    filtered = filter_df(df)
    
    # --- Подбор совпадений (B, I, J, K, E, Rating) по правилам ---
    filtered = add_matches_column(filtered, new_col_name="Matches")

    # --- Верхняя панель метрик ---
    c1, c2 = st.columns(2)
    c1.caption(f"Rows total: {len(df)}")
    c2.success(f"Filtered rows: {len(filtered)}")

    # --- Таблица ---
    st.data_editor(filtered, use_container_width=True, disabled=True)

    # --- Экспорт ---
    export_col1, export_col2 = st.columns(2)
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    with export_col1:
        st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="filtered_export.csv", mime="text/csv")

    xlsx_buf = to_excel_bytes(filtered)
    with export_col2:
        if xlsx_buf:
            st.download_button("⬇️ Download XLSX", data=xlsx_buf, file_name="filtered_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.caption("Для XLSX установи пакет `xlsxwriter` (или оставь CSV).")

    # --- Обновить (сброс кеша) ---
    if st.button("Refresh"):
        load_sheet_df.clear()
        load_group_age_map.clear()
        load_rating_bp_map.clear()
        st.rerun()

if __name__ == "__main__":
    main()
