import os
import json
import textwrap
import unicodedata
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ===================== Page / UX =====================
st.set_page_config(
    page_title="Extra-classes Latam",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Simple password gate (как в исходном коде) ---
st.session_state.setdefault("is_authed", False)
if not st.session_state["is_authed"]:
    pwd = st.text_input("Password:", type="password")
    target_pwd = os.getenv("APP_PASSWORD") or st.secrets.get("APP_PASSWORD") or "Kodland123"
    if pwd == target_pwd:
        st.session_state["is_authed"] = True
        st.rerun()
    else:
        if pwd:
            st.error("Wrong password")
        st.stop()

st.title("📋 Extra-classes Latam")
st.caption("Use this instrument for extra-class analysis")

# ===================== Config =====================
DEFAULT_SHEET_ID = "1BtET9YSSLv1vSqWejO8tka4kdJygsnrlXYSEOavG4uA"
DEFAULT_WS_NAME  = "Form Responses 1"

SHEET_ID = os.getenv("GSHEET_ID") or st.secrets.get("GSHEET_ID", DEFAULT_SHEET_ID)
WS_NAME  = os.getenv("GSHEET_WS") or st.secrets.get("GSHEET_WS", DEFAULT_WS_NAME)

SCOPE = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

# ===================== Auth =====================

def _authorize_client():
    sa_json = os.getenv("GCP_SERVICE_ACCOUNT") or st.secrets.get("GCP_SERVICE_ACCOUNT")
    if not sa_json:
        st.error("Service account key not found. Put JSON into st.secrets['GCP_SERVICE_ACCOUNT'] or env var.")
        st.stop()
    try:
        sa_info = json.loads(sa_json)
    except Exception:
        st.error("GCP_SERVICE_ACCOUNT must be a JSON string (the full service account key).")
        st.stop()
    creds  = ServiceAccountCredentials.from_json_keyfile_dict(sa_info, SCOPE)
    client = gspread.authorize(creds)
    return client

# ===================== Data =====================

@st.cache_data(show_spinner=False, ttl=300)
def load_sheet_df(sheet_id: str, worksheet_name: str, rng: str = "A:Q") -> pd.DataFrame:
    client = _authorize_client()
    sh = client.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet_name)
    values = ws.get(
        rng,
        value_render_option="UNFORMATTED_VALUE",
        date_time_render_option="FORMATTED_STRING",
    )
    if not values:
        return pd.DataFrame()
    header = values[0]
    rows = values[1:]
    rows = [r + [None] * (len(header) - len(r)) for r in rows]
    df = pd.DataFrame(rows, columns=header)
    return df.replace({"": pd.NA})

# ===================== Helpers =====================

def _to_datetime_series(s: pd.Series) -> pd.Series:
    if s.empty:
        return pd.to_datetime(pd.Series([], dtype="object"))
    s_str = s.astype(str)
    dt = pd.to_datetime(s_str, errors="coerce", dayfirst=False, infer_datetime_format=True)
    miss = dt.isna()
    if miss.any():
        dt2 = pd.to_datetime(s_str[miss], errors="coerce", dayfirst=True, infer_datetime_format=True)
        dt.loc[miss] = dt2
    return dt

def _is_numeric_col(s: pd.Series) -> bool:
    if s.empty:
        return False
    num = pd.to_numeric(s, errors="coerce")
    non_na = num.notna().sum()
    filled = s.notna().sum()
    return filled > 0 and (non_na / max(1, filled)) >= 0.8

@st.cache_data(show_spinner=False)
def _unique_list_for_multiselect(col: pd.Series) -> list:
    vals = col.dropna().astype(str).str.strip()
    uniq = sorted(vals.unique())
    if col.isna().any():
        uniq = ["(blank)"] + uniq
    return uniq[:2000]

# ---- Robust header normalization & force-multiselect rules ----

def _norm_ascii(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.lower().strip().split())

def _main_label(s: str) -> str:
    """Return only the first header line without hints like '(range)'."""
    first = str(s).splitlines()[0]
    # trim parenthetical hints in first line
    if "(" in first:
        first = first.split("(")[0]
    return _norm_ascii(first)

# tokens to match by substring on the first line or full header
_FORCE_MULTI_TOKENS = {
    # Spanish / Portuguese
    "numero de grupo", "numero do grupo",
    "modulo", "leccion", "licao", "licao", "licao", "licao",  # include pt variants
    # English fallbacks
    "group number", "module", "lesson number", "lesson",
}

def _is_force_multiselect(col_name: str) -> bool:
    main = _main_label(col_name)
    full = _norm_ascii(col_name)
    return any(tok in main for tok in _FORCE_MULTI_TOKENS) or any(tok in full for tok in _FORCE_MULTI_TOKENS)

# ---- Display aliases for filter labels ----
# Пишем "сырые" названия как на листе, АЛГОРИТМ сам нормализует (первая строка, без акцентов, нижний регистр)
DISPLAY_ALIASES_RAW = {
    # Шапки колонок (первая строка)
    "Nombre": "Tutor's name",
    "Líder de Equipo": "TL",
    "Curso": "Course",
    "Tipo de grupo": "Group type",
    "Número de grupo": "Group number",
    "Duración de la clase": "Class duration",
    "Nombre del estudiante": "Student's name",
    "Enlace al perfil del estudiante en BO": "Student's BO",
    "Módulo": "Module",
    "Lección": "Lesson",
    "Motivo de la clase extra": "Reason for extra class",
    "Enlace de grabación de la clase": "Recording",
    "Tipo de lección extra": "Type of extra lesson",
    "Timestamp": "Date of the report",
    "Fecha de la clase": "Date of the class",
    "Email Address": "Tutor's personal email",
    "Email Corporativo": "Tutor's corporate email",
}

# Преобразуем ключи в нормализованные (как делает _main_label)
DISPLAY_ALIASES = { _main_label(k): v for k, v in DISPLAY_ALIASES_RAW.items() }

def _display_label(col_name: str, suffix: str = "") -> str:
    """Красивое имя для фильтра: алиас по первой строке заголовка + опциональный суффикс."""
    base = DISPLAY_ALIASES.get(_main_label(col_name), str(col_name).splitlines()[0].strip())
    return f"{base}{suffix}"

# ===================== Main =====================

with st.spinner("Loading data…"):
    df_raw = load_sheet_df(SHEET_ID, WS_NAME, rng="A:Q")

if df_raw.empty:
    st.warning(f"No data. Check access to the file and that the tab '{WS_NAME}' contains a header row in A:Q.""Also share the sheet with your service account e‑mail.")
    st.stop()

_df = df_raw.copy()
col_order = list(_df.columns)

colA_name = col_order[0] if len(col_order) >= 1 else None
colL_name = col_order[11] if len(col_order) >= 12 else None

_dtA = _to_datetime_series(_df[colA_name]) if colA_name else pd.Series([], dtype="datetime64[ns]")
_dtL = _to_datetime_series(_df[colL_name]) if colL_name else pd.Series([], dtype="datetime64[ns]")

st.sidebar.header("Filters")

# Date range for A
if colA_name is not None:
    a_min, a_max = _dtA.min(), _dtA.max()
    if pd.isna(a_min) or pd.isna(a_max):
        st.sidebar.caption(f"Column A ('{colA_name}') doesn't look like dates → skipping range filter.")
    else:
        a_def = (a_min.date(), a_max.date())
        a_range = st.sidebar.date_input(f"{_display_label(colA_name)}", value=a_def)
        if isinstance(a_range, tuple) and len(a_range) == 2:
            startA, endA = pd.to_datetime(a_range[0]), pd.to_datetime(a_range[1])
            endA = endA + pd.Timedelta(days=1)
            _df = _df[(_dtA >= startA) & (_dtA < endA) | _dtA.isna()]

# Date range for L
if colL_name is not None:
    l_min, l_max = _dtL.min(), _dtL.max()
    if pd.isna(l_min) or pd.isna(l_max):
        st.sidebar.caption(f"Column L ('{colL_name}') doesn't look like dates → skipping range filter.")
    else:
        l_def = (l_min.date(), l_max.date())
        l_range = st.sidebar.date_input(f"{_display_label(colL_name)}", value=l_def, key="date_L")
        if isinstance(l_range, tuple) and len(l_range) == 2:
            startL, endL = pd.to_datetime(l_range[0]), pd.to_datetime(l_range[1])
            endL = endL + pd.Timedelta(days=1)
            _df = _df[(_dtL >= startL) & (_dtL < endL) | _dtL.isna()]

st.sidebar.divider()

# Filters for ALL other columns
for idx, col in enumerate(col_order):
    if col is None or idx in (0, 11):
        continue

    col_series = _df[col]
    force_multi = _is_force_multiselect(col)

    if _is_numeric_col(col_series) and not force_multi:
        col_num = pd.to_numeric(col_series, errors="coerce")
        nmin = float(np.nanmin(col_num.values)) if np.isfinite(np.nanmin(col_num.values)) else 0.0
        nmax = float(np.nanmax(col_num.values)) if np.isfinite(np.nanmax(col_num.values)) else 0.0
        if not np.isfinite(nmin) or not np.isfinite(nmax) or nmin == nmax:
            opts = _unique_list_for_multiselect(col_series)
            label = _display_label(col)
            sel = st.sidebar.multiselect(label, opts, key=f"ms_{idx}")
            if sel:
                base = col_series.astype(str).str.strip()
                mask = base.isin([s for s in sel if s != "(blank)"]) | (col_series.isna() if "(blank)" in sel else False)
                _df = _df[mask]
        else:
            label = _display_label(col, " (range)")
            lo, hi = st.sidebar.slider(
                label,
                min_value=float(np.floor(nmin)),
                max_value=float(np.ceil(nmax)),
                value=(float(np.floor(nmin)), float(np.ceil(nmax))),
            )
            num = pd.to_numeric(_df[col], errors="coerce")
            _df = _df[(num >= lo) & (num <= hi) | num.isna()]
    else:
        opts = _unique_list_for_multiselect(col_series)
        label = _display_label(col)
        sel = st.sidebar.multiselect(label, opts, key=f"ms_{idx}")
        if sel:
            base = col_series.astype(str).str.strip()
            mask = base.isin([s for s in sel if s != "(blank)"]) | (col_series.isna() if "(blank)" in sel else False)
            _df = _df[mask]

# ===================== Output =====================

st.success(f"Rows: {len(_df)} (of {len(df_raw)})")

show_cols = [c for c in col_order if c in _df.columns]
view = _df.loc[:, show_cols].copy()

ROW, HEADER = 34, 44
height = min(900, HEADER + ROW * max(1, min(len(view), 200)))

st.dataframe(view, use_container_width=True, height=height)

st.download_button("⬇️ Download CSV", view.to_csv(index=False).encode("utf-8"), file_name="filtered.csv", mime="text/csv")

st.caption(textwrap.dedent(
    f"""
    **Notes**
    • Sheet ID: `{SHEET_ID}`  |  Tab: `{WS_NAME}`  |  Range: A:Q
    """
))

if st.button("Refresh"):
    load_sheet_df.clear()
    _unique_list_for_multiselect.clear()
    st.rerun()
