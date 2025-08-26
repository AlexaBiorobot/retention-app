import os
import io
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
    page_title="Form Responses Viewer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Simple password gate (как в исходном коде) ---
st.session_state.setdefault("is_authed", False)
if not st.session_state["is_authed"]:
    pwd = st.text_input("Password:", type="password")
    # можно задать свой пароль через переменную окружения/секреты
    target_pwd = os.getenv("APP_PASSWORD") or st.secrets.get("APP_PASSWORD") or "Kodland123"
    if pwd == target_pwd:
        st.session_state["is_authed"] = True
        st.rerun()
    else:
        if pwd:
            st.error("Wrong password")
        st.stop()

st.title("📋 Form Responses Viewer (A:Q)")
st.caption("Loads a Google Sheet tab and lets you filter by every column. Columns A and L have date-range filters.")

# ===================== Config =====================
DEFAULT_SHEET_ID = "1BtET9YSSLv1vSqWejO8tka4kdJygsnrlXYSEOavG4uA"  # from your link
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
    """Read header + rows from A:Q and return a DataFrame.
    Empty strings are converted to <NA>."""
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
    rows = [r + [None] * (len(header) - len(r)) for r in rows]  # pad
    df = pd.DataFrame(rows, columns=header)
    return df.replace({"": pd.NA})

# ===================== Helpers =====================

def _to_datetime_series(s: pd.Series) -> pd.Series:
    """Parse column to datetime with two passes (US/ISO then dayfirst)."""
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
    return uniq[:2000]  # hard cap for UI

# --- label normalizer & forced-multiselect list ---

def _norm_label(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.lower().strip().split())

FORCE_MULTI = {
    # Spanish
    "Número de grupo", "Numero de grupo",
    "Módulo", "Modulo",
    "Lección", "Leccion",
    # English fallbacks
    "Group number", "Module", "Lesson", "Lesson number",
}
FORCE_MULTI_NORM = { _norm_label(x) for x in FORCE_MULTI }

# --- export utils (как в исходнике) ---
def to_excel_bytes(data: pd.DataFrame) -> Optional[io.BytesIO]:
    try:
        import xlsxwriter  # noqa: F401
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
            data.to_excel(w, index=False)
        buf.seek(0)
        return buf
    except Exception:
        return None

# ===================== Main =====================

with st.spinner("Loading data…"):
    df_raw = load_sheet_df(SHEET_ID, WS_NAME, rng="A:Q")

if df_raw.empty:
    st.warning(
    f"No data. Check access to the file and that the tab '{WS_NAME}' contains a header row in A:Q.\n"
    "Also share the sheet with your service account e-mail."
)
    st.stop()

# Work on a copy
_df = df_raw.copy()

# Original order (A:Q)
col_order = list(_df.columns)

# A, L positional names
colA_name = col_order[0] if len(col_order) >= 1 else None
colL_name = col_order[11] if len(col_order) >= 12 else None

# Parse A & L to datetime
_dtA = _to_datetime_series(_df[colA_name]) if colA_name else pd.Series([], dtype="datetime64[ns]")
_dtL = _to_datetime_series(_df[colL_name]) if colL_name else pd.Series([], dtype="datetime64[ns]")

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")

# Date range for A
if colA_name is not None:
    a_min, a_max = _dtA.min(), _dtA.max()
    if pd.isna(a_min) or pd.isna(a_max):
        st.sidebar.caption(f"Column A ('{colA_name}') doesn't look like dates → skipping range filter.")
    else:
        a_def = (a_min.date(), a_max.date())
        a_range = st.sidebar.date_input(f"Date range — A: {colA_name}", value=a_def)
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
        l_range = st.sidebar.date_input(f"Date range — L: {colL_name}", value=l_def, key="date_L")
        if isinstance(l_range, tuple) and len(l_range) == 2:
            startL, endL = pd.to_datetime(l_range[0]), pd.to_datetime(l_range[1])
            endL = endL + pd.Timedelta(days=1)
            _df = _df[(_dtL >= startL) & (_dtL < endL) | _dtL.isna()]

st.sidebar.divider()

# Filters for ALL other columns
for idx, col in enumerate(col_order):
    if col is None:
        continue
    if idx in (0, 11):  # skip A & L here (already filtered by date)
        continue

    col_series = _df[col]
    force_multi = _norm_label(col) in FORCE_MULTI_NORM

    if _is_numeric_col(col_series) and not force_multi:
        col_num = pd.to_numeric(col_series, errors="coerce")
        nmin = float(np.nanmin(col_num.values)) if np.isfinite(np.nanmin(col_num.values)) else 0.0
        nmax = float(np.nanmax(col_num.values)) if np.isfinite(np.nanmax(col_num.values)) else 0.0
        if not np.isfinite(nmin) or not np.isfinite(nmax) or nmin == nmax:
            opts = _unique_list_for_multiselect(col_series)
            sel = st.sidebar.multiselect(f"{col}", opts)
            if sel:
                base = col_series.astype(str).str.strip()
                mask = base.isin([s for s in sel if s != "(blank)"]) | (col_series.isna() if "(blank)" in sel else False)
                _df = _df[mask]
        else:
            lo, hi = st.sidebar.slider(
                f"{col} (range)",
                min_value=float(np.floor(nmin)),
                max_value=float(np.ceil(nmax)),
                value=(float(np.floor(nmin)), float(np.ceil(nmax))),
            )
            num = pd.to_numeric(_df[col], errors="coerce")
            _df = _df[(num >= lo) & (num <= hi) | num.isna()]
    else:
        opts = _unique_list_for_multiselect(col_series)
        sel = st.sidebar.multiselect(f"{col}", opts)
        if sel:
            base = col_series.astype(str).str.strip()
            mask = base.isin([s for s in sel if s != "(blank)"]) | (col_series.isna() if "(blank)" in sel else False)
            _df = _df[mask]

# ===================== Output =====================

st.success(f"Rows: {len(_df)} (of {len(df_raw)})")

# Preserve original column order and show only A:Q
show_cols = [c for c in col_order if c in _df.columns]
view = _df.loc[:, show_cols].copy()

# Table size
ROW, HEADER = 34, 44
height = min(900, HEADER + ROW * max(1, min(len(view), 200)))

st.dataframe(view, use_container_width=True, height=height)

# Downloads
c1, c2 = st.columns(2)
with c1:
    st.download_button("⬇️ Download CSV", view.to_csv(index=False).encode("utf-8"), file_name="filtered.csv", mime="text/csv")
with c2:
    buf = to_excel_bytes(view)
    if buf is not None:
        st.download_button("⬇️ Download Excel", buf, file_name="filtered.xlsx")
    else:
        st.caption("Excel export is unavailable; install xlsxwriter.")

st.caption(textwrap.dedent(
    f"""
    **Notes**
    • Sheet ID: `{SHEET_ID}`  |  Tab: `{WS_NAME}`  |  Range: A:Q
    • Share the Google Sheet with your service account e‑mail (Viewer is enough).
    • Configure secrets (Streamlit Cloud / local `.streamlit/secrets.toml`):

      ```toml
      GSHEET_ID = "{DEFAULT_SHEET_ID}"
      GSHEET_WS = "{DEFAULT_WS_NAME}"
      GCP_SERVICE_ACCOUNT = "{{ your service account JSON here as a single line }}"
      APP_PASSWORD = "Kodland123"  # or your own
      ```
    """
))

# --- Refresh like in your original app ---
if st.button("Refresh"):
    load_sheet_df.clear()
    _unique_list_for_multiselect.clear()
    st.rerun()
