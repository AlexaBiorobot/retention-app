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

    # A:Q (17 колонок). Первая строка — заголовки.
    values = ws.get("A:Q")
    if not values:
        return pd.DataFrame()

    header = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=header)

    # Нормализация пустых
    df = df.replace({"": pd.NA})
    return df

def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # A:Q -> 17 колонок; кол-во столбцов может отличаться, но D и K должны существовать
    if len(df.columns) < 11:
        st.error("Ожидалось минимум 11 колонок (до K). Проверь диапазон A:Q и заголовки.")
        st.stop()

    colD = df.columns[3]   # D
    colK = df.columns[10]  # K

    # D == "active" (case-insensitive)
    d_active = df[colD].astype(str).str.strip().str.lower() == "active"

    # K не пустое и < 32
    k_num = pd.to_numeric(df[colK], errors="coerce")
    k_ok = k_num.notna() & (k_num < 32)

    out = df.loc[d_active & k_ok].copy()
    out[colK] = k_num.loc[out.index]  # вернуть числовое K
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

def main():
    st.title("Initial export from Google Sheets (A:Q, D='active', K < 32)")

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
            # селект, если таб существует
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

    # --- Фильтр по условиям задачи ---
    filtered = filter_df(df)

    # --- Верхняя панель метрик ---
    c1, c2 = st.columns(2)
    c1.caption(f"Rows total: {len(df)}")
    c2.success(f"Filtered rows: {len(filtered)}")

    # --- Таблица ---
    st.dataframe(filtered, use_container_width=True)

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
        st.rerun()

if __name__ == "__main__":
    main()
