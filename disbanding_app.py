import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# === ЯВНО ЗАДАЁМ SS И TAB ===
SHEET_ID = "1Jbb4p1cZCo67ZRiW5cmFUq-c9ijo5VMH_hFFMVYeJk4"
WS_NAME  = "data"

ST_SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

@st.cache_data(show_spinner=False, ttl=300)
def load_sheet_df(sheet_id: str, worksheet_name: str = "data") -> pd.DataFrame:
    sa_info = dict(st.secrets["gcp_service_account"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(sa_info, ST_SCOPE)
    client = gspread.authorize(creds)

    sh = client.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet_name)

    values = ws.get("A:Q")  # первая строка — заголовки
    if not values:
        return pd.DataFrame()

    header = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=header).replace({"": pd.NA})
    return df

def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # A:Q -> 17 колонок; D=индекс 3, K=индекс 10 (0-based)
    colD = df.columns[3]
    colK = df.columns[10]

    d_active = df[colD].astype(str).str.strip().str.lower() == "active"
    k_num = pd.to_numeric(df[colK], errors="coerce")
    k_ok = k_num.notna() & (k_num < 32)

    out = df.loc[d_active & k_ok].copy()
    out[colK] = k_num.loc[out.index]
    return out

def main():
    st.set_page_config(page_title="Groups & Tutors loader", layout="wide")
    st.title("Initial export from Google Sheets (A:Q, D='active', K < 32)")

    # --- используем константы, но можно переопределить через Secrets/сайдбар ---
    default_sheet_id = st.secrets.get("GSHEET_ID", SHEET_ID)
    default_ws_name  = st.secrets.get("GSHEET_WS", WS_NAME)

    sheet_id = st.sidebar.text_input("Google Sheet ID", value=default_sheet_id)
    ws_name = default_ws_name

    # Показать список вкладок и ссылки
    try:
        sa_info = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(sa_info, ST_SCOPE)
        client = gspread.authorize(creds)
        sh = client.open_by_key(sheet_id)

        ws_names = [ws.title for ws in sh.worksheets()]
        ws_name = st.sidebar.selectbox(
            "Worksheet",
            ws_names,
            index=ws_names.index(default_ws_name) if default_ws_name in ws_names else 0
        )
        # ссылки
        selected_ws = sh.worksheet(ws_name)
        gid = getattr(selected_ws, "id", None) or selected_ws._properties.get("sheetId")
        sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"
        tab_url = f"{sheet_url}#gid={gid}"
        st.sidebar.markdown(f"[Открыть Google Sheet]({sheet_url})")
        st.sidebar.markdown(f"[Открыть вкладку **{ws_name}**]({tab_url})")
        st.markdown(f"🔗 **Sheet:** [{sheet_id}]({sheet_url}) • **Tab:** [{ws_name}]({tab_url})")
    except Exception as e:
        st.sidebar.error(f"Не удалось получить вкладки/ссылки: {e}")

    if not sheet_id:
        st.error("Не указан Sheet ID.")
        st.stop()

    with st.spinner("Loading data from Google Sheets…"):
        df = load_sheet_df(sheet_id, ws_name)

    if df.empty:
        st.warning(f"Пусто: проверь вкладку '{ws_name}' и доступ сервисного аккаунта.")
        return

    filtered = filter_df(df)

    st.caption(f"Всего строк в листе: {len(df)}")
    st.success(f"Отфильтровано строк: {len(filtered)}")
    st.dataframe(filtered, use_container_width=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Скачать CSV", data=csv, file_name="filtered_export.csv", mime="text/csv")

    if st.button("Обновить данные"):
        load_sheet_df.clear()
        st.rerun()

if __name__ == "__main__":
    main()
