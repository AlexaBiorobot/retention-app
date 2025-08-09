import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

ST_SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

@st.cache_data(show_spinner=False, ttl=300)
def load_sheet_df(sheet_id: str, worksheet_name: str = "data") -> pd.DataFrame:
    # Авторизация по service account из secrets
    sa_info = dict(st.secrets["gcp_service_account"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(sa_info, ST_SCOPE)
    client = gspread.authorize(creds)

    # Открываем шит/лист
    sh = client.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet_name)

    # Берём значения только A:Q (17 колонок)
    values = ws.get("A:Q")  # первая строка — заголовки
    if not values:
        return pd.DataFrame()

    header = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=header)

    # Нормализация
    df = df.replace({"": pd.NA})
    return df


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Безопасно адресуемся по индексам столбцов (A:Q → 17 колонок):
    # D = индекс 3; K = индекс 10 (0-based)
    colD = df.columns[3]
    colK = df.columns[10]

    # D должно быть "active" (регистронезависимо)
    d_active = df[colD].astype(str).str.strip().str.lower() == "active"

    # K не пустое и < 32 (конвертируем в число)
    k_num = pd.to_numeric(df[colK], errors="coerce")
    k_ok = k_num.notna() & (k_num < 32)

    out = df.loc[d_active & k_ok].copy()
    # Если полезно — можно подставить числовой K обратно
    out[colK] = k_num.loc[out.index]
    return out


def main():
    st.set_page_config(page_title="Groups & Tutors loader", layout="wide")
    st.title("Initial export from Google Sheets (A:Q, D='active', K < 32)")

    sheet_id = st.secrets["GSHEET_ID"]
    with st.spinner("Loading data from Google Sheets…"):
        df = load_sheet_df(sheet_id, "data")

    if df.empty:
        st.warning("Пусто: проверь, что лист 'data' существует и сервисный аккаунт имеет доступ.")
        return

    filtered = filter_df(df)

    st.caption(f"Всего строк в листе: {len(df)}")
    st.success(f"Отфильтровано строк: {len(filtered)}")

    st.dataframe(filtered, use_container_width=True)

    # Кнопка на скачивание
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Скачать CSV", data=csv, file_name="filtered_export.csv", mime="text/csv")

    # Кнопка обновить (сброс кеша)
    if st.button("Обновить данные"):
        load_sheet_df.clear()  # сброс cache_data
        st.rerun()


if __name__ == "__main__":
    main()
