import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import altair as alt

# ===================== Настройки =====================
SPREADSHEET_ID = "1fR8_Ay7jpzmPCAl6dWSCC7sWw5VJOaNpu5Zp8b78LRg"
SHEET_NAME = "Form Responses 1"

# ===================== Авторизация Google Sheets =====================
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)

# ===================== Загрузка данных =====================
sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
data = sheet.get_all_records()
df = pd.DataFrame(data)

# ===================== Приведение типов =====================
df["A"] = pd.to_datetime(df["A"], errors="coerce")   # Дата фидбека
df["S"] = pd.to_numeric(df["S"], errors="coerce")
df["G"] = pd.to_numeric(df["G"], errors="coerce")

# ===================== Фильтры =====================
st.sidebar.header("Фильтры")

# Фильтр по курсу (N)
courses = df["N"].dropna().unique()
selected_courses = st.sidebar.multiselect("Выбери курс", courses, default=courses)

# Фильтр по дате (A)
min_date, max_date = df["A"].min(), df["A"].max()
date_range = st.sidebar.date_input("Диапазон дат", [min_date, max_date])

# Применение фильтров
df_filtered = df[df["N"].isin(selected_courses)]
df_filtered = df_filtered[(df_filtered["A"] >= pd.to_datetime(date_range[0])) &
                          (df_filtered["A"] <= pd.to_datetime(date_range[1]))]

# ===================== Группировка =====================
df_grouped = df_filtered.groupby("S", as_index=False)["G"].mean()

# ===================== График =====================
st.title("40 week courses")

chart = (
    alt.Chart(df_grouped)
    .mark_line(point=True)
    .encode(
        x=alt.X("S:O", title="S"),
        y=alt.Y("G:Q", title="Average G"),
        tooltip=["S", "G"]
    )
)

st.altair_chart(chart, use_container_width=True)
