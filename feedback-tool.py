import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import altair as alt
import json

SPREADSHEET_ID = "1fR8_Ay7jpzmPCAl6dWSCC7sWw5VJOaNpu5Zp8b78LRg"
SHEET_NAME = "Form Responses 1"

# ---- Авторизация через st.secrets ----
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]

# В secrets должен быть ключ GCP_SERVICE_ACCOUNT = '''{...json...}'''
sa_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT"])
creds = ServiceAccountCredentials.from_json_keyfile_dict(sa_info, scope)
client = gspread.authorize(creds)

# ---- Загрузка данных ----
ws = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
data = ws.get_all_records()  # использует заголовки первой строки
df = pd.DataFrame(data)

# ---- Приведение типов (подстрой под реальные имена столбцов) ----
# A — дата фидбека, N — курс, S — неделя, G — оценка
df["A"] = pd.to_datetime(df["A"], errors="coerce")
df["S"] = pd.to_numeric(df["S"], errors="coerce")
df["G"] = pd.to_numeric(df["G"], errors="coerce")

# ---- Фильтры ----
st.sidebar.header("Фильтры")
courses = sorted([c for c in df["N"].dropna().unique()])
selected_courses = st.sidebar.multiselect("Курс (N)", courses, default=courses)

min_date, max_date = df["A"].min(), df["A"].max()
date_range = st.sidebar.date_input("Дата фидбека (A)", [min_date.date(), max_date.date()])

df_f = df[df["N"].isin(selected_courses)]
df_f = df_f[(df_f["A"] >= pd.to_datetime(date_range[0])) &
            (df_f["A"] <= pd.to_datetime(date_range[1]))]

# ---- Группировка: avg(G) по S ----
agg = df_f.groupby("S", as_index=False)["G"].mean().rename(columns={"G": "avg_G"}).sort_values("S")

# ---- График ----
st.title("Av score")
chart = (
    alt.Chart(agg)
      .mark_line(point=True)
      .encode(
          x=alt.X("S:Q", title="S"),
          y=alt.Y("avg_G:Q", title="Average G"),
          tooltip=["S", "avg_G"]
      )
)
st.altair_chart(chart, use_container_width=True)
