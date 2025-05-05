import os, json
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe

# загрузить DataFrame из кода — либо дублировать функцию load_data_from_gsheet(),
# либо импортировать её из вашего модуля:
from tutor_tool_web import load_data_from_gsheet

# === авторизация ===
sa_info = json.loads(os.environ["GCP_SERVICE_ACCOUNT"])
creds   = ServiceAccountCredentials.from_json_keyfile_dict(
    sa_info,
    ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
)
client  = gspread.authorize(creds)

# === получить DF ===
df = load_data_from_gsheet()   # здесь — уже со всеми override-ами и чисткой

# === открыть гугл-таблицу и «залить» в лист Exported ===
SS_ID     = "https://docs.google.com/spreadsheets/d/1wJvMIf62izX10-r_-B1QtfKWRzobZHCH8dufVsCCVko/edit?gid=1572100264#gid=1572100264"
SHEET_NAME = "data2"
sh = client.open_by_key(SS_ID)
try:
    ws = sh.worksheet(SHEET_NAME)
    ws.clear()
except gspread.exceptions.WorksheetNotFound:
    ws = sh.add_worksheet(SHEET_NAME, rows=1000, cols=50)

set_with_dataframe(ws, df)
