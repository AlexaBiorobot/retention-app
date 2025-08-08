import os
import json
import time

import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe
from gspread.exceptions import APIError, WorksheetNotFound

# —————————————————————————————
# Вспомогательные функции для retry при 503
# —————————————————————————————

def api_retry_open(client, key, max_attempts=5, backoff=1.0):
    for attempt in range(1, max_attempts + 1):
        try:
            return client.open_by_key(key)
        except APIError as e:
            code = getattr(e.response, "status_code", None) or getattr(e.response, "status", None)
            if code == 503 and attempt < max_attempts:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise

def api_retry_worksheet(spreadsheet, name, max_attempts=5, backoff=1.0):
    for attempt in range(1, max_attempts + 1):
        try:
            return spreadsheet.worksheet(name)
        except APIError as e:
            code = getattr(e.response, "status_code", None) or getattr(e.response, "status", None)
            if code == 503 and attempt < max_attempts:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise

# === авторизация ===
sa_info = json.loads(os.environ["GCP_SERVICE_ACCOUNT"])
creds   = ServiceAccountCredentials.from_json_keyfile_dict(
    sa_info,
    ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
)
client  = gspread.authorize(creds)

# === получить DF ===
from tutor_tool_web import load_data_from_gsheet
df = load_data_from_gsheet()   # ваш основной загрузчик с override-ами

# === открыть гугл-таблицу и «залить» в лист Exported ===
SS_ID      = "1wJvMIf62izX10-r_-B1QtfKWRzobZHCH8dufVsCCVko"
SHEET_NAME = "data2"

# вместо client.open_by_key делаем с retry
sh = api_retry_open(client, SS_ID)

# вместо sh.worksheet тоже с retry, но обрабатывая отсутствие листа
try:
    ws = api_retry_worksheet(sh, SHEET_NAME)
    ws.clear()
except WorksheetNotFound:
    ws = sh.add_worksheet(SHEET_NAME, rows=1000, cols=50)

# запись с помощью set_with_dataframe
set_with_dataframe(ws, df)

# === второй экспорт ===
SS_ID_2      = "1QbdVTacl2UdSYI5PSHuFOxEXYdRXjh0hc5Q9I5zP0fU"
SHEET_NAME_2 = "data2"

sh2 = api_retry_open(client, SS_ID_2)

try:
    ws2 = api_retry_worksheet(sh2, SHEET_NAME_2)
    ws2.clear()
except WorksheetNotFound:
    ws2 = sh2.add_worksheet(SHEET_NAME_2, rows=1000, cols=50)

set_with_dataframe(ws2, df)
