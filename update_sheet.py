name: Update Google Sheet

on:
  workflow_dispatch:
  schedule:
    - cron: "0 0 * * *"  # например, раз в сутки в полночь

jobs:
  push-to-sheet:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install deps
        run: |
          pip install \
            gspread \
            oauth2client \
            gspread-dataframe \
            pandas

      - name: Run update script
        env:
          GCP_SERVICE_ACCOUNT: ${{ secrets.GCP_SERVICE_ACCOUNT }}
          MAIN_SS_ID:          ${{ secrets.MAIN_SS_ID }}
          MAIN_SHEET:          ${{ secrets.MAIN_SHEET }}
          LEADS_SS_ID:         ${{ secrets.LEADS_SS_ID }}
          LEADS_SHEET:         ${{ secrets.LEADS_SHEET }}
        run: python update_sheet.py
