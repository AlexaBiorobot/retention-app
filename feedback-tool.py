import streamlit as st
import pandas as pd
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import altair as alt
import json
import string
import re
import unicodedata
from functools import lru_cache
from collections import Counter

# online-переводчик (если доступен)
try:
    from deep_translator import GoogleTranslator
    _gt = GoogleTranslator(source="auto", target="en")
except Exception:
    _gt = None  # нет библиотеки/интернета — используем запасной вариант

st.set_page_config(layout="wide", page_title="Feedback LatAm")

SPREADSHEET_ID = "1fR8_Ay7jpzmPCAl6dWSCC7sWw5VJOaNpu5Zp8b78LRg"

# === Refunds (LatAm) constants ===
REFUNDS_SHEET_ID = "1ITOBSlVk4trLSKAkc5vobQrdTz6ve1Z5ljf1CnQfDJo"
REFUNDS_TAB_NAME = "Refunds - LatAm"

# ---- Авторизация через st.secrets (строка JSON) ----
@st.cache_resource
def get_gs_client():
    scope = ["https://www.googleapis.com/auth/spreadsheets.readonly",
             "https://www.googleapis.com/auth/drive.readonly"]
    sa_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(sa_info, scope)
    return gspread.authorize(creds)

client = get_gs_client()

# ==================== УТИЛИТЫ ====================

def _excel_cols(n_cols: int) -> list[str]:
    """A, B, ..., Z, AA, AB, ..., ZZ (хватает для листов до 702 колонок)."""
    letters = list(string.ascii_uppercase)
    cols = []
    for a in [""] + letters:          # "" -> A..Z, затем AA..AZ, BA..BZ, ...
        for b in letters:
            name = (a + b) if a else b
            cols.append(name)
            if len(cols) >= n_cols:
                return cols
    return cols[:n_cols]

@st.cache_data(show_spinner=False)
def load_sheet_as_letter_df_cached(sheet_name: str) -> pd.DataFrame:
    ws = client.open_by_key(SPREADSHEET_ID).worksheet(sheet_name)
    values = ws.get('A:R')  # читаем только до R (хватает FR1/FR2)
    if not values or len(values) < 2:
        return pd.DataFrame()
    num_cols = len(values[0])
    letters = _excel_cols(num_cols)
    return pd.DataFrame(values[1:], columns=letters)

@st.cache_data(show_spinner=False)
def load_refunds_letter_df_cached() -> pd.DataFrame:
    try:
        ws = client.open_by_key(REFUNDS_SHEET_ID).worksheet(REFUNDS_TAB_NAME)
        vals = ws.get('A:AV')  # до AV включительно (нужны L/AV/AS/AU)
    except Exception:
        return pd.DataFrame()
    if not vals or len(vals) < 2:
        return pd.DataFrame()
    cols = _excel_cols(len(vals[0]))
    return pd.DataFrame(vals[1:], columns=cols)

@st.cache_data(show_spinner=False)
def load_qa_letter_df_cached() -> pd.DataFrame:
    try:
        ws = client.open_by_key(REFUNDS_SHEET_ID).worksheet("QA for analytics")
        vals = ws.get('A:Z')  # QA используют B/D/F/H/I → Z достаточно
    except Exception:
        return pd.DataFrame()
    if not vals or len(vals) < 2:
        return pd.DataFrame()
    cols = _excel_cols(len(vals[0]))
    return pd.DataFrame(vals[1:], columns=cols)

def safe_minmax(dt1_min, dt1_max, dt2_min, dt2_max):
    mins = [d for d in [dt1_min, dt2_min] if pd.notna(d)]
    maxs = [d for d in [dt1_max, dt2_max] if pd.notna(d)]
    return (min(mins) if mins else pd.NaT, max(maxs) if maxs else pd.NaT)

def filter_df(df: pd.DataFrame, course_col: str, date_col: str,
              selected_courses, date_range):
    if df.empty:
        return df
    dff = df.copy()
    if selected_courses:
        dff = dff[dff[course_col].isin(selected_courses)]
    else:
        return dff.iloc[0:0]
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1])
        dff = dff[(dff[date_col] >= start_dt) & (dff[date_col] <= end_dt)]
    elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
        only_dt = pd.to_datetime(date_range[0])
        dff = dff[dff[date_col].dt.date == only_dt.date()]
    return dff

def apply_filters_and_aggregate(df: pd.DataFrame, course_col: str, date_col: str,
                                x_col: str, y_col: str,
                                selected_courses=None, date_range=None):
    if df.empty or not {course_col, date_col, x_col, y_col}.issubset(df.columns):
        return pd.DataFrame(columns=[x_col, "avg_y", "count"])
    dff = filter_df(df, course_col, date_col, selected_courses, date_range)
    if dff.empty or not {x_col, y_col}.issubset(dff.columns):
        return pd.DataFrame(columns=[x_col, "avg_y", "count"])
    grp = dff.dropna(subset=[x_col, y_col])
    if grp.empty:
        return pd.DataFrame(columns=[x_col, "avg_y", "count"])
    agg = (grp.groupby(x_col, as_index=False)
             .agg(avg_y=(y_col, "mean"),
                  count=(y_col, "size"))
             .sort_values(x_col))
    return agg
                                    
def add_bucket(dff: pd.DataFrame, date_col: str, granularity: str) -> pd.DataFrame:
    out = dff.copy()
    if out.empty:
        return out
    if granularity == "День":
        out["bucket"] = out[date_col].dt.floor("D")
    elif granularity == "Неделя":
        out["bucket"] = out[date_col].dt.to_period("W-MON").dt.start_time
    elif granularity == "Месяц":
        out["bucket"] = out[date_col].dt.to_period("M").dt.to_timestamp()
    elif granularity == "Год":
        out["bucket"] = out[date_col].dt.to_period("Y").dt.to_timestamp()
    else:
        out["bucket"] = out[date_col].dt.floor("D")
    return out

def ensure_bucket_and_label(dff: pd.DataFrame, date_col: str, granularity: str) -> pd.DataFrame:
    if dff.empty:
        return dff.copy()
    out = dff.copy()
    if "bucket" not in out.columns or not pd.api.types.is_datetime64_any_dtype(out["bucket"]):
        out = add_bucket(out, date_col, granularity)

    if granularity == "День":
        fmt = "%Y-%m-%d"
    elif granularity == "Неделя":
        fmt = "W%W (%Y-%m-%d)"
    elif granularity == "Месяц":
        fmt = "%Y-%m"
    elif granularity == "Год":
        fmt = "%Y"
    else:
        fmt = "%Y-%m-%d"

    out["bucket_label"] = out["bucket"].dt.strftime(fmt)
    return out

@st.cache_data(show_spinner=False)
def prep_distribution(df_f: pd.DataFrame, value_col: str, allowed_values: list, label_title: str):
    if df_f.empty:
        return pd.DataFrame(), [], [], label_title
    d = df_f[df_f[value_col].isin(allowed_values)].copy()
    if d.empty:
        return pd.DataFrame(), [], [], label_title

    d["val"] = d[value_col].astype(int)
    d["val_str"] = d["val"].astype(str)

    grp = (d.groupby(["bucket", "bucket_label", "val", "val_str"], as_index=False)
             .size()
             .rename(columns={"size": "count"}))

    totals = (grp.groupby(["bucket", "bucket_label"], as_index=False)["count"]
                .sum()
                .rename(columns={"count": "total"}))

    out = grp.merge(totals, on=["bucket", "bucket_label"], how="left")
    out["pct"] = out["count"] / out["total"]

    bucket_order = (out[["bucket", "bucket_label"]]
                    .drop_duplicates()
                    .sort_values("bucket")["bucket_label"].tolist())

    val_order = [str(v) for v in allowed_values]
    return out, bucket_order, val_order, label_title

def prep_distribution_text_fr2(df_f: pd.DataFrame, text_col: str, granularity: str, title: str):
    """
    Готовит распределение по периодам для текстовой колонки (FR2),
    переводит значения на EN, считает count и % внутри периода.

    Возвращает: (out_df, bucket_order, cat_order, title)
    где out_df имеет колонки: bucket, bucket_label, cat_en, count, total, pct
    """
    if df_f.empty or text_col not in df_f.columns:
        return pd.DataFrame(), [], [], title

    d = df_f.copy()
    d = d.dropna(subset=["A", text_col])

    # приводим к периодам
    d = add_bucket(d, "A", granularity)
    d = ensure_bucket_and_label(d, "A", granularity)

    rows = []
    splitter = re.compile(r"[;,/\n|]+")
    for _, r in d.iterrows():
        raw = str(r[text_col]).strip()
        if not raw:
            continue
        parts = splitter.split(raw) if splitter.search(raw) else [raw]
        for p in parts:
            val = p.strip()
            if not val:
                continue
            en = translate_es_to_en(val)
            rows.append((r["bucket"], r["bucket_label"], en))

    if not rows:
        return pd.DataFrame(), [], [], title

    out = (pd.DataFrame(rows, columns=["bucket","bucket_label","cat_en"])
             .groupby(["bucket","bucket_label","cat_en"], as_index=False)
             .size()
             .rename(columns={"size":"count"}))

    totals = (out.groupby(["bucket","bucket_label"], as_index=False)["count"]
                .sum().rename(columns={"count":"total"}))
    out = out.merge(totals, on=["bucket","bucket_label"], how="left")
    out["pct"] = out["count"] / out["total"]

    bucket_order = (out[["bucket","bucket_label"]]
                    .drop_duplicates()
                    .sort_values("bucket")["bucket_label"].tolist())

    # категории сортируем по сумме за всё время (по убыванию)
    cat_order = (out.groupby("cat_en", as_index=False)["count"]
                   .sum()
                   .sort_values("count", ascending=False)["cat_en"]
                   .tolist())

    return out, bucket_order, cat_order, title

@st.cache_data(show_spinner=False)
def _to_percentile_0_100(df: pd.DataFrame, val_col: str) -> pd.Series:
    """
    Возвращает Series с перцентилем (0–100) для каждого значения val_col в df,
    считанным внутри df (учтены дубликаты; метод 'average').
    """
    if df.empty or val_col not in df.columns:
        return pd.Series([], dtype=float)
    return df[val_col].rank(pct=True, method="average") * 100.0

@st.cache_data(show_spinner=False)
def _pack_full_tooltip(df_src: pd.DataFrame, x_col: str, legend_title: str):
    """
    Готовит DF для «общего» тултипа по целому столбику.
    Требуемые колонки на входе: x_col, 'val', 'count'. 'total' можно не передавать.
    Возвращает: (packed_df, tip_cols) где tip_cols — список строковых столбцов "1","2",...
    """
    if df_src.empty or not {x_col, "val", "count"}.issubset(df_src.columns):
        return pd.DataFrame(columns=[x_col, "total"]), []

    d = df_src[[x_col, "val", "count"]].copy()
    d["val"] = pd.to_numeric(d["val"], errors="coerce")
    d["count"] = pd.to_numeric(d["count"], errors="coerce").fillna(0).astype(int)
    d = d.dropna(subset=["val"])
    if d.empty:
        return pd.DataFrame(columns=[x_col, "total"]), []

    d["val"] = d["val"].astype(int)

    # wide: индексы — период (x_col); колонки — числовые значения шкалы (1..)
    g = d.groupby([x_col, "val"], as_index=False)["count"].sum()
    wide = (
        g.pivot(index=x_col, columns="val", values="count")
         .fillna(0).astype(int)
         .reset_index()
    )

    # список значений шкалы по возрастанию
    val_cols = sorted([c for c in wide.columns if c != x_col])

    # total по периоду
    wide["total"] = wide[val_cols].sum(axis=1).astype(int)

    # в тултипе хотим строки: "1 — 12 (18%)", "2 — 3 (5%)", ...
    tip_cols = []
    for v in val_cols:
        out_col = str(v)   # имя столбца для тултипа
        tip_cols.append(out_col)

        def _fmt_row(r):
            c = int(r.get(v, 0))
            t = int(r.get("total", 0))
            pct = (c / t) if t > 0 else 0.0
            return f"{v} — {c} ({pct:.0%})"

        wide[out_col] = wide.apply(_fmt_row, axis=1)

    return wide[[x_col, "total"] + tip_cols], tip_cols

# ===== Аспекты и перевод =====
ASPECTS_ES_EN = [
    ("La materia que se enseñó", "The subject that was taught"),
    ("La explicación del profesor", "The teacher's explanation"),
    ("Actividades realizadas en la sala", "Activities done in the classroom"),
    ("Tareas para hacer en casa", "Homework to do at home"),
    ("La forma en que se comportó la clase", "How the class behaved"),
    ("Cuando me aclararon las dudas", "When my questions were clarified"),
    ("Cuando me trataron bien y con atención", "When I was treated well and attentively"),
]

def _norm(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = unicodedata.normalize("NFKD", s).casefold().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

_ASPECTS_NORM = [(_norm(es), es, en) for es, en in ASPECTS_ES_EN]

_BASIC_ES_EN = {
    "si": "yes", "sí": "yes", "no": "no", "ok": "ok",
    "materia":"subject","explicacion":"explanation","explicación":"explanation",
    "profesor":"teacher","actividades":"activities","sala":"classroom",
    "tareas":"homework","casa":"home","forma":"way","comporto":"behaved",
    "comportó":"behaved","clase":"class","aclararon":"clarified","dudas":"doubts",
    "trataron":"treated","bien":"well","atencion":"attention","atención":"attention"
}

def _naive_translate_es_en(text: str) -> str:
    tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", text)
    mapped = []
    for t in tokens:
        k = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii").lower()
        mapped.append(_BASIC_ES_EN.get(k, t))
    out = " ".join(mapped).strip()
    return out[0].upper() + out[1:] if out else ""

@lru_cache(maxsize=5000)
def translate_es_to_en(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if _gt is not None:
        try:
            return _gt.translate(t)
        except Exception:
            pass
    return _naive_translate_es_en(t)

# ===== Dislike-аспекты (из F), со стабильным EN-переводом =====
DISLIKE_ES_EN = [
    ("Explica mejor el contenido.", "Explain the content better."),
    ("Tener menos retrasos o ausencias.", "Have fewer delays or absences."),
    ("Hacer más preguntas.", "Ask more questions."),
    ("Prestar más atención a los estudiantes.", "Pay more attention to students."),
    ("Mejorar la disciplina de la clase.", "Improve class discipline."),
    ("Responde más en WhatsApp.", "Reply more on WhatsApp."),
]

def _norm_local(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = unicodedata.normalize("NFKD", s).casefold().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

# универсальная сборка для произвольного списка аспектов [(es, en), ...]
@st.cache_data(show_spinner=False)
def build_aspects_counts_generic(df: pd.DataFrame, text_col: str, date_col: str,
                                 granularity: str, aspects_es_en: list[tuple[str, str]]):
    need_cols = [date_col, text_col]
    if df.empty or not all(c in df.columns for c in need_cols):
        return (pd.DataFrame(columns=["bucket","bucket_label","aspect","aspect_en","count"]),
                pd.DataFrame(columns=["en","mention","total"]))

    d = df[need_cols].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    if d.empty:
        return (pd.DataFrame(columns=["bucket","bucket_label","aspect","aspect_en","count"]),
                pd.DataFrame(columns=["en","mention","total"]))

    aspects_norm = [(_norm_local(es), es, en) for es, en in aspects_es_en]

    d = d.rename(columns={date_col: "A", text_col: "TXT"})
    d = add_bucket(d, "A", granularity)
    d = ensure_bucket_and_label(d, "A", granularity)

    rows, _unknown = [], []
    for _, r in d.iterrows():
        txt = str(r["TXT"]).strip()
        if not txt:
            continue
        parts = re.split(r"[;,/\n|]+", txt) if re.search(r"[;,/\n|]", txt) else [txt]
        for p in parts:
            t = _norm_local(p.strip())
            if not t:
                continue
            for es_norm, es, en in aspects_norm:
                if t == es_norm or es_norm in t:
                    rows.append((r["bucket"], r["bucket_label"], f"{es} (EN: {en})", en, 1))
                    break

    counts = pd.DataFrame(rows, columns=["bucket","bucket_label","aspect","aspect_en","count"])
    if not counts.empty:
        counts = counts.groupby(["bucket","bucket_label","aspect","aspect_en"], as_index=False)["count"].sum()

    return counts, pd.DataFrame(columns=["en","mention","total"])

# ===== FR2: шаблонные фразы для D и E =====
FR2_D_TEMPL_ES_EN = [
    ("Sí, todo a tiempo", "Yes, everything on time"),
    ("Empezó tarde la clase", "The class started late"),
    ("El profesor llegó muy tarde o no vino en absoluto", "The teacher arrived very late or didn't come at all"),
]

FR2_E_TEMPL_ES_EN = [
    ("Sí", "Yes"),
    ("Terminó demasiado pronto", "Finished too soon"),
    ("Fue demasiado larga", "It was too long"),
    ("No hubo clase", "There was no class"),
]

@st.cache_data(show_spinner=False)
def build_template_counts(
    df: pd.DataFrame,
    text_col: str,
    date_col: str,
    granularity: str,
    templates_es_en: list[tuple[str, str]],
):
    """
    FR2:D/E. Быстро: split+explode, нормализация, матч по eq/contains.
    ВАЖНО: запятую НЕ используем как разделитель (чтобы не ломать 'Sí, todo a tiempo').
    """
    need_cols = [date_col, text_col]
    if df.empty or not all(c in df.columns for c in need_cols):
        return pd.DataFrame(columns=["bucket","bucket_label","templ_es","templ_en","count"])

    d = df[need_cols].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    if d.empty:
        return pd.DataFrame(columns=["bucket","bucket_label","templ_es","templ_en","count"])

    d = d.rename(columns={date_col: "A", text_col: "TXT"})
    d = add_bucket(d, "A", granularity)
    d = ensure_bucket_and_label(d, "A", granularity)

    # split + explode (БЕЗ запятой!) — корректный вариант
    d["TXT"] = d["TXT"].astype(str).str.strip().replace({"nan": ""})
    d["piece"] = d["TXT"].str.split(r"[;\/\n|]+", regex=True)
    d = d.explode("piece")
    d["piece"] = d["piece"].astype(str).str.strip()
    d = d[d["piece"] != ""]

    if d.empty:
        return pd.DataFrame(columns=["bucket","bucket_label","templ_es","templ_en","count"])

    norm = (d["piece"].str.normalize("NFKD")
                    .str.encode("ascii","ignore").str.decode("ascii")
                    .str.lower().str.replace(r"[^\w\s]", " ", regex=True)
                    .str.replace(r"\s+", " ", regex=True).str.strip())
    d["norm"] = norm

    rows = []
    for es, en in templates_es_en:
        es_norm = _norm_local(es)
        mask = d["norm"].eq(es_norm) | d["norm"].str.contains(es_norm, na=False)
        if mask.any():
            tmp = d.loc[mask, ["bucket","bucket_label"]].copy()
            tmp["templ_es"] = es
            tmp["templ_en"] = en
            rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["bucket","bucket_label","templ_es","templ_en","count"])

    out = (pd.concat(rows, ignore_index=True)
             .groupby(["bucket","bucket_label","templ_es","templ_en"], as_index=False)
             .size().rename(columns={"size":"count"}))
    return out

@st.cache_data(show_spinner=False)
def build_aspects_counts(df: pd.DataFrame, text_col: str, date_col: str, granularity: str):
    """
    FR1:E. Быстро: split+explode, нормализация, матч по eq/contains, без циклов по строкам.
    Запятая ИСПОЛЬЗУЕТСЯ как разделитель (как в твоей исходной версии).
    """
    need_cols = [date_col, text_col]
    if df.empty or not all(c in df.columns for c in need_cols):
        return (pd.DataFrame(columns=["bucket","bucket_label","aspect","count"]),
                pd.DataFrame(columns=["en","mention","total"]))

    d = df[need_cols].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    if d.empty:
        return (pd.DataFrame(columns=["bucket","bucket_label","aspect","count"]),
                pd.DataFrame(columns=["en","mention","total"]))

    d = d.rename(columns={date_col: "A", text_col: "TXT"})
    d = add_bucket(d, "A", granularity)
    d = ensure_bucket_and_label(d, "A", granularity)

    # split + explode (С ЗАПЯТОЙ)
    d["TXT"] = d["TXT"].astype(str).str.strip().replace({"nan": ""})
    d["piece"] = d["TXT"].str.split(r"[;,/\n|]+", regex=True)
    d = d.explode("piece")
    d["piece"] = d["piece"].astype(str).str.strip()
    d = d[d["piece"] != ""]

    if d.empty:
        return (pd.DataFrame(columns=["bucket","bucket_label","aspect","count"]),
                pd.DataFrame(columns=["en","mention","total"]))

    # нормализация
    norm = (d["piece"].str.normalize("NFKD")
                    .str.encode("ascii","ignore").str.decode("ascii")
                    .str.lower().str.replace(r"[^\w\s]", " ", regex=True)
                    .str.replace(r"\s+", " ", regex=True).str.strip())
    d["norm"] = norm

    # матчим по eq или contains для каждого шаблона (их немного → быстро)
    rows = []
    known_mask = pd.Series(False, index=d.index)
    for es_norm, es, en in _ASPECTS_NORM:
        m = d["norm"].eq(es_norm) | d["norm"].str.contains(es_norm, na=False)
        if m.any():
            tmp = d.loc[m, ["bucket","bucket_label"]].copy()
            tmp["aspect"] = es + " (EN: " + en + ")"
            rows.append(tmp)
            known_mask |= m

    if rows:
        counts = (pd.concat(rows, ignore_index=True)
                    .groupby(["bucket","bucket_label","aspect"], as_index=False)
                    .size().rename(columns={"size":"count"}))
    else:
        counts = pd.DataFrame(columns=["bucket","bucket_label","aspect","count"])

    # неизвестные — всё, что не сматчилось
    unk = d.loc[~known_mask, "piece"].astype(str)
    if unk.empty:
        unknown_df = pd.DataFrame(columns=["en","mention","total"])
    else:
        unknown_df = (unk.value_counts().rename_axis("mention").reset_index(name="total"))
        unknown_df["en"] = unknown_df["mention"].apply(translate_es_to_en_safe)
        unknown_df = unknown_df[["en","mention","total"]]

    return counts, unknown_df

@st.cache_data(show_spinner=False)
def build_aspects_counts_by_S(df: pd.DataFrame) -> pd.DataFrame:
    """Счётчики аспектов по урокам S (только шаблонные аспекты, EN)."""
    if df.empty or not {"S","E"}.issubset(df.columns):
        return pd.DataFrame(columns=["S","aspect_en","count"])
    d = df[["S","E"]].copy()
    d["S"] = pd.to_numeric(d["S"], errors="coerce")
    d = d.dropna(subset=["S"])
    d["S"] = d["S"].astype(int)

    rows = []
    for _, r in d.iterrows():
        txt = str(r["E"] or "").strip()
        if not txt:
            continue
        parts = re.split(r"[;,/\n|]+", txt) if re.search(r"[;,/\n|]", txt) else [txt]
        for p in parts:
            t = _norm(p.strip())
            if not t:
                continue
            for es_norm, es, en in _ASPECTS_NORM:
                if t == es_norm or es_norm in t:
                    rows.append((int(r["S"]), en, 1))
                    break

    if not rows:
        return pd.DataFrame(columns=["S","aspect_en","count"])
    out = pd.DataFrame(rows, columns=["S","aspect_en","count"])
    out = out.groupby(["S","aspect_en"], as_index=False)["count"].sum()
    return out

@st.cache_data(show_spinner=False)
def build_aspects_counts_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Счётчики аспектов по месяцам R (EN-лейблы). Возвращает [R, aspect_en, count]."""
    if df.empty or not {"R","E"}.issubset(df.columns):
        return pd.DataFrame(columns=["R","aspect_en","count"])
    d = df[["R","E"]].copy()
    d["R"] = pd.to_numeric(d["R"], errors="coerce")
    d = d.dropna(subset=["R"])
    d["R"] = d["R"].astype(int)

    rows = []
    for _, r in d.iterrows():
        txt = str(r["E"] or "").strip()
        if not txt:
            continue
        parts = re.split(r"[;,/\n|]+", txt) if re.search(r"[;,/\n|]", txt) else [txt]
        for p in parts:
            t = _norm(p.strip())
            if not t:
                continue
            for es_norm, es, en in _ASPECTS_NORM:
                if t == es_norm or es_norm in t:
                    rows.append((int(r["R"]), en, 1))
                    break

    if not rows:
        return pd.DataFrame(columns=["R","aspect_en","count"])
    out = pd.DataFrame(rows, columns=["R","aspect_en","count"])
    return out.groupby(["R","aspect_en"], as_index=False)["count"].sum()

def aspect_to_en_label(s: str) -> str:
    """Из 'ES (EN: EN)' достаём 'EN'."""
    m = re.search(r"\(EN:\s*(.*?)\)\s*$", str(s))
    return m.group(1).strip() if m else str(s)

@st.cache_data(show_spinner=False)
def build_dislike_counts_by_S(df: pd.DataFrame) -> pd.DataFrame:
    """Счётчики dislike-аспектов по урокам S из колонки F (FR1)."""
    if df.empty or not {"S", "F"}.issubset(df.columns):
        return pd.DataFrame(columns=["S", "aspect_en", "count"])

    d = df[["S", "F"]].copy()
    d["S"] = pd.to_numeric(d["S"], errors="coerce")
    d = d.dropna(subset=["S"])
    d["S"] = d["S"].astype(int)

    dislike_norm = [(_norm_local(es), en) for es, en in DISLIKE_ES_EN]

    rows = []
    for _, r in d.iterrows():
        txt = str(r["F"] or "").strip()
        if not txt:
            continue
        parts = re.split(r"[;,/\n|]+", txt) if re.search(r"[;,/\n|]", txt) else [txt]
        for p in parts:
            t = _norm_local(p.strip())
            if not t:
                continue
            for es_norm, en in dislike_norm:
                if t == es_norm or es_norm in t:
                    rows.append((int(r["S"]), en, 1))
                    break

    if not rows:
        return pd.DataFrame(columns=["S", "aspect_en", "count"])

    out = pd.DataFrame(rows, columns=["S", "aspect_en", "count"])
    out = out.groupby(["S", "aspect_en"], as_index=False)["count"].sum()
    return out

@st.cache_data(show_spinner=False)
def build_template_counts_by_R(
    df: pd.DataFrame,
    text_col: str,
    templates_es_en: list[tuple[str, str]],
) -> pd.DataFrame:
    """
    Счётчики шаблонов по урокам R (FR2).
    Возвращает DataFrame [R(int), templ_en(str), count(int)].
    Важно: запятая НЕ является разделителем (чтобы не ломать 'Sí, todo a tiempo').
    """
    if df.empty or not {"R", text_col}.issubset(df.columns):
        return pd.DataFrame(columns=["R", "templ_en", "count"])

    d = df[["R", text_col]].copy()
    d["R"] = pd.to_numeric(d["R"], errors="coerce")
    d = d.dropna(subset=["R"])
    d["R"] = d["R"].astype(int)

    # нормализованные шаблоны
    tmpl_norm = [(_norm_local(es), en) for es, en in templates_es_en]

    rows = []
    splitter = re.compile(r"[;\/\n|]+")  # ; / | и перенос строки (БЕЗ запятой)
    for _, r in d.iterrows():
        raw = str(r[text_col] or "").strip()
        if not raw:
            continue
        parts = splitter.split(raw) if splitter.search(raw) else [raw]
        for p in parts:
            t = _norm_local(p.strip())
            if not t:
                continue
            for es_norm, en in tmpl_norm:
                if t == es_norm or es_norm in t:
                    rows.append((int(r["R"]), en, 1))
                    break

    if not rows:
        return pd.DataFrame(columns=["R", "templ_en", "count"])

    out = (pd.DataFrame(rows, columns=["R", "templ_en", "count"])
           .groupby(["R", "templ_en"], as_index=False)["count"].sum())
    return out

def _safe_text(x):
    """Всегда возвращает строку (безопасно для сортировки)."""
    if x is None:
        return ""
    try:
        if isinstance(x, float) and pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)

def translate_es_to_en_safe(text: str) -> str:
    """Перевод с защитой от эксепшенов и None."""
    s = _safe_text(text)
    try:
        out = translate_es_to_en(s)
        return _safe_text(out)  # гарантия строки
    except Exception:
        return s

def ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Гарантирует наличие перечисленных колонок; если нет — создаёт пустые."""
    if df.empty:
        # создаём пустой df с нужными колонками
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df

# ==================== ДАННЫЕ ====================

df1 = load_sheet_as_letter_df_cached("Form Responses 1")
df2 = load_sheet_as_letter_df_cached("Form Responses 2")

# Гарантируем наличие всех колонок, которые используются в коде ниже
req_fr1 = ["A", "N", "S", "G", "E", "F", "H", "R"]
req_fr2 = ["A", "M", "R", "I", "Q", "D", "E", "F", "G", "H", "J", "K"]
df1 = ensure_columns(df1, req_fr1)
df2 = ensure_columns(df2, req_fr2)

# Приведение типов — только для реально существующих колонок
if not df1.empty:
    if "A" in df1.columns:
        df1["A"] = pd.to_datetime(df1["A"], errors="coerce")
    for col in ["S", "G", "R"]:
        if col in df1.columns:
            df1[col] = pd.to_numeric(df1[col], errors="coerce")

if not df2.empty:
    if "A" in df2.columns:
        df2["A"] = pd.to_datetime(df2["A"], errors="coerce")
    # числовые оси/оценки
    for col in ["R", "Q", "I", "F", "G", "H"]:
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors="coerce")

# === Оси анализа: теперь работаем по МЕСЯЦАМ ===
AX_FR1 = "R"  # FR1: номер месяца
AX_FR2 = "Q"  # FR2: номер месяца
AX_NAME = "Month"

# ==================== ЕДИНЫЕ ФИЛЬТРЫ ====================

st.sidebar.header("Filters")

if st.sidebar.button("Refresh data"):
    load_sheet_as_letter_df_cached.clear()
    load_refunds_letter_df_cached.clear()
    load_qa_letter_df_cached.clear()
    st.rerun()

# Курсы
courses_union = sorted(list(set(
    ([] if df1.empty else df1["N"].dropna().unique().tolist()) +
    ([] if df2.empty else df2["M"].dropna().unique().tolist())
)))
if "courses_selected" not in st.session_state:
    st.session_state["courses_selected"] = courses_union.copy()

b1, b2 = st.sidebar.columns(2)
if b1.button("Select all"):
    st.session_state["courses_selected"] = courses_union.copy()
    st.rerun()
if b2.button("Clear"):
    st.session_state["courses_selected"] = []
    st.rerun()

selected_courses = st.sidebar.multiselect(
    "Курсы",
    options=courses_union,
    default=st.session_state["courses_selected"],
    key="courses_selected",
    help="Можно выбрать несколько; поиск поддерживается."
)
st.sidebar.caption(f"Выбрано: {len(selected_courses)} из {len(courses_union)}")

# Дата
min1, max1 = (df1["A"].min(), df1["A"].max()) if not df1.empty else (pd.NaT, pd.NaT)
min2, max2 = (df2["A"].min(), df2["A"].max()) if not df2.empty else (pd.NaT, pd.NaT)
glob_min, glob_max = safe_minmax(min1, max1, min2, max2)
if pd.isna(glob_min) or pd.isna(glob_max):
    date_range = st.sidebar.date_input("Дата фидбека (A)", [])
else:
    date_range = st.sidebar.date_input("Дата фидбека (A)", [glob_min.date(), glob_max.date()])

# Гранулярность
granularity = st.sidebar.selectbox(
    "Гранулярность для распределения",
    ["День", "Неделя", "Месяц", "Год"],
    index=1  # по умолчанию Неделя
)
BAR_SIZE = {"День": 18, "Неделя": 44, "Месяц": 56, "Год": 64}
bar_size = BAR_SIZE.get(granularity, 36)

TOP_K = 6

def _apply_q_filter(df_src: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает df_src, отфильтрованный по выбранным месяцам (FR2: Q),
    если список selected_months не пуст. Приводит Q к int.
    """
    if df_src.empty or "Q" not in df_src.columns:
        return df_src.copy()
    d = df_src.copy()
    d["Q"] = pd.to_numeric(d["Q"], errors="coerce")
    d = d.dropna(subset=["Q"])
    d["Q"] = d["Q"].astype(int)
    if selected_months:
        d = d[d["Q"].isin(selected_months)]
    return d

# ---- БАЗОВЫЕ ФИЛЬТРЫ ПО КУРСАМ/ДАТЕ ДЛЯ ОБОИХ ЛИСТОВ ----
df1_base = filter_df(df1, "N", "A", selected_courses, date_range)
df2_base = filter_df(df2, "M", "A", selected_courses, date_range)

# ---- ЕДИНЫЙ ФИЛЬТР МЕСЯЦЕВ (union FR1:R и FR2:Q) ----
if not df1_base.empty and AX_FR1 in df1_base.columns:
    fr1_vals = pd.to_numeric(df1_base[AX_FR1], errors="coerce").dropna().astype(int).unique().tolist()
else:
    fr1_vals = []
if not df2_base.empty and AX_FR2 in df2_base.columns:
    fr2_vals = pd.to_numeric(df2_base[AX_FR2], errors="coerce").dropna().astype(int).unique().tolist()
else:
    fr2_vals = []

months_options = sorted(list(set(fr1_vals + fr2_vals)))

if "months_selected" not in st.session_state:
    st.session_state["months_selected"] = months_options.copy()

sb1, sb2 = st.sidebar.columns(2)
if sb1.button("All months"):
    st.session_state["months_selected"] = months_options.copy()
    st.rerun()
if sb2.button("Clear months"):
    st.session_state["months_selected"] = []
    st.rerun()

selected_months = st.sidebar.multiselect(
    "Месяцы (FR1:R / FR2:Q)",
    options=months_options,
    default=st.session_state["months_selected"],
    key="months_selected",
    help="Единый фильтр по номеру месяца"
)

selected_lessons: list[int] = []

# ==================== ПРИМЕНЕНИЕ ФИЛЬТРОВ ====================

# Верхние графики (средние)
agg1 = apply_filters_and_aggregate(df1, "N", "A", AX_FR1, "G", selected_courses, date_range)
if not agg1.empty and selected_months:
    agg1[AX_FR1] = pd.to_numeric(agg1[AX_FR1], errors="coerce").astype("Int64")
    agg1 = agg1[agg1[AX_FR1].isin(selected_months)]
    agg1[AX_FR1] = agg1[AX_FR1].astype(int)

agg2 = apply_filters_and_aggregate(df2, "M", "A", AX_FR2, "I", selected_courses, date_range)
if not agg2.empty and selected_months:
    agg2[AX_FR2] = pd.to_numeric(agg2[AX_FR2], errors="coerce").astype("Int64")
    agg2 = agg2[agg2[AX_FR2].isin(selected_months)]
    agg2[AX_FR2] = agg2[AX_FR2].astype(int)

# Нижние распределения (сырые) + S/R-фильтр
df1_f = df1_base.copy()
if not df1_f.empty and selected_months:
    df1_f["ax_num"] = pd.to_numeric(df1_f[AX_FR1], errors="coerce")
    df1_f = df1_f[df1_f["ax_num"].isin(selected_months)]

df2_f = df2_base.copy()
if not df2_f.empty and selected_months:
    df2_f["ax_num"] = pd.to_numeric(df2_f[AX_FR2], errors="coerce")
    df2_f = df2_f[df2_f["ax_num"].isin(selected_months)]

if not df1_f.empty:
    df1_f = df1_f.dropna(subset=["A", "G"])
    df1_f = add_bucket(df1_f, "A", granularity)
    df1_f = ensure_bucket_and_label(df1_f, "A", granularity)

if not df2_f.empty:
    df2_f = df2_f.dropna(subset=["A", "I"])
    df2_f = add_bucket(df2_f, "A", granularity)
    df2_f = ensure_bucket_and_label(df2_f, "A", granularity)

fr1_out, fr1_bucket_order, fr1_val_order, fr1_title = prep_distribution(df1_f, "G", [1,2,3,4,5], "Score")
fr2_out, fr2_bucket_order, fr2_val_order, fr2_title = prep_distribution(df2_f, "I", list(range(1,11)), "Score")

# ==================== ОТРИСОВКА ====================

# === Tabs ===
section = st.sidebar.radio("Раздел", ["Feedback", "Refunds (LatAm)", "QA (analytics)"], index=0)

if section == "Feedback":

    # ---------- ЕДИНАЯ «РЕАЛИСТИЧНАЯ» ШКАЛА (перцентиль 0–100) ПО УРОКАМ ----------
    st.subheader("Lesson scores (percentile 0–100)")
    
    st.markdown(
        """
    **How to read this chart (percentile 0–100):**
    - For each response we convert the raw score to a **percentile (0–100)** *within the currently filtered data* (average-ties). 
    Then we **average those percentiles per month (module)** — separately for **Monthly feedback** and **Lesson feedback**. Each line is that monthly average.
    - At the same month, the line that is **higher** had relatively stronger feedback **that month** (given the current filters).
    - A line **rising over months** suggests improving relative feedback; **falling** suggests weakening relative feedback.
    - Move the cursor over a point to see the **month (module)**, **source** (Monthly vs Lesson), **average percentile**, and **number of answers**.
    - Percentiles are **relative** to the current filters (courses, date range, months). Changing filters changes the baseline and therefore the plotted values.
    """
    )
    
    # Источники под текущие фильтры и по выбранным урокам
    # FR1: берём месяц (AX_FR1="R") и оценку G
    df1_pct = df1_base.copy()
    if not df1_pct.empty:
        df1_pct[AX_FR1] = pd.to_numeric(df1_pct[AX_FR1], errors="coerce")  # R
        df1_pct["G"] = pd.to_numeric(df1_pct["G"], errors="coerce")
        df1_pct = df1_pct.dropna(subset=[AX_FR1, "G"])
        if selected_months:
            df1_pct = df1_pct[df1_pct[AX_FR1].astype(int).isin(selected_months)]
        if not df1_pct.empty:
            df1_pct[AX_FR1] = df1_pct[AX_FR1].astype(int)
            # перцентиль в пределах текущей выборки FR1
            df1_pct["score100"] = _to_percentile_0_100(df1_pct, "G")
            fr1_u = (df1_pct.groupby(AX_FR1, as_index=False)
                              .agg(avg_score100=("score100","mean"),
                                   count=("score100","size"))
                              .rename(columns={AX_FR1: "Month"})   # ключевой момент: ось назовём "Month"
                              .assign(source="Monthly feedback"))
        else:
            fr1_u = pd.DataFrame(columns=["Month","avg_score100","count","source"])
    else:
        fr1_u = pd.DataFrame(columns=["Month","avg_score100","count","source"])
    
    # FR2: берём месяц (AX_FR2="Q") и оценку I
    df2_pct = df2_base.copy()
    if not df2_pct.empty:
        df2_pct[AX_FR2] = pd.to_numeric(df2_pct[AX_FR2], errors="coerce")  # Q
        df2_pct["I"] = pd.to_numeric(df2_pct["I"], errors="coerce")
        df2_pct = df2_pct.dropna(subset=[AX_FR2, "I"])
        if selected_months:
            df2_pct = df2_pct[df2_pct[AX_FR2].astype(int).isin(selected_months)]
        if not df2_pct.empty:
            df2_pct[AX_FR2] = df2_pct[AX_FR2].astype(int)
            df2_pct["score100"] = _to_percentile_0_100(df2_pct, "I")
            fr2_u = (df2_pct.groupby(AX_FR2, as_index=False)
                              .agg(avg_score100=("score100","mean"),
                                   count=("score100","size"))
                              .rename(columns={AX_FR2: "Month"})   # приводим к общей оси "Month"
                              .assign(source="Lesson feedback"))
        else:
            fr2_u = pd.DataFrame(columns=["Month","avg_score100","count","source"])
    else:
        fr2_u = pd.DataFrame(columns=["Month","avg_score100","count","source"])
    
    unified = pd.concat([fr1_u, fr2_u], ignore_index=True)
    
    if unified.empty:
        st.info("Нет данных для объединённой шкалы (перцентиль 0–100) при текущих фильтрах.")
    else:
        ymin = float(unified["avg_score100"].min())
        ymax = float(unified["avg_score100"].max())
        pad = (ymax - ymin) * 0.1 if ymax > ymin else 2.5
        y_scale_u = alt.Scale(domain=[max(0, ymin - pad), min(100, ymax + pad)], nice=False, clamp=True)
    
        # Ось месяцев строго как целые
        unified["Month"] = pd.to_numeric(unified["Month"], errors="coerce").astype("Int64")
        unified = unified.dropna(subset=["Month"]).copy()
        unified["Month"] = unified["Month"].astype(int)

    
        ch_unified = (
            alt.Chart(unified)
              .mark_line(point=True)
              .encode(
                  x=alt.X("Month:O", title="Month", sort="ascending"),  # дискретные целые метки
                  y=alt.Y("avg_score100:Q", title="Percentile score (0–100)", scale=y_scale_u),
                  color=alt.Color("source:N", title="Source"),
                  tooltip=[
                      alt.Tooltip("source:N", title="Source"),
                      alt.Tooltip("avg_score100:Q", title="Avg percentile", format=".1f"),
                      alt.Tooltip("count:Q", title="Answers"),
                  ],
              )
              .properties(height=380)
        )
        st.altair_chart(ch_unified, use_container_width=True, theme=None)
    
    with st.expander("Average score per lessons — show/hide", expanded=False):
        col1, col2 = st.columns([1, 1])
    
        with col1:
            st.markdown("**Monthly feedback — average lesson score**")  # fix typo: Montly -> Monthly
            if agg1.empty:
                st.info("Нет данных для выбранных фильтров.")
            else:
                y_min = float(agg1["avg_y"].min()) if len(agg1) else 0.0
                y_max = float(agg1["avg_y"].max()) if len(agg1) else 5.0
                pad = (y_max - y_min) * 0.1 if y_max > y_min else 0.5
                y_scale = alt.Scale(domain=[y_min - pad, y_max + pad], nice=False, clamp=True)
    
                chart1 = (
                    alt.Chart(agg1)
                      .mark_line(
                          color="#f59e0b",
                          point=alt.OverlayMarkDef(color="#f59e0b", filled=True)
                      )
                      .encode(
                          x=alt.X(f"{AX_FR1}:O", title="Month", sort="ascending"),
                          y=alt.Y("avg_y:Q", title="Average score", scale=y_scale),
                          tooltip=[
                              alt.Tooltip("avg_y:Q", title="Average score", format=".2f"),
                              alt.Tooltip("count:Q", title="Answers"),
                          ],
                      )
                      .properties(height=380)
                )
                st.altair_chart(chart1, use_container_width=True, theme=None)
    
        with col2:
            st.markdown("**Lesson feedback — average lesson score**")
            if agg2.empty:
                st.info("Нет данных для выбранных фильтров.")
            else:
                y_min2 = float(agg2["avg_y"].min()) if len(agg2) else 0.0
                y_max2 = float(agg2["avg_y"].max()) if len(agg2) else 10.0
                pad2 = (y_max2 - y_min2) * 0.1 if y_max2 > y_min2 else 0.5
                y_scale2 = alt.Scale(domain=[y_min2 - pad2, y_max2 + pad2], nice=False, clamp=True)
    
                chart2 = (
                    alt.Chart(agg2).mark_line(point=True)
                      .encode(
                          x=alt.X(f"{AX_FR2}:O", title="Month", sort="ascending"),
                          y=alt.Y("avg_y:Q", title="Average score", scale=y_scale2),
                          tooltip=[
                              alt.Tooltip("avg_y:Q", title="Average score", format=".2f"),
                              alt.Tooltip("count:Q", title="Answers")
                          ])
                      .properties(height=380)
                )
                st.altair_chart(chart2, use_container_width=True, theme=None)
    
    # ---------- РАСПРЕДЕЛЕНИЕ ПО месяцам (в %) ДЛЯ ТЕХ ЖЕ ШКАЛ ----------
    st.markdown("---")
    st.subheader("Scores — distribution throughout the course")

    @st.cache_data(show_spinner=False)
    def _build_numeric_counts_by_axis(df_src: pd.DataFrame, axis_col: str, val_col: str, allowed_vals: list[int] | None):
        if df_src.empty or axis_col not in df_src.columns or val_col not in df_src.columns:
            return pd.DataFrame(columns=[axis_col, "val", "val_str", "count", "total"])
    
        d = df_src.copy()
        d[axis_col] = pd.to_numeric(d[axis_col], errors="coerce")
        d[val_col]  = pd.to_numeric(d[val_col],  errors="coerce")
        d = d.dropna(subset=[axis_col, val_col])
        if d.empty:
            return pd.DataFrame(columns=[axis_col, "val", "val_str", "count", "total"])
    
        if allowed_vals is not None:
            d = d[d[val_col].isin(allowed_vals)]
            if d.empty:
                return pd.DataFrame(columns=[axis_col, "val", "val_str", "count", "total"])
    
        d[axis_col] = d[axis_col].astype(int)
        d["val"] = d[val_col].astype(int)
        d["val_str"] = d["val"].astype(str)
    
        grp = (d.groupby([axis_col, "val", "val_str"], as_index=False)
                 .size().rename(columns={"size": "count"}))
        totals = (grp.groupby(axis_col, as_index=False)["count"]
                    .sum().rename(columns={"count": "total"}))
        out = grp.merge(totals, on=axis_col, how="left")
        return out

    @st.cache_data(show_spinner=False)
    def _pack_full_tooltip_axis(df_src: pd.DataFrame, x_col: str, legend_title: str):
        need = {x_col, "val", "count"}
        if df_src.empty or not need.issubset(df_src.columns):
            return pd.DataFrame(columns=[x_col, "total"]), []
    
        d = df_src[[x_col, "val", "count"]].copy()
        d["val"] = pd.to_numeric(d["val"], errors="coerce")
        d["count"] = pd.to_numeric(d["count"], errors="coerce").fillna(0).astype(int)
        d = d.dropna(subset=["val"])
        if d.empty:
            return pd.DataFrame(columns=[x_col, "total"]), []
    
        d["val"] = d["val"].astype(int)
    
        wide = (
            d.groupby([x_col, "val"], as_index=False)["count"].sum()
             .pivot(index=x_col, columns="val", values="count")
             .fillna(0).astype(int)
             .reset_index()
        )
    
        val_cols = sorted([c for c in wide.columns if c != x_col])
        wide["total"] = wide[val_cols].sum(axis=1).astype(int)
    
        tips = []
        for v in val_cols:
            out_col = f"tt_{v}"
            tips.append((out_col, str(v)))
    
            def _fmt_row(r):
                c = int(r.get(v, 0))
                t = int(r.get("total", 0))
                if t <= 0:
                    return f"{c} (0%)" if c else ""
                pct = c / t
                return f"{c} ({pct:.0%})" if c > 0 else ""
    
            wide[out_col] = wide.apply(_fmt_row, axis=1).astype(str)
    
        return wide[[x_col, "total"] + [c for c, _ in tips]], tips
    
    def _make_percent_stack_by_axis(out_df: pd.DataFrame, axis_col: str, legend_title: str):
        if out_df.empty:
            return None
        axis_order = sorted(out_df[axis_col].unique().tolist())
        base = alt.Chart(out_df).transform_calculate(pct='datum.count / datum.total')
        bars = (
            base.mark_bar(size=28, stroke=None, strokeWidth=0)
                .encode(
                    x=alt.X(f"{axis_col}:O", title="Month", sort=axis_order),
                    y=alt.Y("count:Q", stack="normalize",
                            axis=alt.Axis(format="%", title="% of answers"),
                            scale=alt.Scale(domain=[0,1], nice=False, clamp=True)),
                    color=alt.Color("val_str:N", title=legend_title,
                                    sort=alt.SortField(field="val", order="ascending"),
                                    legend=alt.Legend(orient="bottom", direction="horizontal",
                                                      columns=5, labelLimit=1000, titleLimit=1000,
                                                      symbolType="square")),
                    order=alt.Order("val:Q", sort="ascending"),
                )
        )
    
        tmp = out_df.copy()
        tmp["val"] = pd.to_numeric(tmp.get("val", tmp["val_str"]), errors="coerce").astype("Int64")
        vals = sorted([int(v) for v in tmp["val"].dropna().unique().tolist()])
    
        wide = tmp.pivot_table(index=axis_col, columns="val", values="count",
                               aggfunc="sum", fill_value=0).reset_index()
        wide["total"] = wide[[v for v in vals]].sum(axis=1).astype(int)
        for v in vals:
            c = wide[v].astype(int)
            t = wide["total"].replace(0, pd.NA)
            pct = (c / t).round(3)
            wide[str(v)] = np.where(
                (wide["total"] > 0) & (c > 0),
                c.astype(str) + " (" + (pct.fillna(0).map(lambda x: f"{x:.0%}")) + ")",
                ""
            )
        tooltip_cols = [str(v) for v in vals]
        wide[tooltip_cols] = wide[tooltip_cols].fillna("").astype(str)
    
        overlay = (
            alt.Chart(wide)
              .mark_bar(size=28, opacity=0.001)
              .encode(
                  x=alt.X(f"{axis_col}:O", sort=axis_order),
                  y=alt.Y("total:Q"),
                  tooltip=[alt.Tooltip("total:Q", title="All answers"),
                           *[alt.Tooltip(f"{col}:N", title=col) for col in tooltip_cols]]
              )
        )
        return (bars + overlay).configure_legend(labelLimit=1000, titleLimit=1000)
    
    # Источники с учётом фильтров
    df1_months_G = df1_base.copy()
    if not df1_months_G.empty and selected_months:
        df1_months_G[AX_FR1 + "_num"] = pd.to_numeric(df1_months_G[AX_FR1], errors="coerce")
        df1_months_G = df1_months_G[df1_months_G[AX_FR1 + "_num"].isin(selected_months)]
    
    df2_months_I = df2_base.copy()
    if not df2_months_I.empty and selected_months:
        df2_months_I[AX_FR2 + "_num"] = pd.to_numeric(df2_months_I[AX_FR2], errors="coerce")
        df2_months_I = df2_months_I[df2_months_I[AX_FR2 + "_num"].isin(selected_months)]
    
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.markdown("**Monthly feedback**")
        out_G_M = _build_numeric_counts_by_axis(df1_months_G, axis_col=AX_FR1, val_col="G", allowed_vals=[1,2,3,4,5])
        if out_G_M.empty:
            st.info("No data")
        else:
            ch_G_M = _make_percent_stack_by_axis(out_G_M, axis_col=AX_FR1, legend_title="Score")
            st.altair_chart(ch_G_M.properties(height=460), use_container_width=True, theme=None)
    
    with right_col:
        st.markdown("**Lesson feedback**")
        out_I_M = _build_numeric_counts_by_axis(df2_months_I, axis_col=AX_FR2, val_col="I", allowed_vals=list(range(1, 11)))
        if out_I_M.empty:
            st.info("No data")
        else:
            ch_I_M = _make_percent_stack_by_axis(out_I_M, axis_col=AX_FR2, legend_title="Score")
            st.altair_chart(ch_I_M.properties(height=460), use_container_width=True, theme=None)
    
    # ---------- РАСПРЕДЕЛЕНИЕ ЗНАЧЕНИЙ ПО месяцам (в %) ДЛЯ ТЕХ ЖЕ ШКАЛ ----------
    
    with st.expander("Scores — distribution in time (show/hide)", expanded=False):
        st.subheader("Scores - distribution in time")
    
        col3, col4 = st.columns([1, 1])
    
        with col3:
            st.markdown("**Monthly feedback**")
            if fr1_out.empty:
                st.info("Нет данных (FR1).")
            else:
                bars1 = (
                    alt.Chart(fr1_out).mark_bar(size=BAR_SIZE.get(granularity, 36), tooltip=False)
                      .encode(
                          x=alt.X("bucket_label:N", title="Period", sort=fr1_bucket_order),
                          y=alt.Y("sum(count):Q", title="Answers"),
                          color=alt.Color("val_str:N", title="Score", sort=fr1_val_order),
                          order=alt.Order("val:Q", sort="ascending"),
                      )
                      .properties(height=420)
                )
    
                # общий тултип на весь столбик
                _tmp1 = fr1_out.rename(columns={"bucket_label": "X"})
                packed1_in = _tmp1[["X", "val", "count"]].copy()
    
                totals1 = (packed1_in.groupby("X", as_index=False)["count"]
                                            .sum().rename(columns={"count": "total"}))
                packed1_in = packed1_in.merge(totals1, on="X", how="left")
    
                packed1_in["val"] = pd.to_numeric(packed1_in["val"], errors="coerce")
                packed1_in = packed1_in.dropna(subset=["val"])
                packed1_in["val"] = packed1_in["val"].astype(int)
                packed1_in["val_str"] = packed1_in["val"].astype(str)
    
                packed1, tips1 = _pack_full_tooltip_axis(packed1_in, x_col="X", legend_title="Score")
    
                overlay1 = (
                    alt.Chart(packed1).mark_bar(size=BAR_SIZE.get(granularity, 36), opacity=0.001)
                      .encode(
                          x=alt.X("X:N", sort=fr1_bucket_order),
                          y=alt.Y("total:Q"),
                          tooltip=[
                              *[alt.Tooltip(f"{col}:N", title=title) for col, title in tips1],
                              alt.Tooltip("total:Q", title="All answers"),
                          ]
                      )
                )
    
                st.altair_chart(
                    (bars1 + overlay1)
                        .properties(
                            height=420,
                            padding={"left": 10, "right": 48, "top": 10, "bottom": 70}
                        )
                        .configure_view(clip=False, stroke=None),
                    use_container_width=True, theme=None
                )
    
        with col4:
            st.markdown("**Lesson feedback**")
            if fr2_out.empty:
                st.info("Нет данных (FR2).")
            else:
                bars2 = (
                    alt.Chart(fr2_out).mark_bar(size=BAR_SIZE.get(granularity, 36), tooltip=False)
                      .encode(
                          x=alt.X("bucket_label:N", title="Period", sort=fr2_bucket_order),
                          y=alt.Y("sum(count):Q", title="Answers"),
                          color=alt.Color("val_str:N", title="Score", sort=fr2_val_order),
                          order=alt.Order("val:Q", sort="ascending"),
                      )
                      .properties(height=420)
                )
    
                _tmp2 = fr2_out.rename(columns={"bucket_label": "X"})
                packed2_in = _tmp2[["X", "val", "count"]].copy()
    
                totals2 = (packed2_in.groupby("X", as_index=False)["count"]
                                            .sum().rename(columns={"count": "total"}))
                packed2_in = packed2_in.merge(totals2, on="X", how="left")
    
                packed2_in["val"] = pd.to_numeric(packed2_in["val"], errors="coerce")
                packed2_in = packed2_in.dropna(subset=["val"])
                packed2_in["val"] = packed2_in["val"].astype(int)
                packed2_in["val_str"] = packed2_in["val"].astype(str)
    
                packed2, tips2 = _pack_full_tooltip_axis(packed2_in, x_col="X", legend_title="Score")
    
                overlay2 = (
                    alt.Chart(packed2).mark_bar(size=BAR_SIZE.get(granularity, 36), opacity=0.001)
                      .encode(
                          x=alt.X("X:N", sort=fr2_bucket_order),
                          y=alt.Y("total:Q"),
                          tooltip=[
                              *[alt.Tooltip(f"{col}:N", title=title) for col, title in tips2],
                              alt.Tooltip("total:Q", title="All answers"),
                          ]
                      )
                )
    
                st.altair_chart(
                    (bars2 + overlay2)
                        .properties(
                            height=420,
                            padding={"left": 10, "right": 48, "top": 10, "bottom": 70}
                        )
                        .configure_view(clip=False, stroke=None),
                    use_container_width=True, theme=None
                )
            
    # --------- ЕДИНАЯ ТАБЛИЦА ПО МЕСЯЦАМ (Month) — I–J / Aspects / Dislike / Comments + итоги ---------
    st.markdown("---")
    st.subheader("Detailed feedback")
    
    KEY_COL_FR1 = "R"   # FR1: Month #
    KEY_COL_FR2 = "Q"   # FR2: Month #
    KEY_LABEL    = "Month"  # имя колонки в выходной таблице
    
    # === Подготовка данных ДЛЯ СВОДНОЙ ТАБЛИЦЫ по МЕСЯЦАМ ===
    
    # 0) FR2 — пары I–J по месяцам (Q)
    ij_pairs_per_m = {}
    if not df2_base.empty and {KEY_COL_FR2, "I", "J"}.issubset(df2_base.columns):
        df2_IJ_src = df2_base.copy()
        dIJ = df2_IJ_src[[KEY_COL_FR2, "I", "J"]].copy()
        dIJ[KEY_COL_FR2] = pd.to_numeric(dIJ[KEY_COL_FR2], errors="coerce")
        dIJ["I"] = pd.to_numeric(dIJ["I"], errors="coerce")
        dIJ = dIJ.dropna(subset=[KEY_COL_FR2, "I", "J"])
        if not dIJ.empty:
            dIJ[KEY_COL_FR2] = dIJ[KEY_COL_FR2].astype(int)
            dIJ["I"] = dIJ["I"].astype(int)
            splitter_ij = re.compile(r"[;\/\n|]+")  # без запятой
            for _, rr in dIJ.iterrows():
                m = int(rr[KEY_COL_FR2])
                score = int(rr["I"])
                raw = str(rr["J"] or "").strip()
                if not raw:
                    continue
                parts = splitter_ij.split(raw) if splitter_ij.search(raw) else [raw]
                for p in parts:
                    t = p.strip()
                    if not t:
                        continue
                    key = (score, translate_es_to_en_safe(t))
                    ij_pairs_per_m.setdefault(m, Counter())[key] += 1
    
    # 1) Aspects (FR1:E) по месяцам (R)
    rows_aspects = []
    unknown_per_m = {}
    if not df1_base.empty and {KEY_COL_FR1, "E"}.issubset(df1_base.columns):
        d1 = df1_base[[KEY_COL_FR1, "E"]].copy()
        d1[KEY_COL_FR1] = pd.to_numeric(d1[KEY_COL_FR1], errors="coerce")
        d1 = d1.dropna(subset=[KEY_COL_FR1])
        d1[KEY_COL_FR1] = d1[KEY_COL_FR1].astype(int)
        for _, rr in d1.iterrows():
            m_val = int(rr[KEY_COL_FR1])
            txt = str(rr["E"] or "").strip()
            if not txt:
                continue
            parts = re.split(r"[;,/\n|]+", txt) if re.search(r"[;,/\n|]", txt) else [txt]
            for p in parts:
                p_clean = p.strip()
                t = _norm(p_clean)
                if not t:
                    continue
                matched = False
                for es_norm, es, en in _ASPECTS_NORM:
                    if t == es_norm or es_norm in t:
                        rows_aspects.append((m_val, en))
                        matched = True
                        break
                if not matched:
                    unknown_per_m.setdefault(m_val, Counter())[p_clean] += 1
    df_as = (pd.DataFrame(rows_aspects, columns=[KEY_LABEL, "aspect_en"])
             if rows_aspects else pd.DataFrame(columns=[KEY_LABEL, "aspect_en"]))
    df_as.rename(columns={KEY_LABEL: KEY_LABEL}, inplace=True)
    
    # 2) Dislike (FR1:F) по месяцам (R)
    rows_dislike = []
    unknown_dislike_per_m = {}
    if not df1_base.empty and {KEY_COL_FR1, "F"}.issubset(df1_base.columns):
        d1d = df1_base[[KEY_COL_FR1, "F"]].copy()
        d1d[KEY_COL_FR1] = pd.to_numeric(d1d[KEY_COL_FR1], errors="coerce")
        d1d = d1d.dropna(subset=[KEY_COL_FR1])
        d1d[KEY_COL_FR1] = d1d[KEY_COL_FR1].astype(int)
        dislike_norm = [(_norm_local(es), en) for es, en in DISLIKE_ES_EN]
        for _, rr in d1d.iterrows():
            m_val = int(rr[KEY_COL_FR1])
            txt = str(rr["F"] or "").strip()
            if not txt:
                continue
            parts = re.split(r"[;,/\n|]+", txt) if re.search(r"[;,/\n|]", txt) else [txt]
            for p in parts:
                p_clean = p.strip()
                t = _norm_local(p_clean)
                if not t:
                    continue
                matched = False
                for es_norm, en in dislike_norm:
                    if t == es_norm or es_norm in t:
                        rows_dislike.append((m_val, en))
                        matched = True
                        break
                if not matched:
                    unknown_dislike_per_m.setdefault(m_val, Counter())[p_clean] += 1
    df_dis = (pd.DataFrame(rows_dislike, columns=[KEY_LABEL, "aspect_en"])
              if rows_dislike else pd.DataFrame(columns=[KEY_LABEL, "aspect_en"]))
    
    # 3) Comments — FR1:H по R + FR2:K по Q (обе в Month)
    comments_per_month = {}
    splitter = re.compile(r"[;\/\n|]+")  # без запятой
    
    # FR1:H по R
    if not df1_base.empty and {KEY_COL_FR1, "H"}.issubset(df1_base.columns):
        d1c = df1_base[[KEY_COL_FR1, "H"]].copy()
        d1c[KEY_COL_FR1] = pd.to_numeric(d1c[KEY_COL_FR1], errors="coerce")
        d1c = d1c.dropna(subset=[KEY_COL_FR1])
        d1c[KEY_COL_FR1] = d1c[KEY_COL_FR1].astype(int)
        for _, rr in d1c.iterrows():
            m = int(rr[KEY_COL_FR1])
            raw = str(rr["H"] or "").strip()
            if not raw:
                continue
            parts = splitter.split(raw) if splitter.search(raw) else [raw]
            for p in parts:
                t = p.strip()
                if t:
                    comments_per_month.setdefault(m, Counter())[t] += 1
    
    # FR2:K по Q
    if not df2_base.empty and {KEY_COL_FR2, "K"}.issubset(df2_base.columns):
        d2c = df2_base[[KEY_COL_FR2, "K"]].copy()
        d2c[KEY_COL_FR2] = pd.to_numeric(d2c[KEY_COL_FR2], errors="coerce")
        d2c = d2c.dropna(subset=[KEY_COL_FR2])
        d2c[KEY_COL_FR2] = d2c[KEY_COL_FR2].astype(int)
        for _, rr in d2c.iterrows():
            m = int(rr[KEY_COL_FR2])
            raw = str(rr["K"] or "").strip()
            if not raw:
                continue
            parts = splitter.split(raw) if splitter.search(raw) else [raw]
            for p in parts:
                t = p.strip()
                if t:
                    comments_per_month.setdefault(m, Counter())[t] += 1
    
    # === Refunds (LatAm) → тексты L по месяцам для Detailed feedback ===
    # Берем: AS (дата/месяц), AU (флаг), AV (курс — опционально фильтруем), L (ТЕКСТ!)
    refunds_L_texts_by_month = {}
    
    # Refunds (LatAm) — читаем через кеш-функцию и страхуем колонки
    dfr = load_refunds_letter_df_cached()
    if dfr.empty:
        dfr = pd.DataFrame(columns=["AS", "AU", "L", "AV"])
    
    # подстрахуемся: создадим пустые нужные колонки, если их нет
    for c in ["AS", "AU", "L", "AV"]:
        if c not in dfr.columns:
            dfr[c] = pd.NA
    
    # дальше оставляй твой код обработки без изменений:
    # AU == TRUE, мягкий фильтр по AV, MonthNum из AS, сбор refunds_L_texts_by_month ...
    if not dfr.empty and {"AS", "AU", "L"}.issubset(dfr.columns):
        dfr["AU"] = dfr["AU"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        dfr = dfr[dfr["AU"]]
        if "AV" in dfr.columns and selected_courses:
            av = dfr["AV"].astype(str).str.strip()
            patt = "|".join([re.escape(c) for c in selected_courses]) if selected_courses else ""
            dfr = dfr[av.isin(selected_courses) | av.str.contains(patt, case=False, na=False)]
        dt = pd.to_datetime(dfr["AS"], errors="coerce")
        as_num = pd.to_numeric(dfr["AS"], errors="coerce")
        dfr["MonthNum"] = dt.dt.month.where(dt.notna(), as_num)
        dfr["MonthNum"] = pd.to_numeric(dfr["MonthNum"], errors="coerce").astype("Int64")
        dfr["L_text"] = dfr["L"].astype(str).str.strip()
        dfr = dfr[dfr["L_text"] != ""]
        for m, grp in dfr.groupby("MonthNum"):
            cnt = Counter()
            for t in grp["L_text"]:
                cnt[translate_es_to_en_safe(t)] += 1
            refunds_L_texts_by_month[int(m)] = cnt
    
    # === Формирование сводной таблицы ===
    all_months = sorted(set(
        list(ij_pairs_per_m.keys()) +
        (df_as[KEY_LABEL].unique().tolist() if not df_as.empty else []) +
        list(unknown_per_m.keys()) +
        (df_dis[KEY_LABEL].unique().tolist() if not df_dis.empty else []) +
        list(unknown_dislike_per_m.keys()) +
        list(comments_per_month.keys()) +
        list(refunds_L_texts_by_month.keys())    # ← тут ДОЛЖЕН быть refunds_L_texts_by_month
    ))
    
    if not all_months:
        st.info("Нет данных для сводной таблицы по выбранным фильтрам.")
    else:
        unified_rows = []
        for m in all_months:
            # I—J
            ij_counter = ij_pairs_per_m.get(m, Counter())
            total_pairs = int(sum(ij_counter.values()))
            if ij_counter:
                ij_items = sorted(
                    ij_counter.items(),
                    key=lambda kv: (-kv[1], -int(kv[0][0]), _safe_text(kv[0][1]).casefold())
                )
                ij_bullets = []
                for (score, txt_en), c in ij_items:
                    pct = (c / total_pairs) if total_pairs else 0.0
                    ij_bullets.append(f"• {score} — {txt_en} — {c} ({pct:.0%})")
                ij_text = "\n".join(ij_bullets)
            else:
                ij_text = ""
    
            # Aspects + Unknown
            if not df_as.empty and m in df_as[KEY_LABEL].values:
                cnt_aspects = (df_as[df_as[KEY_LABEL] == m]["aspect_en"]
                               .value_counts()
                               .sort_values(ascending=False))
                total_aspects_known = int(cnt_aspects.sum())
            else:
                cnt_aspects = pd.Series(dtype=int)
                total_aspects_known = 0
    
            unk_as = unknown_per_m.get(m, Counter())
            total_aspects_unknown = int(sum(unk_as.values()))
            unk_as_items = sorted(unk_as.items(), key=lambda x: (-x[1], _safe_text(x[0]).casefold()))[:10]
            rest_as_unk = total_aspects_unknown - sum(c for _, c in unk_as_items)
            total_aspects_all = total_aspects_known + total_aspects_unknown
    
            aspects_bul = []
            for en_name, c in cnt_aspects.items():
                pct = (c / total_aspects_all) if total_aspects_all else 0.0
                aspects_bul.append(f"• {en_name} — {c} ({pct:.0%})")
            if unk_as_items:
                if aspects_bul: aspects_bul.append("")
                for raw_txt, c in unk_as_items:
                    pct = (c / total_aspects_all) if total_aspects_all else 0.0
                    aspects_bul.append(f"• {translate_es_to_en_safe(raw_txt)} — {c} ({pct:.0%})")
                if rest_as_unk > 0:
                    aspects_bul.append(f"• … (+{rest_as_unk})")
    
            # Dislike + Unknown
            if not df_dis.empty and m in df_dis[KEY_LABEL].values:
                cnt_dis = (df_dis[df_dis[KEY_LABEL] == m]["aspect_en"]
                           .value_counts()
                           .sort_values(ascending=False))
                total_dis_known = int(cnt_dis.sum())
            else:
                cnt_dis = pd.Series(dtype=int)
                total_dis_known = 0
    
            unk_dis = unknown_dislike_per_m.get(m, Counter())
            total_dis_unknown = int(sum(unk_dis.values()))
            unk_dis_items = sorted(unk_dis.items(), key=lambda x: (-x[1], _safe_text(x[0]).casefold()))[:10]
            rest_dis_unk = total_dis_unknown - sum(c for _, c in unk_dis_items)
            total_dis_all = total_dis_known + total_dis_unknown
    
            dis_bul = []
            for en_name, c in cnt_dis.items():
                pct = (c / total_dis_all) if total_dis_all else 0.0
                dis_bul.append(f"• {en_name} — {c} ({pct:.0%})")
            if unk_dis_items:
                if dis_bul: dis_bul.append("")
                for raw_txt, c in unk_dis_items:
                    pct = (c / total_dis_all) if total_dis_all else 0.0
                    dis_bul.append(f"• {translate_es_to_en_safe(raw_txt)} — {c} ({pct:.0%})")
                if rest_dis_unk > 0:
                    dis_bul.append(f"• … (+{rest_dis_unk})")
    
            # Comments (EN)
            comm = comments_per_month.get(m, Counter())
            total_comm = int(sum(comm.values()))
            if comm:
                comm_items = sorted(comm.items(), key=lambda x: (-x[1], _safe_text(x[0]).casefold()))
                comm_bul = [f"• {translate_es_to_en_safe(txt)} — {c} ({(c/total_comm if total_comm else 0):.0%})"
                            for txt, c in comm_items]
                comm_txt = "\n".join(comm_bul)
            else:
                comm_txt = ""
    
            total_all = total_aspects_all + total_dis_all + total_comm + total_refunds_l
    
            # --- Refunds: только тексты из L (AU=TRUE), сгруппированные по месяцу ---
            l_counter = refunds_L_texts_by_month.get(m, Counter())
            total_refunds_l = int(sum(l_counter.values()))
            if l_counter:
                l_items = sorted(
                    l_counter.items(),
                    key=lambda kv: (-kv[1], _safe_text(kv[0]).casefold())
                )
                refunds_l_text = "\n".join(
                    f"• {phrase_en} — {c} ({(c/total_refunds_l if total_refunds_l else 0):.0%})"
                    for phrase_en, c in l_items
                )
            else:
                refunds_l_text = ""
    
            # --- Refunds: тексты L по месяцу (EN + count + %) ---
            L_counter = refunds_L_texts_by_month.get(m, Counter())
            total_L = int(sum(L_counter.values()))
            if L_counter:
                L_items = sorted(L_counter.items(), key=lambda kv: (-kv[1], _safe_text(kv[0]).casefold()))
                # можно ограничить верхушку TOP_N
                TOP_N = 12
                L_items = L_items[:TOP_N]
                L_bullets = [f"• {txt_en} — {c} ({(c/total_L if total_L else 0):.0%})"
                             for txt_en, c in L_items]
                if total_L > sum(c for _, c in L_items):
                    L_bullets.append(f"• … (+{total_L - sum(c for _, c in L_items)})")
                L_text_block = "\n".join(L_bullets)
            else:
                L_text_block = ""
        
            unified_rows.append({
                KEY_LABEL: int(m),
                "Score with argumentation": ij_text,
                "Total scores": total_pairs,
                "What liked": "\n".join(aspects_bul),
                "Total liked": total_aspects_all,
                "What disliked": "\n".join(dis_bul),
                "Total disliked": total_dis_all,
                "Other comments": comm_txt,
                "Total comments": total_comm,
                "Refunds": refunds_l_text,       # ← новое поле с буллетами по L
                "Total refunds": total_refunds_l,  # ← общее число строк-рефандов в месяце
                "Total mentions (all)": total_all,
            })
    
        unified_table = pd.DataFrame(unified_rows).sort_values(KEY_LABEL).reset_index(drop=True)
        height = min(1000, 140 + 28 * len(unified_table))
        st.dataframe(
            unified_table[
                [
                    KEY_LABEL,
                    "Score with argumentation", "Total scores",
                    "What liked", "Total liked",
                    "What disliked", "Total disliked",
                    "Other comments", "Total comments",
                    "Refunds",             # ← новое
                    "Total refunds",      # ← новое
                    "Total mentions (all)",
                ]
            ],
            use_container_width=True,
            height=height
        )
    
    # --------- ГРАФИК «Распределение по месяцам (ось X — R)» В % ---------
    st.markdown("---")
    st.subheader("Liked throughout the course")
    
    # источник: FR1 (df1_base) + (опционально) фильтр выбранных месяцев
    df_aspects_m = df1_base.copy()
    if not df_aspects_m.empty and selected_months:
        df_aspects_m["R_num"] = pd.to_numeric(df_aspects_m["R"], errors="coerce")
        df_aspects_m = df_aspects_m[df_aspects_m["R_num"].isin(selected_months)]
    
    cnt_by_m_all = build_aspects_counts_by_month(df_aspects_m)
    
    if cnt_by_m_all.empty:
        st.info("Нет данных для графика по месяцам.")
    else:
        base = (
            alt.Chart(cnt_by_m_all)
              .transform_aggregate(count='sum(count)', groupby=['R', 'aspect_en'])
              .transform_joinaggregate(total='sum(count)', groupby=['R'])
              .transform_calculate(pct='datum.count / datum.total')
        )
    
        legend_domain_en = [en for _, en in ASPECTS_ES_EN]
    
        bars_r = (
            base.mark_bar(size=28, stroke=None, strokeWidth=0)
                .encode(
                    x=alt.X("R:O", title="Month", sort="ascending"),
                    y=alt.Y(
                        "count:Q",
                        stack="normalize",
                        axis=alt.Axis(format="%", title="% of answers"),
                        scale=alt.Scale(domain=[0, 1], nice=False, clamp=True)
                    ),
                    color=alt.Color(
                        "aspect_en:N",
                        title="Liked",
                        scale=alt.Scale(domain=legend_domain_en),
                        legend=alt.Legend(
                            orient="bottom", direction="horizontal",
                            columns=2, labelLimit=1000, titleLimit=1000, symbolType="square",
                        ),
                    ),
                    tooltip=[
                        # Месяц (R) — убрали
                        alt.Tooltip("aspect_en:N", title="Liked"),
                        alt.Tooltip("count:Q",   title="Answers"),
                        alt.Tooltip("pct:Q",     title="%", format=".0%"),
                        alt.Tooltip("total:Q",   title="All answers"),
                    ],
                )
        ).configure_legend(labelLimit=1000, titleLimit=1000)
        
        st.altair_chart(
            bars_r.properties(title="Likes per months", height=460),
            use_container_width=True,
            theme=None
        )
    
    # --- спрятанный блок "Liked in time" под тогл ---
    with st.expander("Liked in time — show/hide", expanded=False):
        st.subheader("Liked in time")
    
        df_aspects = df1_base.copy()
        if not df_aspects.empty and selected_lessons:
            df_aspects["S_num"] = pd.to_numeric(df_aspects["S"], errors="coerce")
            df_aspects = df_aspects[df_aspects["S_num"].isin(selected_lessons)]
    
        asp_counts, _unknown_all = build_aspects_counts(
            df_aspects, text_col="E", date_col="A", granularity=granularity
        )
    
        if asp_counts.empty:
            st.info("Не нашёл упоминаний аспектов (лист 'Form Responses 1', колонка E).")
        else:
            asp_counts["aspect_en"] = asp_counts["aspect"].apply(aspect_to_en_label)
    
            bucket_order = (
                asp_counts[["bucket","bucket_label"]]
                .drop_duplicates()
                .sort_values("bucket")["bucket_label"].tolist()
            )
    
            totals_by_bucket = (
                asp_counts.groupby("bucket_label", as_index=False)["count"]
                          .sum().rename(columns={"count":"total"})
            )
            y_max = max(1, int(totals_by_bucket["total"].max()))
            y_scale_bar = alt.Scale(domain=[0, y_max * 1.1], nice=False, clamp=True)
    
            present = asp_counts["aspect"].unique().tolist()
    
            bars = (
                alt.Chart(asp_counts)
                  .mark_bar(size=max(40, BAR_SIZE.get(granularity, 36)))
                  .encode(
                      x=alt.X("bucket_label:N", title="Period", sort=bucket_order),
                      y=alt.Y("sum(count):Q", title="Answers", scale=y_scale_bar),
                      color=alt.Color("aspect:N", title="Liked", sort=present)
                  )
            )
    
            # общий тултип (как было)
            wide = (
                asp_counts
                .pivot_table(index=["bucket","bucket_label"], columns="aspect_en",
                             values="count", aggfunc="sum", fill_value=0)
            )
    
            col_order = list(wide.sum(axis=0).sort_values(ascending=False).index)
            TOP_K = 6
            top_names = col_order[:TOP_K]
    
            def _pack_row_named(r, names=top_names):
                total = int(r.sum())
                out = {"total": total}
                for i, name in enumerate(names, start=1):
                    c = int(r.get(name, 0))
                    out[f"t{i}"] = f"{name} — {c} ({c/total:.0%})" if total > 0 and c > 0 else ""
                return pd.Series(out)
    
            packed = wide.apply(_pack_row_named, axis=1).reset_index()
    
            bubble = (
                alt.Chart(packed)
                  .mark_bar(size=max(40, BAR_SIZE.get(granularity, 36)), opacity=0.001)
                  .encode(
                      x=alt.X("bucket_label:N", sort=bucket_order),
                      y=alt.Y("total:Q", scale=y_scale_bar),
                      tooltip=[
                          alt.Tooltip("t1:N", title=""),
                          alt.Tooltip("t2:N", title=""),
                          alt.Tooltip("t3:N", title=""),
                          alt.Tooltip("t4:N", title=""),
                          alt.Tooltip("t5:N", title=""),
                          alt.Tooltip("t6:N", title=""),
                      ],
                  )
            )
    
            st.altair_chart(
                (bars + bubble).properties(height=460),
                theme=None, use_container_width=True
            )
    
    # Подсказка, если онлайн-переводчик недоступен
    if _gt is None:
        st.caption("⚠️ deep-translator недоступен — используется упрощённый пословный перевод.")
    
    # --------- ГРАФИК «Распределение по месяцам (ось X — R)» В % — DISLIKED ---------
    st.markdown("---")
    st.subheader("Disliked throughout the course")
    
    # источник: FR1 (df1_base) + (опционально) фильтр выбранных месяцев (по R)
    df_dislike_m = df1_base.copy()
    if not df_dislike_m.empty and selected_months:
        df_dislike_m["R_num"] = pd.to_numeric(df_dislike_m["R"], errors="coerce")
        df_dislike_m = df_dislike_m[df_dislike_m["R_num"].isin(selected_months)]
    
    # Подсчёт: из FR1 берём R (месяц) и F (dislike-тексты), матчим на DISLIKE_ES_EN
    if df_dislike_m.empty or not {"R", "F"}.issubset(df_dislike_m.columns):
        cnt_by_m_dis = pd.DataFrame(columns=["R","aspect_en","count"])
    else:
        dsrc = df_dislike_m[["R","F"]].copy()
        dsrc["R"] = pd.to_numeric(dsrc["R"], errors="coerce")
        dsrc = dsrc.dropna(subset=["R"])
        dsrc["R"] = dsrc["R"].astype(int)
    
        dislike_norm = [(_norm_local(es), en) for es, en in DISLIKE_ES_EN]
        rows_dis = []
        for _, r in dsrc.iterrows():
            raw = str(r["F"] or "").strip()
            if not raw:
                continue
            parts = re.split(r"[;,/\n|]+", raw) if re.search(r"[;,/\n|]", raw) else [raw]
            for p in parts:
                t = _norm_local(p.strip())
                if not t:
                    continue
                for es_norm, en in dislike_norm:
                    if t == es_norm or es_norm in t:
                        rows_dis.append((int(r["R"]), en, 1))
                        break
    
        cnt_by_m_dis = (
            pd.DataFrame(rows_dis, columns=["R","aspect_en","count"])
            if rows_dis else pd.DataFrame(columns=["R","aspect_en","count"])
        )
    
    if cnt_by_m_dis.empty:
        st.info("Нет данных для графика по месяцам (Disliked).")
    else:
        base = (
            alt.Chart(cnt_by_m_dis)
              .transform_aggregate(count='sum(count)', groupby=['R', 'aspect_en'])
              .transform_joinaggregate(total='sum(count)', groupby=['R'])
              .transform_calculate(pct='datum.count / datum.total')
        )
    
        dislike_domain_en = [en for _, en in DISLIKE_ES_EN]
    
        bars_r_dis = (
            base.mark_bar(size=28, stroke=None, strokeWidth=0)
                .encode(
                    x=alt.X("R:O", title="Month", sort="ascending"),
                    y=alt.Y(
                        "count:Q",
                        stack="normalize",
                        axis=alt.Axis(format="%", title="% of answers"),
                        scale=alt.Scale(domain=[0, 1], nice=False, clamp=True)
                    ),
                    color=alt.Color(
                        "aspect_en:N",
                        title="Disliked",
                        scale=alt.Scale(domain=dislike_domain_en),
                        legend=alt.Legend(
                            orient="bottom", direction="horizontal",
                            columns=2, labelLimit=1000, titleLimit=1000, symbolType="square",
                        ),
                    ),
                    tooltip=[
                        # Месяц не показываем в тултипе
                        alt.Tooltip("aspect_en:N", title="Disliked"),
                        alt.Tooltip("count:Q",   title="Answers"),
                        alt.Tooltip("pct:Q",     title="%", format=".0%"),
                        alt.Tooltip("total:Q",   title="All answers"),
                    ],
                )
        ).configure_legend(labelLimit=1000, titleLimit=1000)
    
        st.altair_chart(
            bars_r_dis.properties(title="Dislikes per months", height=460),
            use_container_width=True,
            theme=None
        )
    
    # ---------- Dislike dynamics (по датам A из FR1, текст из F) — как Liked in time ----------
    with st.expander("Dislike in time — show/hide", expanded=False):
        st.subheader("Dislike in time")
    
        # Источник данных и агрегация (у тебя выше уже есть build_aspects_counts_generic)
        df_dislike_src = filter_df(df1, "N", "A", selected_courses, date_range)
        dis_counts, _ = build_aspects_counts_generic(
            df_dislike_src, text_col="F", date_col="A", granularity=granularity,
            aspects_es_en=DISLIKE_ES_EN
        )
    
        if dis_counts.empty:
            st.info("Нет данных для графика Dislike in time (Form Responses 1, колонка F).")
        else:
            # порядок периодов
            bucket_order = (
                dis_counts[["bucket", "bucket_label"]]
                .drop_duplicates().sort_values("bucket")["bucket_label"].tolist()
            )
    
            # высота шкалы Y — как в Liked in time
            totals_by_bucket = (
                dis_counts.groupby("bucket_label", as_index=False)["count"]
                .sum().rename(columns={"count": "total"})
            )
            y_max = max(1, int(totals_by_bucket["total"].max()))
            y_scale_bar = alt.Scale(domain=[0, y_max * 1.1], nice=False, clamp=True)
    
            # легенда: только присутствующие в данных dislike-аспекты
            expected_labels = [f"{es} (EN: {en})" for es, en in DISLIKE_ES_EN]
            present = [lbl for lbl in expected_labels if lbl in dis_counts["aspect"].unique()]
    
            # сами столбики
            bars = (
                alt.Chart(dis_counts)
                  .mark_bar(size=max(40, BAR_SIZE.get(granularity, 36)))
                  .encode(
                      x=alt.X("bucket_label:N", title="Period", sort=bucket_order),
                      y=alt.Y("sum(count):Q", title="Answers", scale=y_scale_bar),
                      color=alt.Color("aspect:N", title="Disliked", sort=present),
                  )
            )
    
            # общий тултип на весь столбик (Top-K строк, как в Liked in time)
            wide = (
                dis_counts
                  .pivot_table(index=["bucket","bucket_label"], columns="aspect_en",
                               values="count", aggfunc="sum", fill_value=0)
            )
            col_order = list(wide.sum(axis=0).sort_values(ascending=False).index)
            TOP_K = 6
    
            def _pack_row_named(r, names=col_order[:TOP_K]):
                total = int(r.sum())
                out = {"total": total}
                for i, name in enumerate(names, start=1):
                    c = int(r.get(name, 0))
                    out[f"t{i}"] = f"{name} — {c} ({c/total:.0%})" if total > 0 and c > 0 else ""
                return pd.Series(out)
    
            packed = wide.apply(_pack_row_named, axis=1).reset_index()
    
            bubble = (
                alt.Chart(packed)
                  .mark_bar(size=max(40, BAR_SIZE.get(granularity, 36)), opacity=0.001)
                  .encode(
                      x=alt.X("bucket_label:N", sort=bucket_order),
                      y=alt.Y("total:Q", scale=y_scale_bar),
                      tooltip=[
                          alt.Tooltip("t1:N", title=""),
                          alt.Tooltip("t2:N", title=""),
                          alt.Tooltip("t3:N", title=""),
                          alt.Tooltip("t4:N", title=""),
                          alt.Tooltip("t5:N", title=""),
                          alt.Tooltip("t6:N", title=""),
                      ],
                  )
            )
    
            st.altair_chart((bars + bubble).properties(height=460),
                            use_container_width=True, theme=None)
    
    # --------- FR2: по урокам (ось X — R) — графики в % по D и E ---------
    st.markdown("---")
    st.subheader("Lesson length")
    
    # Источник — df2_base + фильтр по выбранным месяцам (Q)
    df2_months = df2_base.copy()
    if not df2_months.empty and selected_months:
        df2_months["Q_num"] = pd.to_numeric(df2_months["Q"], errors="coerce")
        df2_months = df2_months[df2_months["Q_num"].isin(selected_months)]
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("**Did the class start in time?**")
        if df2_months.empty:
            st.info("Нет данных для графика по D.")
        else:
            dfD = df2_months.copy()
            dfD["R"] = pd.to_numeric(dfD["Q"], errors="coerce")  # временно переименовали ось
            cnt_by_m_D = build_template_counts_by_R(dfD, text_col="D", templates_es_en=FR2_D_TEMPL_ES_EN)
            if cnt_by_m_D.empty:
                st.info("Нет данных для графика по D.")
            else:
                # --- D ---
                legend_domain_D = [en for _, en in FR2_D_TEMPL_ES_EN]
                base_D = (
                    alt.Chart(cnt_by_m_D)
                      .transform_aggregate(count='sum(count)', groupby=['R', 'templ_en'])
                      .transform_joinaggregate(total='sum(count)', groupby=['R'])
                      .transform_calculate(pct='datum.count / datum.total')
                )
                
                bars_D = (
                    base_D.mark_bar(size=28, stroke=None, strokeWidth=0)
                      .encode(
                          x=alt.X("R:O", title="Month", sort="ascending"),
                          y=alt.Y(
                              "count:Q",
                              stack="normalize",
                              axis=alt.Axis(format="%", title="% of answers"),
                              scale=alt.Scale(domain=[0, 1], nice=False, clamp=True)
                          ),
                          color=alt.Color(
                              "templ_en:N",
                              title="Start time",
                              scale=alt.Scale(domain=legend_domain_D),
                              legend=alt.Legend(
                                  orient="bottom", direction="horizontal",
                                  columns=2, labelLimit=1000, titleLimit=1000, symbolType="square",
                              ),
                          ),
                          # порядок слоёв как в "Disliked throughout the course"
                          order=alt.Order("count:Q", sort="ascending"),
                          tooltip=[
                              # Месяц не показываем
                              alt.Tooltip("templ_en:N", title="Start time"),
                              alt.Tooltip("count:Q",   title="Answers"),
                              alt.Tooltip("pct:Q",     title="%", format=".0%"),
                              alt.Tooltip("total:Q",   title="All answers"),
                          ],
                      )
                ).configure_legend(labelLimit=1000, titleLimit=1000)
                
                st.altair_chart(
                    bars_D.properties(title="Start time — share by months", height=460),
                    use_container_width=True, theme=None
                )
    
    
    with col_right:
        st.markdown("**Was the class the right length?**")
        if df2_months.empty:
            st.info("Нет данных для графика по E.")
        else:
            dfE = df2_months.copy()
            dfE["R"] = pd.to_numeric(dfE["Q"], errors="coerce")  # временно переименовали ось
            cnt_by_m_E = build_template_counts_by_R(dfE, text_col="E", templates_es_en=FR2_E_TEMPL_ES_EN)
            if cnt_by_m_E.empty:
                st.info("Нет данных для графика по E.")
            else:
                # --- E ---
                legend_domain_E = [en for _, en in FR2_E_TEMPL_ES_EN]
                base_E = (
                    alt.Chart(cnt_by_m_E)
                      .transform_aggregate(count='sum(count)', groupby=['R', 'templ_en'])
                      .transform_joinaggregate(total='sum(count)', groupby=['R'])
                      .transform_calculate(pct='datum.count / datum.total')
                )
                
                bars_E = (
                    base_E.mark_bar(size=28, stroke=None, strokeWidth=0)
                      .encode(
                          x=alt.X("R:O", title="Month", sort="ascending"),
                          y=alt.Y(
                              "count:Q",
                              stack="normalize",
                              axis=alt.Axis(format="%", title="% of answers"),
                              scale=alt.Scale(domain=[0, 1], nice=False, clamp=True)
                          ),
                          color=alt.Color(
                              "templ_en:N",
                              title="Class length",
                              scale=alt.Scale(domain=legend_domain_E),
                              legend=alt.Legend(
                                  orient="bottom", direction="horizontal",
                                  columns=2, labelLimit=1000, titleLimit=1000, symbolType="square",
                              ),
                          ),
                          order=alt.Order("count:Q", sort="ascending"),
                          tooltip=[
                              alt.Tooltip("templ_en:N", title="Class length"),
                              alt.Tooltip("count:Q",   title="Answers"),
                              alt.Tooltip("pct:Q",     title="%", format=".0%"),
                              alt.Tooltip("total:Q",   title="All answers"),
                          ],
                      )
                ).configure_legend(labelLimit=1000, titleLimit=1000)
                
                st.altair_chart(
                    bars_E.properties(title="Class length — share by months", height=460),
                    use_container_width=True, theme=None
                )
    
    # ---------- FR2 — Lesson length (D/E) в стиле "Dislike in time" ----------
    with st.expander("Lesson length — show/hide", expanded=False):
        st.subheader("Lesson length")
    
        # Источник: df2_base + единый фильтр по выбранным месяцам (Q)
        df2_text_src = _apply_q_filter(df2_base)
    
        # ===== D: Did the class start in time? =====
        st.markdown("**Did the class start in time?**")
        if df2_text_src.empty or "D" not in df2_text_src.columns:
            st.info("Нет данных (колонка D отсутствует или фильтры пустые).")
        else:
            d_cnt = build_template_counts(
                df2_text_src, text_col="D", date_col="A",
                granularity=granularity, templates_es_en=FR2_D_TEMPL_ES_EN
            )
            if d_cnt.empty:
                st.info("Нет совпадений с шаблонными фразами для D.")
            else:
                # порядок периодов
                d_bucket_order = (
                    d_cnt[["bucket","bucket_label"]]
                    .drop_duplicates()
                    .sort_values("bucket")["bucket_label"].tolist()
                )
    
                # максимум по Y (как в "Liked/Dislike in time")
                d_totals = (
                    d_cnt.groupby("bucket_label", as_index=False)["count"]
                         .sum().rename(columns={"count":"total"})
                )
                y_max_d = max(1, int(d_totals["total"].max()))
                y_scale_d = alt.Scale(domain=[0, y_max_d * 1.1], nice=False, clamp=True)
    
                # столбики: сумма упоминаний по периоду, цвета — по шаблонам
                barsD = (
                    alt.Chart(d_cnt)
                      .mark_bar(size=max(40, BAR_SIZE.get(granularity, 36)))
                      .encode(
                          x=alt.X("bucket_label:N", title="Period", sort=d_bucket_order),
                          y=alt.Y("sum(count):Q", title="Answers", scale=y_scale_d),
                          color=alt.Color(
                              "templ_en:N",
                              title="Start time",
                              # фиксированный порядок легенды по исходному списку:
                              scale=alt.Scale(domain=[en for _, en in FR2_D_TEMPL_ES_EN]),
                          ),
                      )
                )
    
                # общий тултип (TOP-K строк «Категория — N (p%)»)
                wideD = (
                    d_cnt
                    .pivot_table(index=["bucket","bucket_label"], columns="templ_en",
                                 values="count", aggfunc="sum", fill_value=0)
                )
                col_order_D = list(wideD.sum(axis=0).sort_values(ascending=False).index)
                TOP_K = 6
    
                def _pack_row_named_D(r, names=col_order_D[:TOP_K]):
                    total = int(r.sum())
                    out = {"total": total}
                    for i, name in enumerate(names, start=1):
                        c = int(r.get(name, 0))
                        out[f"t{i}"] = f"{name} — {c} ({c/total:.0%})" if total > 0 and c > 0 else ""
                    return pd.Series(out)
    
                packedD = wideD.apply(_pack_row_named_D, axis=1).reset_index()
    
                bubbleD = (
                    alt.Chart(packedD)
                      .mark_bar(size=max(40, BAR_SIZE.get(granularity, 36)), opacity=0.001)
                      .encode(
                          x=alt.X("bucket_label:N", sort=d_bucket_order),
                          y=alt.Y("total:Q", scale=y_scale_d),
                          tooltip=[
                              alt.Tooltip("t1:N", title=""),
                              alt.Tooltip("t2:N", title=""),
                              alt.Tooltip("t3:N", title=""),
                              alt.Tooltip("t4:N", title=""),
                              alt.Tooltip("t5:N", title=""),
                              alt.Tooltip("t6:N", title=""),
                          ],
                      )
                )
    
                st.altair_chart(
                    (barsD + bubbleD).properties(height=420),
                    use_container_width=True, theme=None
                )
    
        # ===== E: Was the class the right length? =====
        st.markdown("**Was the class the right length?**")
        if df2_text_src.empty or "E" not in df2_text_src.columns:
            st.info("Нет данных (колонка E отсутствует или фильтры пустые).")
        else:
            e_cnt = build_template_counts(
                df2_text_src, text_col="E", date_col="A",
                granularity=granularity, templates_es_en=FR2_E_TEMPL_ES_EN
            )
            if e_cnt.empty:
                st.info("Нет совпадений с шаблонными фразами для E.")
            else:
                # порядок периодов
                e_bucket_order = (
                    e_cnt[["bucket","bucket_label"]]
                    .drop_duplicates()
                    .sort_values("bucket")["bucket_label"].tolist()
                )
    
                # максимум по Y
                e_totals = (
                    e_cnt.groupby("bucket_label", as_index=False)["count"]
                         .sum().rename(columns={"count":"total"})
                )
                y_max_e = max(1, int(e_totals["total"].max()))
                y_scale_e = alt.Scale(domain=[0, y_max_e * 1.1], nice=False, clamp=True)
    
                # столбики
                barsE = (
                    alt.Chart(e_cnt)
                      .mark_bar(size=max(40, BAR_SIZE.get(granularity, 36)))
                      .encode(
                          x=alt.X("bucket_label:N", title="Period", sort=e_bucket_order),
                          y=alt.Y("sum(count):Q", title="Answers", scale=y_scale_e),
                          color=alt.Color(
                              "templ_en:N",
                              title="Class length",
                              scale=alt.Scale(domain=[en for _, en in FR2_E_TEMPL_ES_EN]),
                          ),
                      )
                )
    
                # общий тултип
                wideE = (
                    e_cnt
                    .pivot_table(index=["bucket","bucket_label"], columns="templ_en",
                                 values="count", aggfunc="sum", fill_value=0)
                )
                col_order_E = list(wideE.sum(axis=0).sort_values(ascending=False).index)
    
                def _pack_row_named_E(r, names=col_order_E[:TOP_K]):
                    total = int(r.sum())
                    out = {"total": total}
                    for i, name in enumerate(names, start=1):
                        c = int(r.get(name, 0))
                        out[f"t{i}"] = f"{name} — {c} ({c/total:.0%})" if total > 0 and c > 0 else ""
                    return pd.Series(out)
    
                packedE = wideE.apply(_pack_row_named_E, axis=1).reset_index()
    
                bubbleE = (
                    alt.Chart(packedE)
                      .mark_bar(size=max(40, BAR_SIZE.get(granularity, 36)), opacity=0.001)
                      .encode(
                          x=alt.X("bucket_label:N", sort=e_bucket_order),
                          y=alt.Y("total:Q", scale=y_scale_e),
                          tooltip=[
                              alt.Tooltip("t1:N", title=""),
                              alt.Tooltip("t2:N", title=""),
                              alt.Tooltip("t3:N", title=""),
                              alt.Tooltip("t4:N", title=""),
                              alt.Tooltip("t5:N", title=""),
                              alt.Tooltip("t6:N", title=""),
                          ],
                      )
                )
    
                st.altair_chart(
                    (barsE + bubbleE).properties(height=420),
                    use_container_width=True, theme=None
                )
    
    # --------- FR2: три графика "Average by R" по колонкам F, G, H ---------
    st.markdown("---")
    st.subheader("Clear explanations and participation")
    
    # Источник — df2_base + фильтр по выбранным месяцам (Q)
    df2_months_avg = df2_base.copy()
    if not df2_months_avg.empty and selected_months:
        df2_months_avg["Q_num"] = pd.to_numeric(df2_months_avg["Q"], errors="coerce")
        df2_months_avg = df2_months_avg[df2_months_avg["Q_num"].isin(selected_months)]
    
    # временно создаём ось R из Q, чтобы переиспользовать _avg_by_r
    df2_avg_src = df2_months_avg.copy()
    if not df2_avg_src.empty:
        df2_avg_src["R"] = pd.to_numeric(df2_avg_src["Q"], errors="coerce")
    
    def _avg_by_r(df_src: pd.DataFrame, y_col: str) -> pd.DataFrame:
        """Агрегация среднего по колонке y_col с учётом выбранных курсов/дат/уроков (R)."""
        if df_src.empty or not {"A", "M", "R", y_col}.issubset(df_src.columns):
            return pd.DataFrame(columns=["R", "avg_y", "count"])
        d = df_src.copy()
        # применим единый фильтр уроков
        if selected_lessons:
            d["R"] = pd.to_numeric(d["R"], errors="coerce")
            d = d[d["R"].isin(selected_lessons)]
        d = d.dropna(subset=["R", y_col])
        if d.empty:
            return pd.DataFrame(columns=["R", "avg_y", "count"])
        out = (d.groupby("R", as_index=False)
                 .agg(avg_y=(y_col, "mean"),
                      count=(y_col, "size"))
                 .sort_values("R"))
        # приведение типа R к int, чтобы ось была аккуратной
        out["R"] = out["R"].astype(int)
        return out
    
    aggF  = _avg_by_r(df2_avg_src, "F") if ("F" in df2.columns and not df2_avg_src.empty) else pd.DataFrame()
    aggG_2 = _avg_by_r(df2_avg_src, "G") if ("G" in df2.columns and not df2_avg_src.empty) else pd.DataFrame()
    aggH  = _avg_by_r(df2_avg_src, "H") if ("H" in df2.columns and not df2_avg_src.empty) else pd.DataFrame()
    
    colF, colG2, colH = st.columns(3)
    
    def _make_avg_chart(df_avg: pd.DataFrame, title_panel: str):
        if df_avg.empty or len(df_avg) == 0:
            return st.info("Нет данных для выбранных фильтров.")
        y_min = float(df_avg["avg_y"].min())
        y_max = float(df_avg["avg_y"].max())
        pad = (y_max - y_min) * 0.1 if y_max > y_min else 0.5
        y_scale = alt.Scale(domain=[y_min - pad, y_max + pad], nice=False, clamp=True)
        chart = (
            alt.Chart(df_avg)
              .mark_line(point=True)
              .encode(
                  x=alt.X("R:Q", title="Month"),
                  y=alt.Y("avg_y:Q", title="Average score", scale=y_scale),
                  tooltip=[
                      alt.Tooltip("avg_y:Q", title="Average score", format=".2f"),
                      alt.Tooltip("count:Q", title="Answers"),
                  ],
              )
              .properties(height=340, title=title_panel)
        )
        st.altair_chart(chart, use_container_width=True, theme=None)
    
    with colF:
        _make_avg_chart(aggF, "Were the material and explanations clear?")
    
    with colG2:
        _make_avg_chart(aggG_2, "Did the teacher explain calmly and in a way that was easy to follow?")
    
    with colH:
        _make_avg_chart(aggH, "Did you feel you could ask questions and participate in class?")
    
    # ---------- FR2: распределение по месяцам (ось X — Q) — F / G / H (в %), как "Scores — distribution throughout the course"
    st.markdown("---")
    st.subheader("Clear explanations & participation — distribution throughout the course")
    
    # Базовый источник c учётом выбранных месяцев (Q)
    df2_q = df2_base.copy()
    if not df2_q.empty and selected_months:
        df2_q["Q_num"] = pd.to_numeric(df2_q["Q"], errors="coerce")
        df2_q = df2_q[df2_q["Q_num"].isin(selected_months)]
    
    def _allowed_int_vals(d: pd.DataFrame, col: str) -> list[int]:
        if d.empty or col not in d.columns:
            return []
        v = pd.to_numeric(d[col], errors="coerce").dropna().astype(int).unique().tolist()
        return sorted(v)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**Were the material and explanations clear?**")
        if df2_q.empty:
            st.info("No data.")
        else:
            allowedF = _allowed_int_vals(df2_q, "F")
            outF = _build_numeric_counts_by_axis(df2_q, axis_col="Q", val_col="F", allowed_vals=allowedF or None)
            if outF.empty:
                st.info("No data.")
            else:
                chF = _make_percent_stack_by_axis(outF, axis_col="Q", legend_title="Score")
                st.altair_chart(chF.properties(height=460), use_container_width=True, theme=None)
    
    with c2:
        st.markdown("**Did the teacher explain calmly and in a way that was easy to follow?**")
        if df2_q.empty:
            st.info("No data.")
        else:
            allowedG = _allowed_int_vals(df2_q, "G")
            outG = _build_numeric_counts_by_axis(df2_q, axis_col="Q", val_col="G", allowed_vals=allowedG or None)
            if outG.empty:
                st.info("No data.")
            else:
                chG = _make_percent_stack_by_axis(outG, axis_col="Q", legend_title="Score")
                st.altair_chart(chG.properties(height=460), use_container_width=True, theme=None)
    
    with c3:
        st.markdown("**Did you feel you could ask questions and participate in class?**")
        if df2_q.empty:
            st.info("No data.")
        else:
            allowedH = _allowed_int_vals(df2_q, "H")
            outH = _build_numeric_counts_by_axis(df2_q, axis_col="Q", val_col="H", allowed_vals=allowedH or None)
            if outH.empty:
                st.info("No data.")
            else:
                chH = _make_percent_stack_by_axis(outH, axis_col="Q", legend_title="Score")
                st.altair_chart(chH.properties(height=460), use_container_width=True, theme=None)
    
    # ---------- FR2: распределения по F / G / H (по типу "Распределение значений") ----------
    @st.cache_data(show_spinner=False)
    def _prep_df2_numeric_dist(df_src: pd.DataFrame, value_col: str, granularity: str):
        """Готовим df с bucket’ами и списком допустимых значений (инт)."""
        if df_src.empty or value_col not in df_src.columns:
            return pd.DataFrame(), [], [], value_col
        d = df_src.copy()
        d = d.dropna(subset=["A", value_col])
        d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
        d = d.dropna(subset=[value_col])
        if d.empty:
            return pd.DataFrame(), [], [], value_col
    
        d[value_col] = d[value_col].astype(int)              # только целые значения шкалы
        d = add_bucket(d, "A", granularity)                  # buckets
        d = ensure_bucket_and_label(d, "A", granularity)     # подписи
    
        allowed_values = sorted(d[value_col].dropna().unique().tolist())
        if not allowed_values:
            return pd.DataFrame(), [], [], value_col
    
        out, bucket_order, val_order, title = prep_distribution(d, value_col, allowed_values, value_col)
        return out, bucket_order, val_order, title
    
    # источник с учётом фильтра по месяцам (Q)
    df2_numeric_src = _apply_q_filter(df2_base)
    
    def _draw_fr2_dist_block(container, df_src, value_col: str, title: str):
        with container:
            st.markdown(f"**{title}**")
            out, bucket_order, val_order, _ = _prep_df2_numeric_dist(df_src, value_col, granularity)
            if out.empty:
                st.info(f"Нет данных по колонке {value_col} для выбранных фильтров.")
                return
    
            # Базовые бары — как в эталоне (без tooltip)
            bars = (
                alt.Chart(out).mark_bar(size=BAR_SIZE.get(granularity, 36), tooltip=False)
                  .encode(
                      x=alt.X("bucket_label:N", title="Period", sort=bucket_order),
                      y=alt.Y("sum(count):Q", title="Answers"),
                      color=alt.Color("val_str:N", title="Score", sort=val_order),
                      order=alt.Order("val:Q", sort="ascending"),
                  )
                  .properties(height=420)
            )
    
            # Общий оверлей-тултип (как в эталоне)
            _tmp = out.rename(columns={"bucket_label": "X"})
            packed_in = _tmp[["X", "val", "count"]].copy()
            totals = (packed_in.groupby("X", as_index=False)["count"]
                                  .sum().rename(columns={"count": "total"}))
            packed_in = packed_in.merge(totals, on="X", how="left")
            packed_in["val"] = pd.to_numeric(packed_in["val"], errors="coerce")
            packed_in = packed_in.dropna(subset=["val"])
            packed_in["val"] = packed_in["val"].astype(int)
            packed_in["val_str"] = packed_in["val"].astype(str)
    
            packed, tips = _pack_full_tooltip_axis(packed_in, x_col="X", legend_title="Score")
    
            overlay = (
                alt.Chart(packed).mark_bar(size=BAR_SIZE.get(granularity, 36), opacity=0.001)
                  .encode(
                      x=alt.X("X:N", sort=bucket_order),
                      y=alt.Y("total:Q"),
                      tooltip=[
                          *[alt.Tooltip(f"{col}:N", title=title_) for col, title_ in tips],
                          alt.Tooltip("total:Q", title="All answers"),
                      ]
                  )
            )
    
            st.altair_chart(
                (bars + overlay)
                  .properties(
                      height=420,
                      padding={"left": 10, "right": 48, "top": 10, "bottom": 70}
                  )
                  .configure_view(clip=False, stroke=None),
                use_container_width=True, theme=None
            )
    
    # --- всё под тоглом ---
    with st.expander("Clear explanations and participation — show/hide", expanded=False):
        st.subheader("Clear explanations and participation in time")
    
        cF, cG, cH = st.columns(3)
    
        _draw_fr2_dist_block(
            cF, df2_numeric_src, "F",
            "Were the material and explanations clear?"
        )
        _draw_fr2_dist_block(
            cG, df2_numeric_src, "G",
            "Did the teacher explain calmly and in a way that was easy to follow?"
        )
        _draw_fr2_dist_block(
            cH, df2_numeric_src, "H",
            "Did you feel you could ask questions and participate in class?"
        )

# ==================== Refunds (LatAm) — single page (2 charts) ====================

elif section == "Refunds (LatAm)":
    st.subheader("Refunds — LatAm")

    # читаем из кеша
    df_ref = load_refunds_letter_df_cached()
    if df_ref.empty:
        df_ref = pd.DataFrame(columns=["AS", "AU", "K"])

    for c in ["AS", "AU", "K"]:
        if c not in df_ref.columns:
            df_ref[c] = pd.NA

    if df_ref.empty or not {"AS", "AU"}.issubset(df_ref.columns):
        st.info("No data (expected columns: AS — month/date, AU — boolean flag).")
    else:
        # 1) Keep only AU == TRUE
        dff = df_ref.copy()
        dff["AU"] = dff["AU"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        dff = dff[dff["AU"]]
        if dff.empty:
            st.info("No rows with AU=TRUE.")
        else:
            # 2) Month key/label from AS (date or numeric month)
            dt = pd.to_datetime(dff["AS"], errors="coerce")
            as_num = pd.to_numeric(dff["AS"], errors="coerce")

            # numeric sort key: YYYYMM for dates, else integer AS
            month_key = (dt.dt.year * 100 + dt.dt.month).where(dt.notna(), as_num)
            month_key = pd.to_numeric(month_key, errors="coerce")

            month_label = dt.dt.to_period("M").astype(str).where(
                dt.notna(), as_num.astype("Int64").astype(str)
            )

            dff = dff.assign(MonthKey=month_key, Month=month_label)
            dff = dff.dropna(subset=["MonthKey"]).copy()
            dff["MonthKey"] = pd.to_numeric(dff["MonthKey"], errors="coerce")
            dff = dff.dropna(subset=["MonthKey"])

            # 3) Reason column (K) -> fill blanks
            if "K" in dff.columns:
                dff["K"] = dff["K"].astype(str).str.strip()
                dff.loc[dff["K"].eq("") | dff["K"].str.lower().eq("nan"), "K"] = "Unspecified"
            else:
                dff["K"] = "Unspecified"

            # ---------- CHART 1: simple count by month ----------
            st.markdown("**Refunds — by month (count)**")

            by_month = (
                dff.groupby(["MonthKey", "Month"], as_index=False)
                   .size()
                   .rename(columns={"size": "count"})
                   .sort_values("MonthKey")
            )
            month_order = by_month["Month"].tolist()

            ch_counts = (
                alt.Chart(by_month)
                   .mark_bar(size=36)
                   .encode(
                       x=alt.X("Month:N", sort=month_order, title="Month"),
                       y=alt.Y("count:Q", title="Refunds (count)"),
                       tooltip=[
                           alt.Tooltip("Month:N", title="Month"),
                           alt.Tooltip("count:Q", title="Refunds"),
                       ],
                   )
                   .properties(height=420)
            )
            st.altair_chart(ch_counts, use_container_width=True, theme=None)

            # ---------- CHART 2: 100% stacked by reason (distinct colors) ----------
            st.markdown("**Refund reasons — share by month (100%)**")

            grp = (
                dff.groupby(["MonthKey", "Month", "K"], as_index=False)
                   .size()
                   .rename(columns={"size": "count"})
            )
            totals = (
                grp.groupby(["MonthKey", "Month"], as_index=False)["count"]
                   .sum()
                   .rename(columns={"count": "total"})
            )
            out = grp.merge(totals, on=["MonthKey", "Month"], how="left")
            month_order = (
                out[["MonthKey", "Month"]]
                .drop_duplicates()
                .sort_values("MonthKey")["Month"]
                .tolist()
            )

            # фиксируем порядок категорий и даём «разные» цвета (категориальная палитра)
            reason_order = (
                out.groupby("K", as_index=False)["count"]
                   .sum()
                   .sort_values("count", ascending=False)["K"]
                   .tolist()
            )

            ch_stack = (
                alt.Chart(out)
                   .transform_calculate(pct="datum.count / datum.total")
                   .mark_bar(size=36)
                   .encode(
                       x=alt.X("Month:N", sort=month_order, title="Month"),
                       y=alt.Y(
                           "count:Q",
                           stack="normalize",
                           axis=alt.Axis(format="%", title="Share of refunds (100%)"),
                           scale=alt.Scale(domain=[0, 1], nice=False, clamp=True),
                       ),
                       color=alt.Color(
                           "K:N",
                           title="Refund reason",
                           sort=reason_order,
                           # палитра с большим числом различимых цветов
                           scale=alt.Scale(scheme="category20")
                       ),
                       order=alt.Order("count:Q", sort="ascending"),
                       tooltip=[
                           alt.Tooltip("Month:N", title="Month"),
                           alt.Tooltip("K:N", title="Reason"),
                           alt.Tooltip("count:Q", title="Count"),
                           alt.Tooltip("pct:Q", title="Share", format=".0%"),
                           alt.Tooltip("total:Q", title="Total in month"),
                       ],
                   )
                   .properties(height=420)
            )
            st.altair_chart(ch_stack, use_container_width=True, theme=None)

# ==================== QA (analytics) — 3 charts ====================
else:
    st.subheader("QA analytics")

    # Load
    dqa = load_qa_letter_df_cached()
    # Ensure required columns exist
    for c in ["I", "H", "B", "F", "D"]:
        if c not in dqa.columns:
            dqa[c] = pd.NA

    if dqa.empty:
        st.info("No data in 'QA for analytics' tab.")
    else:
        # Types
        dqa["B"] = pd.to_datetime(dqa["B"], errors="coerce")   # Lesson date
        dqa["H"] = pd.to_numeric(dqa["H"], errors="coerce")    # Month
        dqa["F"] = pd.to_numeric(dqa["F"], errors="coerce")    # QA marker (avg)
        dqa = dqa.dropna(subset=["I"])                         # Course present

        # Apply global filters: courses + date
        if selected_courses:
            dqa = dqa[dqa["I"].isin(selected_courses)]
        # date_range can be [] or [start, end]
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_dt = pd.to_datetime(date_range[0])
            end_dt   = pd.to_datetime(date_range[1])
            dqa = dqa[(dqa["B"] >= start_dt) & (dqa["B"] <= end_dt)]
        elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
            only_dt = pd.to_datetime(date_range[0])
            dqa = dqa[dqa["B"].dt.date == only_dt.date()]

        # Apply global month filter (if any)
        if selected_months:
            dqa = dqa[dqa["H"].astype("Int64").isin(pd.Series(selected_months, dtype="Int64"))]

        # ---------- Chart 1: Average F by Month (H) ----------
        st.markdown("**Average QA marker (F) by Month**")

        df_avg = dqa.dropna(subset=["H", "F"]).copy()
        if df_avg.empty:
            st.info("No data for average (F) by Month.")
        else:
            df_avg["H"] = df_avg["H"].astype(int)
            agg = (df_avg.groupby("H", as_index=False)
                         .agg(avg_F=("F", "mean"), count=("F", "size"))
                         .sort_values("H"))
            y_min = float(agg["avg_F"].min())
            y_max = float(agg["avg_F"].max())
            pad = (y_max - y_min) * 0.1 if y_max > y_min else 0.5
            y_scale = alt.Scale(domain=[y_min - pad, y_max + pad], nice=False, clamp=True)

            ch1 = (
                alt.Chart(agg)
                  .mark_line(point=True)
                  .encode(
                      x=alt.X("H:O", title="Month", sort="ascending"),
                      y=alt.Y("avg_F:Q", title="Average QA marker (F)", scale=y_scale),
                      tooltip=[
                          alt.Tooltip("H:O", title="Month"),
                          alt.Tooltip("avg_F:Q", title="Average F", format=".2f"),
                          alt.Tooltip("count:Q", title="Answers"),
                      ],
                  )
                  .properties(height=360)
            )
            st.altair_chart(ch1, use_container_width=True, theme=None)

        st.markdown("---")

        # === QA (LatAm) — D text distribution by Month (H) ===
        st.markdown("**QA marker (D) — distribution by Month (100%)**")
        
        # Гарантируем нужные колонки
        need = ["H", "D", "I", "B"]   # Month, QA marker (text), Course, Lesson date
        for c in need:
            if c not in dqa.columns:
                dqa[c] = pd.NA
        
        src = dqa.copy()
        
        # Месяц только из H:
        src["H_raw"] = src["H"].astype(str).str.strip()
        h_num = pd.to_numeric(src["H_raw"], errors="coerce")
        src["MonthNum"] = h_num.astype("Int64")
        # Лейбл месяца: если H — число -> используем его, иначе оставляем исходную строку
        src["Month"] = np.where(h_num.notna(), h_num.astype("Int64").astype(str), src["H_raw"])
        
        # Текст категории из D:
        src["D_txt"] = src["D"].astype(str).fillna("").str.strip().replace({"nan": ""})
        
        # Фильтр по курсам (I), если выбран
        if selected_courses:
            src = src[src["I"].astype(str).isin(selected_courses)]
        
        # Фильтр по месяцам — только когда MonthNum числовой
        if selected_months and src["MonthNum"].notna().any():
            src = src[src["MonthNum"].isin(pd.Series(selected_months, dtype="Int64"))]
        
        # Чистим пустые
        src = src[(src["Month"].astype(str).str.len() > 0) & (src["D_txt"] != "")]
        if src.empty:
            st.info("No data for distribution (text D) by Month.")
        else:
            # Группировка и totals
            grp = (
                src.groupby(["Month", "D_txt"], as_index=False)
                   .size().rename(columns={"size": "count"})
            )
            totals = grp.groupby("Month", as_index=False)["count"] \
                        .sum().rename(columns={"count": "total"})
            out = grp.merge(totals, on="Month", how="left")
        
            # Порядок месяцев: по числовому MonthNum, если он есть; иначе — алфавит
            if src["MonthNum"].notna().any():
                month_order = (
                    src[["Month", "MonthNum"]]
                      .dropna(subset=["MonthNum"])
                      .drop_duplicates()
                      .sort_values("MonthNum")["Month"].tolist()
                )
            else:
                month_order = sorted(out["Month"].unique().tolist(), key=lambda x: str(x))
        
            # Порядок категорий — по общей частоте
            cat_order = (
                out.groupby("D_txt", as_index=False)["count"]
                   .sum().sort_values("count", ascending=False)["D_txt"].tolist()
            )
        
            ch = (
                alt.Chart(out)
                  .mark_bar(size=28, stroke=None, strokeWidth=0)
                  .encode(
                      x=alt.X("Month:N", title="Month", sort=month_order),
                      y=alt.Y(
                          "count:Q",
                          stack="normalize",
                          axis=alt.Axis(format="%", title="Share (100%)"),
                          scale=alt.Scale(domain=[0, 1], nice=False, clamp=True),
                      ),
                      color=alt.Color(
                          "D_txt:N",
                          title="QA marker (D)",
                          sort=cat_order,
                          scale=alt.Scale(scheme="category20")  # разные цвета
                      ),
                      order=alt.Order("count:Q", sort="ascending"),
                      tooltip=[
                          alt.Tooltip("Month:N", title="Month"),
                          alt.Tooltip("D_txt:N", title="Text"),
                          alt.Tooltip("count:Q", title="Count"),
                          alt.Tooltip("total:Q", title="Total in month"),
                      ],
                  )
                  .properties(height=420)
            ).configure_legend(labelLimit=1200, titleLimit=1200, symbolType="square")
        
            st.altair_chart(ch, use_container_width=True, theme=None)



        # ---------- Chart 3: Average F over time (by lesson date B, bucketed; line) ----------
        st.markdown("**Average QA marker (F) over time**")

        df_time = dqa.dropna(subset=["B", "F"]).copy()
        if df_time.empty:
            st.info("No data for average (F) over time.")
        else:
            # Bucket by global granularity ("Day", "Week", "Month", "Year") using helpers
            df_time = add_bucket(df_time, "B", granularity)
            df_time = ensure_bucket_and_label(df_time, "B", granularity)

            agg_t = (df_time.groupby(["bucket", "bucket_label"], as_index=False)
                              .agg(avg_F=("F", "mean"), count=("F", "size"))
                              .sort_values("bucket"))
            bucket_order = agg_t["bucket_label"].tolist()

            y_min = float(agg_t["avg_F"].min())
            y_max = float(agg_t["avg_F"].max())
            pad = (y_max - y_min) * 0.1 if y_max > y_min else 0.5
            y_scale = alt.Scale(domain=[y_min - pad, y_max + pad], nice=False, clamp=True)

            ch3 = (
                alt.Chart(agg_t)
                  .mark_line(point=True)
                  .encode(
                      x=alt.X("bucket_label:N", title="Period", sort=bucket_order),
                      y=alt.Y("avg_F:Q", title="Average QA marker (F)", scale=y_scale),
                      tooltip=[
                          alt.Tooltip("bucket_label:N", title="Period"),
                          alt.Tooltip("avg_F:Q", title="Average F", format=".2f"),
                          alt.Tooltip("count:Q", title="Answers"),
                      ],
                  )
                  .properties(height=380)
            )
            st.altair_chart(ch3, use_container_width=True, theme=None)

