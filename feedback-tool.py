import streamlit as st
import pandas as pd
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

st.set_page_config(layout="wide", page_title="40 week courses")

SPREADSHEET_ID = "1fR8_Ay7jpzmPCAl6dWSCC7sWw5VJOaNpu5Zp8b78LRg"

# ---- Авторизация через st.secrets (строка JSON) ----
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]
sa_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT"])
creds = ServiceAccountCredentials.from_json_keyfile_dict(sa_info, scope)
client = gspread.authorize(creds)

# ==================== УТИЛИТЫ ====================

def load_sheet_as_letter_df(sheet_name: str) -> pd.DataFrame:
    ws = client.open_by_key(SPREADSHEET_ID).worksheet(sheet_name)
    values = ws.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame()
    num_cols = len(values[0])
    letters = list(string.ascii_uppercase)[:num_cols]  # A..Z
    return pd.DataFrame(values[1:], columns=letters)

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
    dff = filter_df(df, course_col, date_col, selected_courses, date_range)
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

def build_template_counts(
    df: pd.DataFrame,
    text_col: str,
    date_col: str,
    granularity: str,
    templates_es_en: list[tuple[str, str]],
):
    need_cols = [date_col, text_col]
    if df.empty or not all(c in df.columns for c in need_cols):
        return pd.DataFrame(columns=["bucket","bucket_label","templ_es","templ_en","count"])

    d = df[need_cols].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    if d.empty:
        return pd.DataFrame(columns=["bucket","bucket_label","templ_es","templ_en","count"])

    tmpl_norm = [(_norm_local(es), es, en) for es, en in templates_es_en]

    d = d.rename(columns={date_col: "A", text_col: "TXT"})
    d = add_bucket(d, "A", granularity)
    d = ensure_bucket_and_label(d, "A", granularity)

    rows = []
    # ⚠️ запятую НЕ используем как разделитель, чтобы не ломать "Sí, todo a tiempo"
    splitter = re.compile(r"[;\/\n|]+")  # ; / | и перенос строки
    for _, r in d.iterrows():
        raw = str(r["TXT"] or "").strip()
        if not raw:
            continue
        parts = splitter.split(raw) if splitter.search(raw) else [raw]
        for p in parts:
            t = _norm_local(p.strip())
            if not t:
                continue
            for es_norm, es, en in tmpl_norm:
                if t == es_norm or es_norm in t:
                    rows.append((r["bucket"], r["bucket_label"], es, en))
                    break

    if not rows:
        return pd.DataFrame(columns=["bucket","bucket_label","templ_es","templ_en","count"])

    out = (pd.DataFrame(rows, columns=["bucket","bucket_label","templ_es","templ_en"])
             .groupby(["bucket","bucket_label","templ_es","templ_en"], as_index=False)
             .size().rename(columns={"size":"count"}))
    return out

def build_aspects_counts(df: pd.DataFrame, text_col: str, date_col: str, granularity: str):
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

    rows, unknown = [], []
    for _, r in d.iterrows():
        txt = str(r["TXT"]).strip()
        if not txt:
            continue
        parts = re.split(r"[;,/\n|]+", txt) if re.search(r"[;,/\n|]", txt) else [txt]
        for p in parts:
            t = _norm(p.strip())
            if not t:
                continue
            matched = False
            for es_norm, es, en in _ASPECTS_NORM:
                if t == es_norm or es_norm in t:
                    rows.append((r["bucket"], r["bucket_label"], f"{es} (EN: {en})", 1))
                    matched = True
                    break
            if not matched:
                unknown.append(p.strip())

    counts = pd.DataFrame(rows, columns=["bucket","bucket_label","aspect","count"])
    if not counts.empty:
        counts = counts.groupby(["bucket","bucket_label","aspect"], as_index=False)["count"].sum()

    unknown_df = pd.DataFrame(unknown, columns=["mention"])
    if not unknown_df.empty:
        unknown_df["mention"] = unknown_df["mention"].str.strip()
        unknown_df = (unknown_df.groupby("mention", as_index=False)
                      .size().rename(columns={"size": "total"})
                      .sort_values("total", ascending=False))
        unknown_df["en"] = unknown_df["mention"].apply(translate_es_to_en)
        unknown_df = unknown_df[["en", "mention", "total"]]
    else:
        unknown_df = pd.DataFrame(columns=["en", "mention", "total"])

    return counts, unknown_df

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

def aspect_to_en_label(s: str) -> str:
    """Из 'ES (EN: EN)' достаём 'EN'."""
    m = re.search(r"\(EN:\s*(.*?)\)\s*$", str(s))
    return m.group(1).strip() if m else str(s)

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


# ==================== ДАННЫЕ ====================

df1 = load_sheet_as_letter_df("Form Responses 1")   # A=date, N=course, S=x, G=y, E=aspects, F=dislike, H=comments
df2 = load_sheet_as_letter_df("Form Responses 2")   # A=date, M=course, R=x, I=y

# Приведение типов
for df, date_col, x_col, y_col in [
    (df1, "A", "S", "G"),
    (df2, "A", "R", "I"),
]:
    if df.empty:
        continue
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

# ДОПОЛНИТЕЛЬНО: FR2 — привести F, G, H к числу (если они шкальные)
if not df2.empty:
    for col in ["F", "G", "H"]:
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors="coerce")

# ==================== ЕДИНЫЕ ФИЛЬТРЫ ====================

st.sidebar.header("Фильтры")

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
granularity = st.sidebar.selectbox("Гранулярность для распределения", ["День", "Неделя", "Месяц", "Год"])
BAR_SIZE = {"День": 18, "Неделя": 44, "Месяц": 56, "Год": 64}
bar_size = BAR_SIZE.get(granularity, 36)

# ---- БАЗОВЫЕ ФИЛЬТРЫ ПО КУРСАМ/ДАТЕ ДЛЯ ОБОИХ ЛИСТОВ ----
df1_base = filter_df(df1, "N", "A", selected_courses, date_range)
df2_base = filter_df(df2, "M", "A", selected_courses, date_range)

# ---- ЕДИНЫЙ ФИЛЬТР УРОКОВ S/R (union значений S из FR1 и R из FR2) ----
if not df1_base.empty and "S" in df1_base.columns:
    s_vals = pd.to_numeric(df1_base["S"], errors="coerce").dropna().astype(int).unique().tolist()
else:
    s_vals = []
if not df2_base.empty and "R" in df2_base.columns:
    r_vals = pd.to_numeric(df2_base["R"], errors="coerce").dropna().astype(int).unique().tolist()
else:
    r_vals = []
lessons_options = sorted(list(set(s_vals + r_vals)))

if "s_selected" not in st.session_state:
    st.session_state["s_selected"] = lessons_options.copy()

sb1, sb2 = st.sidebar.columns(2)
if sb1.button("All S/R"):
    st.session_state["s_selected"] = lessons_options.copy()
    st.rerun()
if sb2.button("Clear S/R"):
    st.session_state["s_selected"] = []
    st.rerun()

selected_lessons = st.sidebar.multiselect(
    "Уроки (S/R)",
    options=lessons_options,
    default=st.session_state["s_selected"],
    key="s_selected",
    help="Фильтр единый для S (FR1) и R (FR2)"
)

# ==================== ПРИМЕНЕНИЕ ФИЛЬТРОВ ====================

# Верхние графики (средние)
agg1 = apply_filters_and_aggregate(df1, "N", "A", "S", "G", selected_courses, date_range)
if not agg1.empty and selected_lessons:
    agg1["S"] = pd.to_numeric(agg1["S"], errors="coerce").astype("Int64")
    agg1 = agg1[agg1["S"].isin(selected_lessons)]
    agg1["S"] = agg1["S"].astype(int)

agg2 = apply_filters_and_aggregate(df2, "M", "A", "R", "I", selected_courses, date_range)
if not agg2.empty and selected_lessons:
    agg2["R"] = pd.to_numeric(agg2["R"], errors="coerce").astype("Int64")
    agg2 = agg2[agg2["R"].isin(selected_lessons)]
    agg2["R"] = agg2["R"].astype(int)

# Нижние распределения (сырые) + S/R-фильтр
df1_f = df1_base.copy()
if not df1_f.empty and selected_lessons:
    df1_f["S_num"] = pd.to_numeric(df1_f["S"], errors="coerce")
    df1_f = df1_f[df1_f["S_num"].isin(selected_lessons)]

df2_f = df2_base.copy()
if not df2_f.empty and selected_lessons:
    df2_f["R_num"] = pd.to_numeric(df2_f["R"], errors="coerce")
    df2_f = df2_f[df2_f["R_num"].isin(selected_lessons)]

if not df1_f.empty:
    df1_f = df1_f.dropna(subset=["A", "G"])
    df1_f = add_bucket(df1_f, "A", granularity)
    df1_f = ensure_bucket_and_label(df1_f, "A", granularity)

if not df2_f.empty:
    df2_f = df2_f.dropna(subset=["A", "I"])
    df2_f = add_bucket(df2_f, "A", granularity)
    df2_f = ensure_bucket_and_label(df2_f, "A", granularity)

fr1_out, fr1_bucket_order, fr1_val_order, fr1_title = prep_distribution(df1_f, "G", [1,2,3,4,5], "G")
fr2_out, fr2_bucket_order, fr2_val_order, fr2_title = prep_distribution(df2_f, "I", list(range(1,11)), "I")

# ==================== ОТРИСОВКА ====================

st.title("40 week courses")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Form Responses 1 — Average by S")
    if agg1.empty:
        st.info("Нет данных для выбранных фильтров.")
    else:
        y_min = float(agg1["avg_y"].min()) if len(agg1) else 0.0
        y_max = float(agg1["avg_y"].max()) if len(agg1) else 5.0
        pad = (y_max - y_min) * 0.1 if y_max > y_min else 0.5
        y_scale = alt.Scale(domain=[y_min - pad, y_max + pad], nice=False, clamp=True)

        chart1 = (
            alt.Chart(agg1).mark_line(point=True)
              .encode(
                  x=alt.X("S:Q", title="S"),
                  y=alt.Y("avg_y:Q", title="Average G", scale=y_scale),
                  tooltip=[
                      alt.Tooltip("S:Q", title="S"),
                      alt.Tooltip("avg_y:Q", title="Average G", format=".2f"),
                      alt.Tooltip("count:Q", title="Кол-во ответов")
                  ])
              .properties(height=380)
        )
        st.altair_chart(chart1, use_container_width=True, theme=None)

with col2:
    st.subheader("Form Responses 2 — Average by R")
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
                  x=alt.X("R:Q", title="R"),
                  y=alt.Y("avg_y:Q", title="Average I", scale=y_scale2),
                  tooltip=[
                      alt.Tooltip("R:Q", title="R"),
                      alt.Tooltip("avg_y:Q", title="Average I", format=".2f"),
                      alt.Tooltip("count:Q", title="Кол-во ответов")
                  ])
              .properties(height=380)
        )
        st.altair_chart(chart2, use_container_width=True, theme=None)

st.markdown("---")
st.subheader(f"Распределение значений (гранулярность: {granularity.lower()})")

col3, col4 = st.columns([1, 1])

with col3:
    st.markdown("**Form Responses 1 — распределение G (1–5)**")
    if fr1_out.empty:
        st.info("Нет данных (FR1).")
    else:
        bars1 = (
            alt.Chart(fr1_out).mark_bar(size=BAR_SIZE.get(granularity, 36))
              .encode(
                  x=alt.X("bucket_label:N", title="Период", sort=fr1_bucket_order),
                  y=alt.Y("sum(count):Q", title="Кол-во ответов"),
                  color=alt.Color("val_str:N", title=fr1_title, sort=fr1_val_order),
                  order=alt.Order("val:Q", sort="ascending"),
                  tooltip=[
                      alt.Tooltip("bucket_label:N", title="Период"),
                      alt.Tooltip("val_str:N", title=fr1_title),
                      alt.Tooltip("count:Q", title="Кол-во"),
                      alt.Tooltip("pct:Q", title="% внутри периода", format=".0%")
                  ])
              .properties(height=420)
        )
        st.altair_chart(bars1, use_container_width=True, theme=None)

with col4:
    st.markdown("**Form Responses 2 — распределение I (1–10)**")
    if fr2_out.empty:
        st.info("Нет данных (FR2).")
    else:
        bars2 = (
            alt.Chart(fr2_out).mark_bar(size=BAR_SIZE.get(granularity, 36))
              .encode(
                  x=alt.X("bucket_label:N", title="Период", sort=fr2_bucket_order),
                  y=alt.Y("sum(count):Q", title="Кол-во ответов"),
                  color=alt.Color("val_str:N", title=fr2_title, sort=fr2_val_order),
                  order=alt.Order("val:Q", sort="ascending"),
                  tooltip=[
                      alt.Tooltip("bucket_label:N", title="Период"),
                      alt.Tooltip("val_str:N", title=fr2_title),
                      alt.Tooltip("count:Q", title="Кол-во"),
                      alt.Tooltip("pct:Q", title="% внутри периода", format=".0%")
                  ])
              .properties(height=420)
        )
        st.altair_chart(bars2, use_container_width=True, theme=None)

# ---------- РАСПРЕДЕЛЕНИЕ ПО УРОКАМ (в %) ДЛЯ ТЕХ ЖЕ ШКАЛ ----------
st.markdown("---")
st.subheader("Распределение по урокам (в %) — те же шкалы G (FR1) и I (FR2)")

def _build_numeric_counts_by_axis(df_src: pd.DataFrame, axis_col: str, val_col: str, allowed_vals: list[int] | None):
    """
    Готовит счётчики по урокам (ось = axis_col: 'S' или 'R').
    Возвращает: [axis_col, val, val_str, count, total]
    """
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

def _make_percent_stack_by_axis(out_df: pd.DataFrame, axis_col: str, legend_title: str):
    """
    Рисует нормированный стек 0–100% по оси axis_col (S/R).
    """
    if out_df.empty:
        return None

    axis_order = sorted(out_df[axis_col].unique().tolist())
    val_order  = sorted(out_df["val"].unique().tolist())
    val_order_str = [str(v) for v in val_order]

    base = alt.Chart(out_df).transform_calculate(pct='datum.count / datum.total')

    chart = (
        base.mark_bar(size=28, stroke=None, strokeWidth=0)
            .encode(
                x=alt.X(f"{axis_col}:O", title=axis_col, sort=axis_order),
                y=alt.Y(
                    "count:Q",
                    stack="normalize",
                    axis=alt.Axis(format="%", title="% от ответов"),
                    scale=alt.Scale(domain=[0, 1], nice=False, clamp=True),
                ),
                color=alt.Color(
                    "val_str:N",
                    title=legend_title,
                    sort=val_order_str,  # порядок в легенде (оставляем числовой)
                    legend=alt.Legend(
                        orient="bottom",
                        direction="horizontal",
                        columns=5,
                        labelLimit=1000,
                        titleLimit=1000,
                        symbolType="square",
                    ),
                ),
                order=alt.Order("val:Q", sort="descending"),  # 👈 ключевая строка
                tooltip=[
                    alt.Tooltip(f"{axis_col}:O", title=("Урок (R)" if axis_col=="R" else "Урок (S)")),
                    alt.Tooltip("val_str:N", title=legend_title),
                    alt.Tooltip("count:Q",  title="Кол-во"),
                    alt.Tooltip("pct:Q",    title="Доля", format=".0%"),
                    alt.Tooltip("total:Q",  title="Всего по уроку"),
                ],
            )
    ).configure_legend(labelLimit=1000, titleLimit=1000)

    return chart

# Источники с учётом текущих фильтров курсов/дат/уроков
# ЛЕВЫЙ: FR2 (I по R)
df2_lessons_I = df2_base.copy()
if not df2_lessons_I.empty and selected_lessons:
    df2_lessons_I["R_num"] = pd.to_numeric(df2_lessons_I["R"], errors="coerce")
    df2_lessons_I = df2_lessons_I[df2_lessons_I["R_num"].isin(selected_lessons)]

# ПРАВЫЙ: FR1 (G по S)
df1_lessons_G = df1_base.copy()
if not df1_lessons_G.empty and selected_lessons:
    df1_lessons_G["S_num"] = pd.to_numeric(df1_lessons_G["S"], errors="coerce")
    df1_lessons_G = df1_lessons_G[df1_lessons_G["S_num"].isin(selected_lessons)]

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("**FR1 — по урокам (S) — G (в %)**")
    out_G_S = _build_numeric_counts_by_axis(df1_lessons_G, axis_col="S", val_col="G", allowed_vals=[1,2,3,4,5])
    if out_G_S.empty:
        st.info("Нет данных по G (FR1) для выбранных фильтров.")
    else:
        ch_G_S = _make_percent_stack_by_axis(out_G_S, axis_col="S", legend_title="G")
        st.altair_chart(ch_G_S.properties(height=460), use_container_width=True, theme=None)

with right_col:
    st.markdown("**FR2 — по урокам (R) — I (в %)**")
    out_I_R = _build_numeric_counts_by_axis(df2_lessons_I, axis_col="R", val_col="I", allowed_vals=list(range(1, 11)))
    if out_I_R.empty:
        st.info("Нет данных по I (FR2) для выбранных фильтров.")
    else:
        ch_I_R = _make_percent_stack_by_axis(out_I_R, axis_col="R", legend_title="I")
        st.altair_chart(ch_I_R.properties(height=460), use_container_width=True, theme=None)


# ---------- НИЖЕ: Аспекты урока — Form Responses 1 ----------
st.markdown("---")
st.subheader("Аспекты урока — Form Responses 1")

df_aspects = df1_base.copy()
if not df_aspects.empty and selected_lessons:
    df_aspects["S_num"] = pd.to_numeric(df_aspects["S"], errors="coerce")
    df_aspects = df_aspects[df_aspects["S_num"].isin(selected_lessons)]

asp_counts, _unknown_all = build_aspects_counts(df_aspects, text_col="E", date_col="A", granularity=granularity)

# ---- График: «Аспекты по датам (ось X — A)» с быстрым общим тултипом ----
st.markdown("**Аспекты по датам (ось X — A)**")
if asp_counts.empty:
    st.info("Не нашёл упоминаний аспектов (лист 'Form Responses 1', колонка E).")
else:
    asp_counts["aspect_en"] = asp_counts["aspect"].apply(aspect_to_en_label)

    bucket_order = (asp_counts[["bucket","bucket_label"]]
                    .drop_duplicates()
                    .sort_values("bucket")["bucket_label"].tolist())

    totals_by_bucket = (asp_counts.groupby("bucket_label", as_index=False)["count"]
                                  .sum().rename(columns={"count":"total"}))
    y_max = max(1, int(totals_by_bucket["total"].max()))
    y_scale_bar = alt.Scale(domain=[0, y_max * 1.1], nice=False, clamp=True)

    present = (asp_counts["aspect"].unique().tolist())

    bars = (
        alt.Chart(asp_counts).mark_bar(size=max(40, BAR_SIZE.get(granularity, 36)))
          .encode(
              x=alt.X("bucket_label:N", title="Период (по A)", sort=bucket_order),
              y=alt.Y("sum(count):Q", title="Кол-во упоминаний", scale=y_scale_bar),
              color=alt.Color("aspect:N", title="Аспект", sort=present)
          )
    )

    # ---------- быстрый «общий тултип» ----------
    wide = (
        asp_counts
        .pivot_table(index=["bucket","bucket_label"], columns="aspect_en",
                     values="count", aggfunc="sum", fill_value=0)
    )

    col_order = list(wide.sum(axis=0).sort_values(ascending=False).index)

    def _pack_row(r, top_k=6):
        total = int(r[col_order].sum())
        if total == 0:
            return pd.Series({"total": 0, "tooltip_text": ""})
        pairs = [(name, int(r[name])) for name in col_order if r[name] > 0]
        pairs.sort(key=lambda x: x[1], reverse=True)
        pairs = pairs[:top_k]
        lines = [f"{name} — {c} ({c/total:.0%})" for name, c in pairs]
        return pd.Series({"total": total, "tooltip_text": "\n".join(lines)})

    packed = wide.apply(_pack_row, axis=1).reset_index()

    bubble = (
        alt.Chart(packed)
          .mark_bar(size=max(40, BAR_SIZE.get(granularity, 36)), opacity=0.001)
          .encode(
              x=alt.X("bucket_label:N", sort=bucket_order),
              y=alt.Y("total:Q", scale=y_scale_bar),
              tooltip=[
                  alt.Tooltip("bucket_label:N", title="Период"),
                  alt.Tooltip("total:Q", title="Всего упоминаний"),
                  alt.Tooltip("tooltip_text:N", title=""),
              ]
          )
    )

    st.altair_chart((bars + bubble).properties(height=460),
                    theme=None, use_container_width=True)

# --------- ГРАФИК «Распределение по урокам (ось X — S)» В % ---------
st.markdown("---")
st.subheader("Распределение по урокам (ось X — S) — график (в %)")

cnt_by_s_all = build_aspects_counts_by_S(df_aspects)
if cnt_by_s_all.empty:
    st.info("Нет данных для графика по урокам.")
else:
    base = (
        alt.Chart(cnt_by_s_all)
          .transform_aggregate(count='sum(count)', groupby=['S', 'aspect_en'])
          .transform_joinaggregate(total='sum(count)', groupby=['S'])
          .transform_calculate(pct='datum.count / datum.total')
    )

    legend_domain_en = [en for _, en in ASPECTS_ES_EN]
    
    bars_s = (
        base.mark_bar(size=28, stroke=None, strokeWidth=0)
            .encode(
                x=alt.X("S:O", title="S", sort="ascending"),
                y=alt.Y(
                    "count:Q",
                    stack="normalize",
                    axis=alt.Axis(format="%", title="% от упоминаний"),
                    scale=alt.Scale(domain=[0, 1], nice=False, clamp=True)
                ),
                color=alt.Color(
                    "aspect_en:N",
                    title="Аспект (EN)",
                    scale=alt.Scale(domain=legend_domain_en),
                    legend=alt.Legend(
                        orient="bottom",
                        direction="horizontal",
                        columns=2,
                        labelLimit=1000,
                        titleLimit=1000,
                        symbolType="square",
                    ),
                ),
                tooltip=[
                    alt.Tooltip("S:O", title="Урок"),
                    alt.Tooltip("aspect_en:N", title="Аспект"),
                    alt.Tooltip("count:Q", title="Кол-во"),
                    alt.Tooltip("pct:Q", title="Доля", format=".0%"),
                    alt.Tooltip("total:Q", title="Всего по уроку"),
                ],
            )
    ).configure_legend(labelLimit=1000, titleLimit=1000)
    
    st.altair_chart(bars_s.properties(height=460), use_container_width=True, theme=None)

# --------- ТАБЛИЦА ВНИЗУ ---------
st.markdown("---")
st.subheader("Распределение по урокам (ось X — S) — таблица")

rows_aspects = []
unknown_per_s = {}

if not df_aspects.empty and {"S","E"}.issubset(df_aspects.columns):
    df_tmp = df_aspects[["S","E"]].copy()
    df_tmp["S"] = pd.to_numeric(df_tmp["S"], errors="coerce")
    df_tmp = df_tmp.dropna(subset=["S"])
    df_tmp["S"] = df_tmp["S"].astype(int)

    for _, rr in df_tmp.iterrows():
        s_val = int(rr["S"])
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
                    rows_aspects.append((s_val, en))
                    matched = True
                    break
            if not matched:
                unknown_per_s.setdefault(s_val, Counter())[p_clean] += 1

if not rows_aspects and not unknown_per_s:
    st.info("Нет данных для выбранных фильтров.")
else:
    df_as = (pd.DataFrame(rows_aspects, columns=["S","aspect_en"])
             if rows_aspects else pd.DataFrame(columns=["S","aspect_en"]))
    lesson_rows = []
    lessons_sorted = sorted(set(list(df_as["S"].unique()) + list(unknown_per_s.keys())))

    for s in lessons_sorted:
        if not df_as.empty and s in df_as["S"].values:
            cnt = (df_as[df_as["S"] == s]["aspect_en"]
                   .value_counts()
                   .sort_values(ascending=False))
            total_tpl = int(cnt.sum())
            parts_text = []
            for en_name, c in cnt.items():
                pct = (c / total_tpl) if total_tpl else 0.0
                parts_text.append(f"• {en_name} — {c} ({pct:.0%})")
            aspects_text = "\n".join(parts_text)
        else:
            aspects_text = ""
            total_tpl = 0

        unk_counter = unknown_per_s.get(s, Counter())
        if unk_counter:
            top_items = unk_counter.most_common(10)
            rest = sum(unk_counter.values()) - sum(c for _, c in top_items)
            unk_parts = [f"• {translate_es_to_en(m)} ({c})" for m, c in top_items]
            if rest > 0:
                unk_parts.append(f"• … (+{rest})")
            unknown_text = "\n".join(unk_parts)
        else:
            unknown_text = ""

        lesson_rows.append({
            "S": int(s),
            "Aspects (EN)": aspects_text,
            "Unknown (EN)": unknown_text,
            "Total (templ.)": total_tpl
        })

    table = pd.DataFrame(lesson_rows).sort_values("S").reset_index(drop=True)
    height = min(800, 100 + 28 * len(table))
    st.dataframe(
        table[["S","Aspects (EN)","Unknown (EN)","Total (templ.)"]],
        use_container_width=True,
        height=height
    )

# Подсказка, если онлайн-переводчик недоступен
if _gt is None:
    st.caption("⚠️ deep-translator недоступен — используется упрощённый пословный перевод.")

# ---------- Dislike dynamics (по датам A, текст из F; лист FR1) ----------
st.markdown("---")
st.subheader("Dislike dynamics (по датам A из FR1, текст из F)")

df_dislike_src = filter_df(df1, "N", "A", selected_courses, date_range)
dis_counts, _ = build_aspects_counts_generic(
    df_dislike_src, text_col="F", date_col="A", granularity=granularity,
    aspects_es_en=DISLIKE_ES_EN
)

if dis_counts.empty:
    st.info("Нет данных для графика Dislike dynamics (Form Responses 1, колонка F).")
else:
    bucket_order = (dis_counts[["bucket","bucket_label"]]
                    .drop_duplicates().sort_values("bucket")["bucket_label"].tolist())

    expected_labels = [f"{es} (EN: {en})" for es, en in DISLIKE_ES_EN]
    present = [lbl for lbl in expected_labels if lbl in dis_counts["aspect"].unique()]

    bars_dis = (
        alt.Chart(dis_counts)
          .mark_bar(size=max(40, bar_size))
          .encode(
              x=alt.X("bucket_label:N", title="Период (по A)", sort=bucket_order),
              y=alt.Y("sum(count):Q", title="Кол-во упоминаний"),
              color=alt.Color("aspect:N", title="Аспект", sort=present)
          )
    )

    # ---------- быстрый «общий тултип» для dislike ----------
    wide = (
        dis_counts
        .pivot_table(index=["bucket","bucket_label"], columns="aspect_en",
                     values="count", aggfunc="sum", fill_value=0)
    )

    col_order = list(wide.sum(axis=0).sort_values(ascending=False).index)

    def _pack_row_dis(r, top_k=6):
        total = int(r[col_order].sum())
        if total == 0:
            return pd.Series({"total": 0, "tooltip_text": ""})
        pairs = [(name, int(r[name])) for name in col_order if r[name] > 0]
        pairs.sort(key=lambda x: x[1], reverse=True)
        pairs = pairs[:top_k]
        lines = [f"{name} — {c} ({c/total:.0%})" for name, c in pairs]
        return pd.Series({"total": total, "tooltip_text": "\n".join(lines)})

    packed_dis = wide.apply(_pack_row_dis, axis=1).reset_index()

    bubble_dis = (
        alt.Chart(packed_dis)
          .mark_bar(size=max(40, bar_size), opacity=0.001)
          .encode(
              x=alt.X("bucket_label:N", sort=bucket_order),
              y=alt.Y("total:Q"),
              tooltip=[
                  alt.Tooltip("bucket_label:N", title="Период"),
                  alt.Tooltip("total:Q", title="Всего упоминаний"),
                  alt.Tooltip("tooltip_text:N", title=""),
              ]
          )
    )

    st.altair_chart((bars_dis + bubble_dis).properties(height=460),
                    use_container_width=True, theme=None)

# --------- DISLIKE: «Распределение по урокам (ось X — S) — график (в %)» из F ---------
st.markdown("---")
st.subheader("Dislike по урокам (ось X — S) — график (в %)")

df_dislike_lessons = df1_base.copy()
if not df_dislike_lessons.empty and selected_lessons:
    df_dislike_lessons["S_num"] = pd.to_numeric(df_dislike_lessons["S"], errors="coerce")
    df_dislike_lessons = df_dislike_lessons[df_dislike_lessons["S_num"].isin(selected_lessons)]

cnt_by_s_dis = build_dislike_counts_by_S(df_dislike_lessons)
if cnt_by_s_dis.empty:
    st.info("Нет данных для dislike-графика по урокам.")
else:
    base_dis = (
        alt.Chart(cnt_by_s_dis)
          .transform_aggregate(count='sum(count)', groupby=['S', 'aspect_en'])
          .transform_joinaggregate(total='sum(count)', groupby=['S'])
          .transform_calculate(pct='datum.count / datum.total')
    )

    dislike_domain_en = [en for _, en in DISLIKE_ES_EN]

    bars_s_dis = (
        base_dis.mark_bar(size=28, stroke=None, strokeWidth=0)
            .encode(
                x=alt.X("S:O", title="S", sort="ascending"),
                y=alt.Y(
                    "count:Q",
                    stack="normalize",
                    axis=alt.Axis(format="%", title="% от упоминаний"),
                    scale=alt.Scale(domain=[0, 1], nice=False, clamp=True)
                ),
                color=alt.Color(
                    "aspect_en:N",
                    title="Dislike-аспект (EN)",
                    scale=alt.Scale(domain=dislike_domain_en),
                    legend=alt.Legend(
                        orient="bottom",
                        direction="horizontal",
                        columns=2,
                        labelLimit=1000,
                        titleLimit=1000,
                        symbolType="square",
                    ),
                ),
                tooltip=[
                    alt.Tooltip("S:O", title="Урок"),
                    alt.Tooltip("aspect_en:N", title="Dislike-аспект"),
                    alt.Tooltip("count:Q", title="Кол-во"),
                    alt.Tooltip("pct:Q", title="Доля", format=".0%"),
                    alt.Tooltip("total:Q", title="Всего по уроку"),
                ],
            )
    ).configure_legend(labelLimit=1000, titleLimit=1000)

    st.altair_chart(bars_s_dis.properties(height=460), use_container_width=True, theme=None)

# --------- DISLIKE: «Распределение по урокам (ось X — S) — таблица» из F ---------
st.markdown("---")
st.subheader("Dislike по урокам (ось X — S) — таблица")

rows_dislike = []
unknown_dislike_per_s = {}

if not df_dislike_lessons.empty and {"S", "F"}.issubset(df_dislike_lessons.columns):
    df_tmp = df_dislike_lessons[["S", "F"]].copy()
    df_tmp["S"] = pd.to_numeric(df_tmp["S"], errors="coerce")
    df_tmp = df_tmp.dropna(subset=["S"])
    df_tmp["S"] = df_tmp["S"].astype(int)

    dislike_norm = [(_norm_local(es), en) for es, en in DISLIKE_ES_EN]

    for _, rr in df_tmp.iterrows():
        s_val = int(rr["S"])
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
                    rows_dislike.append((s_val, en))
                    matched = True
                    break
            if not matched:
                unknown_dislike_per_s.setdefault(s_val, Counter())[p_clean] += 1

if not rows_dislike and not unknown_dislike_per_s:
    st.info("Нет данных для таблицы dislike по урокам.")
else:
    df_dis = (pd.DataFrame(rows_dislike, columns=["S", "aspect_en"])
              if rows_dislike else pd.DataFrame(columns=["S", "aspect_en"]))
    lesson_rows = []
    lessons_sorted = sorted(set(list(df_dis["S"].unique()) + list(unknown_dislike_per_s.keys())))

    for s in lessons_sorted:
        if not df_dis.empty and s in df_dis["S"].values:
            cnt = (df_dis[df_dis["S"] == s]["aspect_en"]
                   .value_counts()
                   .sort_values(ascending=False))
            total_tpl = int(cnt.sum())
            parts_text = []
            for en_name, c in cnt.items():
                pct = (c / total_tpl) if total_tpl else 0.0
                parts_text.append(f"• {en_name} — {c} ({pct:.0%})")
            aspects_text = "\n".join(parts_text)
        else:
            aspects_text = ""
            total_tpl = 0

        unk_counter = unknown_dislike_per_s.get(s, Counter())
        if unk_counter:
            top_items = unk_counter.most_common(10)
            rest = sum(unk_counter.values()) - sum(c for _, c in top_items)
            unk_parts = [f"• {translate_es_to_en(m)} ({c})" for m, c in top_items]
            if rest > 0:
                unk_parts.append(f"• … (+{rest})")
            unknown_text = "\n".join(unk_parts)
        else:
            unknown_text = ""

        lesson_rows.append({
            "S": int(s),
            "Dislike (EN)": aspects_text,
            "Unknown (EN)": unknown_text,
            "Total (templ.)": total_tpl
        })

    dislike_table = pd.DataFrame(lesson_rows).sort_values("S").reset_index(drop=True)
    height = min(800, 100 + 28 * len(dislike_table))
    st.dataframe(
        dislike_table[["S", "Dislike (EN)", "Unknown (EN)", "Total (templ.)"]],
        use_container_width=True,
        height=height
    )

# --------- ADDITIONAL COMMENTS (колонка H, по урокам S) ---------
st.markdown("---")
st.subheader("Additional comments — по урокам (S)")

df_additional = df1_base.copy()
if not df_additional.empty and selected_lessons:
    df_additional["S_num"] = pd.to_numeric(df_additional["S"], errors="coerce")
    df_additional = df_additional[df_additional["S_num"].isin(selected_lessons)]

if df_additional.empty or not {"S", "H"}.issubset(df_additional.columns):
    st.info("Нет данных для Additional comments (ожидаются колонки S и H на листе Form Responses 1).")
else:
    df_tmp = df_additional[["S", "H"]].copy()
    df_tmp["S"] = pd.to_numeric(df_tmp["S"], errors="coerce")
    df_tmp = df_tmp.dropna(subset=["S"])
    df_tmp["S"] = df_tmp["S"].astype(int)

    comments_per_s = {}
    for _, rr in df_tmp.iterrows():
        s_val = int(rr["S"])
        txt = str(rr["H"] or "").strip()
        if not txt:
            continue
        parts = re.split(r"[;,/\n|]+", txt) if re.search(r"[;,/\n|]", txt) else [txt]
        for p in parts:
            p_clean = p.strip()
            if not p_clean:
                continue
            comments_per_s.setdefault(s_val, Counter())[p_clean] += 1

    if not comments_per_s:
        st.info("Нет данных для Additional comments по выбранным фильтрам.")
    else:
        rows = []
        for s in sorted(comments_per_s.keys()):
            counter = comments_per_s[s]
            total = int(sum(counter.values()))
            items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
            bullets_en = []
            for txt, c in items:
                pct = (c / total) if total else 0.0
                bullets_en.append(f"• {translate_es_to_en(txt)} — {c} ({pct:.0%})")
            rows.append({
                "S": s,
                "Comments (EN)": "\n".join(bullets_en),
                "Total comments": total,
            })

        add_table = pd.DataFrame(rows).sort_values("S").reset_index(drop=True)
        height = min(900, 120 + 28 * len(add_table))
        st.dataframe(
            add_table[["S", "Comments (EN)", "Total comments"]],
            use_container_width=True,
            height=height
        )

# ---------- FR2: распределения по шаблонным текстам D и E ----------
st.markdown("---")
st.subheader(f"Form Responses 2 — шаблонные распределения (гранулярность: {granularity.lower()})")

# Источник: df2_base (курсы/даты) + фильтр по урокам (R)
def _apply_r_filter(df_src: pd.DataFrame) -> pd.DataFrame:
    if df_src.empty or "R" not in df_src.columns:
        return df_src
    if selected_lessons:
        d = df_src.copy()
        d["R_num"] = pd.to_numeric(d["R"], errors="coerce")
        d = d[d["R_num"].isin(selected_lessons)]
        return d
    return df_src

df2_text_src = _apply_r_filter(df2_base)

colD, colE = st.columns(2)

# ---- D (по шаблонам) ----
with colD:
    st.markdown("**FR2 — распределение по D (шаблоны)**")
    if df2_text_src.empty or "D" not in df2_text_src.columns:
        st.info("Нет данных (колонка D отсутствует или фильтры пустые).")
    else:
        d_cnt = build_template_counts(df2_text_src, text_col="D", date_col="A",
                                      granularity=granularity, templates_es_en=FR2_D_TEMPL_ES_EN)
        if d_cnt.empty:
            st.info("Нет совпадений с шаблонными фразами для D.")
        else:
            # период и легенда
            d_bucket_order = (d_cnt[["bucket","bucket_label"]]
                              .drop_duplicates()
                              .sort_values("bucket")["bucket_label"].tolist())
            legend_domain = [en for _, en in FR2_D_TEMPL_ES_EN]

            # добавим total/pct для тултипов
            d_totals = (d_cnt.groupby(["bucket","bucket_label"], as_index=False)["count"]
                            .sum().rename(columns={"count":"total"}))
            d_out = d_cnt.merge(d_totals, on=["bucket","bucket_label"], how="left")
            d_out["pct"] = d_out["count"] / d_out["total"]

            barsD = (
                alt.Chart(d_out)
                   .mark_bar(size=BAR_SIZE.get(granularity, 36))
                   .encode(
                       x=alt.X("bucket_label:N", title="Период", sort=d_bucket_order),
                       y=alt.Y("sum(count):Q", title="Кол-во ответов"),
                       color=alt.Color(
                           "templ_en:N",
                           title="D (EN)",
                           scale=alt.Scale(domain=legend_domain),
                       ),
                       tooltip=[
                           alt.Tooltip("bucket_label:N", title="Период"),
                           alt.Tooltip("templ_en:N", title="Шаблон (EN)"),
                           alt.Tooltip("count:Q", title="Кол-во"),
                           alt.Tooltip("pct:Q", title="% внутри периода", format=".0%"),
                       ],
                   )
                   .properties(height=420)
            )
            st.altair_chart(barsD, use_container_width=True, theme=None)

# ---- E (по шаблонам) ----
with colE:
    st.markdown("**FR2 — распределение по E (шаблоны)**")
    if df2_text_src.empty or "E" not in df2_text_src.columns:
        st.info("Нет данных (колонка E отсутствует или фильтры пустые).")
    else:
        e_cnt = build_template_counts(df2_text_src, text_col="E", date_col="A",
                                      granularity=granularity, templates_es_en=FR2_E_TEMPL_ES_EN)
        if e_cnt.empty:
            st.info("Нет совпадений с шаблонными фразами для E.")
        else:
            e_bucket_order = (e_cnt[["bucket","bucket_label"]]
                              .drop_duplicates()
                              .sort_values("bucket")["bucket_label"].tolist())
            legend_domain_e = [en for _, en in FR2_E_TEMPL_ES_EN]

            e_totals = (e_cnt.groupby(["bucket","bucket_label"], as_index=False)["count"]
                           .sum().rename(columns={"count":"total"}))
            e_out = e_cnt.merge(e_totals, on=["bucket","bucket_label"], how="left")
            e_out["pct"] = e_out["count"] / e_out["total"]

            barsE = (
                alt.Chart(e_out)
                   .mark_bar(size=BAR_SIZE.get(granularity, 36))
                   .encode(
                       x=alt.X("bucket_label:N", title="Период", sort=e_bucket_order),
                       y=alt.Y("sum(count):Q", title="Кол-во ответов"),
                       color=alt.Color(
                           "templ_en:N",
                           title="E (EN)",
                           scale=alt.Scale(domain=legend_domain_e),
                       ),
                       tooltip=[
                           alt.Tooltip("bucket_label:N", title="Период"),
                           alt.Tooltip("templ_en:N", title="Шаблон (EN)"),
                           alt.Tooltip("count:Q", title="Кол-во"),
                           alt.Tooltip("pct:Q", title="% внутри периода", format=".0%"),
                       ],
                   )
                   .properties(height=420)
            )
            st.altair_chart(barsE, use_container_width=True, theme=None)

# --------- FR2: по урокам (ось X — R) — графики в % по D и E ---------
st.markdown("---")
st.subheader("FR2 — распределение по урокам (ось X — R) — графики (в %)")

# Источник — df2_base + фильтр по выбранным урокам (R)
df2_lessons = df2_base.copy()
if not df2_lessons.empty and selected_lessons:
    df2_lessons["R_num"] = pd.to_numeric(df2_lessons["R"], errors="coerce")
    df2_lessons = df2_lessons[df2_lessons["R_num"].isin(selected_lessons)]

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**FR2 — по D (шаблоны), % внутри R**")
    cnt_by_r_D = build_template_counts_by_R(df2_lessons, text_col="D", templates_es_en=FR2_D_TEMPL_ES_EN)
    if cnt_by_r_D.empty:
        st.info("Нет данных для графика по D.")
    else:
        legend_domain_D = [en for _, en in FR2_D_TEMPL_ES_EN]
        base_D = (
            alt.Chart(cnt_by_r_D)
              .transform_aggregate(count='sum(count)', groupby=['R', 'templ_en'])
              .transform_joinaggregate(total='sum(count)', groupby=['R'])
              .transform_calculate(pct='datum.count / datum.total')
        )
        bars_D = (
            base_D.mark_bar(size=28, stroke=None, strokeWidth=0)
              .encode(
                  x=alt.X("R:O", title="R", sort="ascending"),
                  y=alt.Y("count:Q",
                          stack="normalize",
                          axis=alt.Axis(format="%", title="% от упоминаний"),
                          scale=alt.Scale(domain=[0, 1], nice=False, clamp=True)),
                  color=alt.Color(
                      "templ_en:N",
                      title="Template (EN)",
                      scale=alt.Scale(domain=legend_domain_D),
                      legend=alt.Legend(
                          orient="bottom", direction="horizontal", columns=2,
                          labelLimit=1000, titleLimit=1000, symbolType="square"
                      ),
                  ),
                  tooltip=[
                      alt.Tooltip("R:O", title="Урок"),
                      alt.Tooltip("templ_en:N", title="Шаблон"),
                      alt.Tooltip("count:Q", title="Кол-во"),
                      alt.Tooltip("pct:Q", title="Доля", format=".0%"),
                      alt.Tooltip("total:Q", title="Всего по уроку"),
                  ],
              )
        ).configure_legend(labelLimit=1000, titleLimit=1000)
        st.altair_chart(bars_D.properties(height=420), use_container_width=True, theme=None)

with col_right:
    st.markdown("**FR2 — по E (шаблоны), % внутри R**")
    cnt_by_r_E = build_template_counts_by_R(df2_lessons, text_col="E", templates_es_en=FR2_E_TEMPL_ES_EN)
    if cnt_by_r_E.empty:
        st.info("Нет данных для графика по E.")
    else:
        legend_domain_E = [en for _, en in FR2_E_TEMPL_ES_EN]
        base_E = (
            alt.Chart(cnt_by_r_E)
              .transform_aggregate(count='sum(count)', groupby=['R', 'templ_en'])
              .transform_joinaggregate(total='sum(count)', groupby=['R'])
              .transform_calculate(pct='datum.count / datum.total')
        )
        bars_E = (
            base_E.mark_bar(size=28, stroke=None, strokeWidth=0)
              .encode(
                  x=alt.X("R:O", title="R", sort="ascending"),
                  y=alt.Y("count:Q",
                          stack="normalize",
                          axis=alt.Axis(format="%", title="% от упоминаний"),
                          scale=alt.Scale(domain=[0, 1], nice=False, clamp=True)),
                  color=alt.Color(
                      "templ_en:N",
                      title="Template (EN)",
                      scale=alt.Scale(domain=legend_domain_E),
                      legend=alt.Legend(
                          orient="bottom", direction="horizontal", columns=2,
                          labelLimit=1000, titleLimit=1000, symbolType="square"
                      ),
                  ),
                  tooltip=[
                      alt.Tooltip("R:O", title="Урок"),
                      alt.Tooltip("templ_en:N", title="Шаблон"),
                      alt.Tooltip("count:Q", title="Кол-во"),
                      alt.Tooltip("pct:Q", title="Доля", format=".0%"),
                      alt.Tooltip("total:Q", title="Всего по уроку"),
                  ],
              )
        ).configure_legend(labelLimit=1000, titleLimit=1000)
        st.altair_chart(bars_E.properties(height=420), use_container_width=True, theme=None)

# --------- FR2: три графика "Average by R" по колонкам F, G, H ---------
st.markdown("---")
st.subheader("Form Responses 2 — Averages by R (F, G, H)")

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

aggF = _avg_by_r(df2_base, "F") if "F" in df2.columns else pd.DataFrame()
aggG_2 = _avg_by_r(df2_base, "G") if "G" in df2.columns else pd.DataFrame()
aggH = _avg_by_r(df2_base, "H") if "H" in df2.columns else pd.DataFrame()

colF, colG2, colH = st.columns(3)

def _make_avg_chart(df_avg: pd.DataFrame, title_y: str):
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
              x=alt.X("R:Q", title="R"),
              y=alt.Y("avg_y:Q", title=title_y, scale=y_scale),
              tooltip=[
                  alt.Tooltip("R:Q", title="R"),
                  alt.Tooltip("avg_y:Q", title=title_y, format=".2f"),
                  alt.Tooltip("count:Q", title="Кол-во ответов"),
              ],
          )
          .properties(height=340)
    )
    st.altair_chart(chart, use_container_width=True, theme=None)

with colF:
    st.markdown("**FR2 — Average F by R**")
    _make_avg_chart(aggF, "Average F")

with colG2:
    st.markdown("**FR2 — Average G by R**")
    _make_avg_chart(aggG_2, "Average G")

with colH:
    st.markdown("**FR2 — Average H by R**")
    _make_avg_chart(aggH, "Average H")

# ---------- FR2: распределения по F / G / H (по типу "Распределение значений") ----------
st.markdown("---")
st.subheader(f"Form Responses 2 — распределение F / G / H (гранулярность: {granularity.lower()})")

def _prep_df2_numeric_dist(df_src: pd.DataFrame, value_col: str, granularity: str):
    """Готовим df с bucket’ами и списком допустимых значений (инт)."""
    if df_src.empty or value_col not in df_src.columns:
        return pd.DataFrame(), [], [], value_col
    d = df_src.copy()
    d = d.dropna(subset=["A", value_col])
    # приводим к числу и убираем NaN
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[value_col])
    if d.empty:
        return pd.DataFrame(), [], [], value_col

    # только целые значения шкалы
    d[value_col] = d[value_col].astype(int)

    # добавляем buckets и подписи
    d = add_bucket(d, "A", granularity)
    d = ensure_bucket_and_label(d, "A", granularity)

    # динамически определяем допустимые значения (встречающиеся целые)
    allowed_values = sorted(d[value_col].dropna().unique().tolist())
    if not allowed_values:
        return pd.DataFrame(), [], [], value_col

    # используем существующую утилиту, чтобы получить out/pct и порядки
    out, bucket_order, val_order, title = prep_distribution(d, value_col, allowed_values, value_col)
    return out, bucket_order, val_order, title

# источник с учётом фильтра по урокам (R)
df2_numeric_src = _apply_r_filter(df2_base)

cF, cG, cH = st.columns(3)

with cF:
    st.markdown("**FR2 — распределение F**")
    outF, orderF, valsF, titleF = _prep_df2_numeric_dist(df2_numeric_src, "F", granularity)
    if outF.empty:
        st.info("Нет данных по колонке F для выбранных фильтров.")
    else:
        barsF = (
            alt.Chart(outF).mark_bar(size=BAR_SIZE.get(granularity, 36))
              .encode(
                  x=alt.X("bucket_label:N", title="Период", sort=orderF),
                  y=alt.Y("sum(count):Q", title="Кол-во ответов"),
                  color=alt.Color("val_str:N", title=titleF, sort=valsF),
                  order=alt.Order("val:Q", sort="ascending"),
                  tooltip=[
                      alt.Tooltip("bucket_label:N", title="Период"),
                      alt.Tooltip("val_str:N", title=titleF),
                      alt.Tooltip("count:Q", title="Кол-во"),
                      alt.Tooltip("pct:Q", title="% внутри периода", format=".0%"),
                  ],
              )
              .properties(height=420)
        )
        st.altair_chart(barsF, use_container_width=True, theme=None)

with cG:
    st.markdown("**FR2 — распределение G**")
    outG, orderG, valsG, titleG = _prep_df2_numeric_dist(df2_numeric_src, "G", granularity)
    if outG.empty:
        st.info("Нет данных по колонке G для выбранных фильтров.")
    else:
        barsG = (
            alt.Chart(outG).mark_bar(size=BAR_SIZE.get(granularity, 36))
              .encode(
                  x=alt.X("bucket_label:N", title="Период", sort=orderG),
                  y=alt.Y("sum(count):Q", title="Кол-во ответов"),
                  color=alt.Color("val_str:N", title=titleG, sort=valsG),
                  order=alt.Order("val:Q", sort="ascending"),
                  tooltip=[
                      alt.Tooltip("bucket_label:N", title="Период"),
                      alt.Tooltip("val_str:N", title=titleG),
                      alt.Tooltip("count:Q", title="Кол-во"),
                      alt.Tooltip("pct:Q", title="% внутри периода", format=".0%"),
                  ],
              )
              .properties(height=420)
        )
        st.altair_chart(barsG, use_container_width=True, theme=None)

with cH:
    st.markdown("**FR2 — распределение H**")
    outH, orderH, valsH, titleH = _prep_df2_numeric_dist(df2_numeric_src, "H", granularity)
    if outH.empty:
        st.info("Нет данных по колонке H для выбранных фильтров.")
    else:
        barsH = (
            alt.Chart(outH).mark_bar(size=BAR_SIZE.get(granularity, 36))
              .encode(
                  x=alt.X("bucket_label:N", title="Период", sort=orderH),
                  y=alt.Y("sum(count):Q", title="Кол-во ответов"),
                  color=alt.Color("val_str:N", title=titleH, sort=valsH),
                  order=alt.Order("val:Q", sort="ascending"),
                  tooltip=[
                      alt.Tooltip("bucket_label:N", title="Период"),
                      alt.Tooltip("val_str:N", title=titleH),
                      alt.Tooltip("count:Q", title="Кол-во"),
                      alt.Tooltip("pct:Q", title="% внутри периода", format=".0%"),
                  ],
              )
              .properties(height=420)
        )
        st.altair_chart(barsH, use_container_width=True, theme=None)

# ---------- FR2: по урокам (ось X — R) — графики (в %) для F / G / H ----------
st.markdown("---")
st.subheader("Form Responses 2 — по урокам (ось X — R) — графики (в %)")

def _build_numeric_counts_by_R(df_src: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Считает, сколько раз встречается каждое целое значение шкалы в колонке value_col
    внутри каждого урока R. Возвращает колонки: R, val, val_str, count, total.
    """
    if df_src.empty or value_col not in df_src.columns or "R" not in df_src.columns:
        return pd.DataFrame(columns=["R","val","val_str","count","total"])

    d = df_src.copy()
    d["R"] = pd.to_numeric(d["R"], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=["R", value_col])
    if d.empty:
        return pd.DataFrame(columns=["R","val","val_str","count","total"])

    d["R"] = d["R"].astype(int)
    d["val"] = d[value_col].astype(int)
    d["val_str"] = d["val"].astype(str)

    grp = (d.groupby(["R","val","val_str"], as_index=False)
             .size().rename(columns={"size":"count"}))
    totals = (grp.groupby("R", as_index=False)["count"]
                .sum().rename(columns={"count":"total"}))
    out = grp.merge(totals, on="R", how="left")
    return out

def _make_percent_stack_by_R(out_df: pd.DataFrame, legend_title: str):
    """
    Рисует нормированный стек по R (0–100%) с тултипом: урок, значение, count, pct, total.
    """
    if out_df.empty:
        return None

    # порядок уроков и значений
    r_order = sorted(out_df["R"].unique().tolist())
    val_order = sorted(out_df["val"].unique().tolist())
    val_order_str = [str(v) for v in val_order]

    base = (
        alt.Chart(out_df)
          .transform_calculate(pct='datum.count / datum.total')
    )

    chart = (
        base.mark_bar(size=28, stroke=None, strokeWidth=0)
            .encode(
                x=alt.X("R:O", title="R", sort=r_order),
                y=alt.Y(
                    "count:Q",
                    stack="normalize",
                    axis=alt.Axis(format="%", title="% от ответов"),
                    scale=alt.Scale(domain=[0, 1], nice=False, clamp=True)
                ),
                color=alt.Color(
                    "val_str:N",
                    title=legend_title,
                    sort=val_order_str,
                    legend=alt.Legend(
                        orient="bottom",
                        direction="horizontal",
                        columns=3,
                        labelLimit=1000,
                        titleLimit=1000,
                        symbolType="square",
                    ),
                ),
                tooltip=[
                    alt.Tooltip("R:O", title="Урок (R)"),
                    alt.Tooltip("val_str:N", title=legend_title),
                    alt.Tooltip("count:Q", title="Кол-во"),
                    alt.Tooltip("pct:Q", title="Доля", format=".0%"),
                    alt.Tooltip("total:Q", title="Всего по уроку"),
                ],
            )
    ).configure_legend(labelLimit=1000, titleLimit=1000)

    return chart

# источник с учётом фильтра по урокам (R)
df2_lessons_src = _apply_r_filter(df2_base)

cF2, cG2, cH2 = st.columns(3)

with cF2:
    st.markdown("**FR2 — по урокам (R) — F (в %)**")
    outF_R = _build_numeric_counts_by_R(df2_lessons_src, "F")
    if outF_R.empty:
        st.info("Нет данных по F для выбранных фильтров.")
    else:
        chF_R = _make_percent_stack_by_R(outF_R, "F")
        st.altair_chart(chF_R.properties(height=460), use_container_width=True, theme=None)

with cG2:
    st.markdown("**FR2 — по урокам (R) — G (в %)**")
    outG_R = _build_numeric_counts_by_R(df2_lessons_src, "G")
    if outG_R.empty:
        st.info("Нет данных по G для выбранных фильтров.")
    else:
        chG_R = _make_percent_stack_by_R(outG_R, "G")
        st.altair_chart(chG_R.properties(height=460), use_container_width=True, theme=None)

with cH2:
    st.markdown("**FR2 — по урокам (R) — H (в %)**")
    outH_R = _build_numeric_counts_by_R(df2_lessons_src, "H")
    if outH_R.empty:
        st.info("Нет данных по H для выбранных фильтров.")
    else:
        chH_R = _make_percent_stack_by_R(outH_R, "H")
        st.altair_chart(chH_R.properties(height=460), use_container_width=True, theme=None)

