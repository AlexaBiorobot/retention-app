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

# простой пословный словарик (fallback)
_BASIC_ES_EN = {
    "si": "yes", "sí": "yes", "no": "no", "ok": "ok",
    "materia":"subject","explicacion":"explanation","explicación":"explanation",
    "profesor":"teacher","actividades":"activities","sala":"classroom",
    "tareas":"homework","casa":"home","forma":"way","comporto":"behaved",
    "comportó":"behaved","clase":"class","aclararon":"clarified","dudas":"doubts",
    "trataron":"treated","bien":"well","atencion":"attention","atención":"attention"
}
STOPWORDS_NORM = set(["si","sí","no","ok","-","y","la","el","los","las","un","una","uno","al","por","para","con","en","de","del"])

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
            if not matched and t not in STOPWORDS_NORM:
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

# ---- помощник для динамических границ Y
def _y_domain(series: pd.Series, pad_ratio: float = 0.1):
    s = series.dropna()
    if s.empty:
        return [0, 1]
    ymin = float(s.min())
    ymax = float(s.max())
    if ymin == ymax:
        delta = max(0.1, abs(ymin) * 0.1)
        return [ymin - delta, ymax + delta]
    pad = (ymax - ymin) * pad_ratio
    return [ymin - pad, ymax + pad]

# ---- безопасная подача данных в Altair
def _records(df: pd.DataFrame):
    if df.empty:
        return []
    clean = (df.copy()
               .astype(object)
               .where(pd.notnull(df), None))
    return clean.to_dict("records")

# ==================== ДАННЫЕ ====================

df1 = load_sheet_as_letter_df("Form Responses 1")   # A=date, N=course, S=x, G=y
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

# ==================== ЕДИНЫЕ ФИЛЬТРЫ ====================

st.sidebar.header("Фильтры")

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

min1, max1 = (df1["A"].min(), df1["A"].max()) if not df1.empty else (pd.NaT, pd.NaT)
min2, max2 = (df2["A"].min(), df2["A"].max()) if not df2.empty else (pd.NaT, pd.NaT)
glob_min, glob_max = safe_minmax(min1, max1, min2, max2)

if pd.isna(glob_min) or pd.isna(glob_max):
    date_range = st.sidebar.date_input("Дата фидбека (A)", [])
else:
    date_range = st.sidebar.date_input("Дата фидбека (A)", [glob_min.date(), glob_max.date()])

granularity = st.sidebar.selectbox("Гранулярность для распределения", ["День", "Неделя", "Месяц", "Год"])
BAR_SIZE = {"День": 18, "Неделя": 44, "Месяц": 56, "Год": 64}
bar_size = BAR_SIZE.get(granularity, 36)

# ==================== ПРИМЕНЕНИЕ ФИЛЬТРОВ ====================

agg1 = apply_filters_and_aggregate(df1, "N", "A", "S", "G", selected_courses, date_range)
agg2 = apply_filters_and_aggregate(df2, "M", "A", "R", "I", selected_courses, date_range)

df1_f = filter_df(df1, "N", "A", selected_courses, date_range)
df2_f = filter_df(df2, "M", "A", selected_courses, date_range)

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
        y_dom1 = _y_domain(agg1["avg_y"], pad_ratio=0.1)
        chart1 = (
            alt.Chart(alt.Data(values=_records(agg1)))
              .mark_line(point=True)
              .encode(
                  x=alt.X("S:Q", title="S"),
                  y=alt.Y("avg_y:Q",
                          title="Average G",
                          scale=alt.Scale(domain=y_dom1, nice=False, clamp=True)),
                  tooltip=[
                      alt.Tooltip("S:Q", title="S"),
                      alt.Tooltip("avg_y:Q", title="Average G", format=".2f"),
                      alt.Tooltip("count:Q", title="Кол-во ответов")
                  ])
              .properties(height=380)
        )
        st.altair_chart(chart1, theme=None, use_container_width=True)

with col2:
    st.subheader("Form Responses 2 — Average by R")
    if agg2.empty:
        st.info("Нет данных для выбранных фильтров.")
    else:
        y_dom2 = _y_domain(agg2["avg_y"], pad_ratio=0.1)
        chart2 = (
            alt.Chart(alt.Data(values=_records(agg2)))
              .mark_line(point=True)
              .encode(
                  x=alt.X("R:Q", title="R"),
                  y=alt.Y("avg_y:Q",
                          title="Average I",
                          scale=alt.Scale(domain=y_dom2, nice=False, clamp=True)),
                  tooltip=[
                      alt.Tooltip("R:Q", title="R"),
                      alt.Tooltip("avg_y:Q", title="Average I", format=".2f"),
                      alt.Tooltip("count:Q", title="Кол-во ответов")
                  ])
              .properties(height=380)
        )
        st.altair_chart(chart2, theme=None, use_container_width=True)

st.markdown("---")
st.subheader(f"Распределение значений (гранулярность: {granularity.lower()})")

col3, col4 = st.columns([1, 1])

with col3:
    st.markdown("**Form Responses 1 — распределение G (1–5)**")
    if fr1_out.empty:
        st.info("Нет данных (FR1).")
    else:
        bars1 = (
            alt.Chart(alt.Data(values=_records(fr1_out)))
              .mark_bar(size=bar_size)
              .encode(
                  x=alt.X("bucket_label:N", title="Период", sort=fr1_bucket_order),
                  y=alt.Y("count:Q", title="Кол-во ответов", aggregate="sum"),
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
        st.altair_chart(bars1, theme=None, use_container_width=True)

with col4:
    st.markdown("**Form Responses 2 — распределение I (1–10)**")
    if fr2_out.empty:
        st.info("Нет данных (FR2).")
    else:
        bars2 = (
            alt.Chart(alt.Data(values=_records(fr2_out)))
              .mark_bar(size=bar_size)
              .encode(
                  x=alt.X("bucket_label:N", title="Период", sort=fr2_bucket_order),
                  y=alt.Y("count:Q", title="Кол-во ответов", aggregate="sum"),
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
        st.altair_chart(bars2, theme=None, use_container_width=True)

# ---------- НИЖЕ: Аспекты урока из FR1 ----------
st.markdown("---")
st.subheader("Аспекты урока — Form Responses 1")

df_aspects = filter_df(df1, "N", "A", selected_courses, date_range)
asp_counts, unknown_all = build_aspects_counts(df_aspects, text_col="E", date_col="A", granularity=granularity)

left, right = st.columns([1, 1])

with left:
    st.markdown("**Аспекты по датам (ось X — A)**")
    if asp_counts.empty:
        st.info("Не нашёл упоминаний аспектов (лист 'Form Responses 1', колонка E).")
    else:
        bucket_order = (asp_counts[["bucket","bucket_label"]]
                        .drop_duplicates()
                        .sort_values("bucket")["bucket_label"].tolist())

        totals_by_bucket = (asp_counts.groupby("bucket_label", as_index=False)["count"]
                                      .sum().rename(columns={"count":"total"}))
        y_max = max(1, int(totals_by_bucket["total"].max()))
        y_scale_bar = alt.Scale(domain=[0, y_max * 1.1], nice=False, clamp=True)

        expected_labels = [f"{es} (EN: {en})" for es, en in ASPECTS_ES_EN]
        present = [lbl for lbl in expected_labels if lbl in asp_counts["aspect"].unique()]

        bars = (
            alt.Chart(alt.Data(values=_records(asp_counts)))
              .mark_bar(size=max(40, bar_size))
              .encode(
                  x=alt.X("bucket_label:N", title="Период (по A)", sort=bucket_order),
                  y=alt.Y("count:Q", title="Кол-во упоминаний", aggregate="sum", scale=y_scale_bar),
                  color=alt.Color("aspect:N", title="Аспект", sort=present)
              )
        )
        st.altair_chart(bars.properties(height=460), theme=None, use_container_width=True)

with right:
    st.markdown("**Распределение по урокам (ось X — S)**")

    # собираем для каждого урока S: шаблонные аспекты и «вне шаблона»
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
                        rows_aspects.append((s_val, en))  # берём только EN для вывода
                        matched = True
                        break
                if not matched and t not in STOPWORDS_NORM:
                    unknown_per_s.setdefault(s_val, Counter())[p_clean] += 1

    if not rows_aspects and not unknown_per_s:
        st.info("Нет данных для выбранных фильтров.")
    else:
        # агрегируем шаблонные
        df_as = (pd.DataFrame(rows_aspects, columns=["S","aspect_en"])
                 if rows_aspects else pd.DataFrame(columns=["S","aspect_en"]))
        lesson_rows = []

        lessons_sorted = sorted(set(list(df_as["S"].unique()) + list(unknown_per_s.keys())))
        for s in lessons_sorted:
            # шаблонные
            if not df_as.empty and s in df_as["S"].values:
                cnt = (df_as[df_as["S"] == s]["aspect_en"]
                       .value_counts()
                       .sort_values(ascending=False))
                total_tpl = int(cnt.sum())
                parts_text = []
                for en_name, c in cnt.items():
                    pct = (c / total_tpl) if total_tpl else 0.0
                    parts_text.append(f"{en_name} — {c} ({pct:.0%})")
                aspects_text = " • ".join(parts_text)
            else:
                aspects_text = ""
                total_tpl = 0

            # вне шаблона
            unk_counter = unknown_per_s.get(s, Counter())
            if unk_counter:
                top_items = unk_counter.most_common(10)
                rest = sum(unk_counter.values()) - sum(c for _, c in top_items)
                unk_parts = [f"{translate_es_to_en(m)} ({c})" for m, c in top_items]
                if rest > 0:
                    unk_parts.append(f"… (+{rest})")
                unknown_text = " • ".join(unk_parts)
            else:
                unknown_text = ""

            lesson_rows.append({"S": int(s),
                                "Aspects (EN)": aspects_text,
                                "Unknown (EN)": unknown_text,
                                "Total (templ.)": total_tpl})

        table = pd.DataFrame(lesson_rows).sort_values("S").reset_index(drop=True)
        st.dataframe(table[["S","Aspects (EN)","Unknown (EN)","Total (templ.)"]],
                     use_container_width=True)

# неизвестные упоминания (не из шаблона) — общий список
st.markdown("#### Упоминания вне шаблона (общий список)")
if 'unknown_all' in locals() and not unknown_all.empty:
    unknown_agg = (unknown_all.groupby(["en","mention"], as_index=False)
                   .agg(total=("total","sum"))
                   .sort_values("total", ascending=False))
    st.dataframe(unknown_agg[["en","mention","total"]], use_container_width=True)
else:
    st.success("Все упоминания соответствуют шаблону.")

# Подсказка, если онлайн-переводчик недоступен
if _gt is None:
    st.caption("⚠️ deep-translator недоступен — используется упрощённый пословный перевод.")
