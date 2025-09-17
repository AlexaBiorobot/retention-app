import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import altair as alt
import json
import string
import re
import unicodedata

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
    # по курсам
    if selected_courses:
        dff = dff[dff[course_col].isin(selected_courses)]
    else:
        return dff.iloc[0:0]  # пусто
    # по датам
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
        out["bucket"] = out[date_col].dt.to_period("W-MON").dt.start_time  # старт недели (Пн)
    elif granularity == "Месяц":
        out["bucket"] = out[date_col].dt.to_period("M").dt.to_timestamp()
    elif granularity == "Год":
        out["bucket"] = out[date_col].dt.to_period("Y").dt.to_timestamp()
    else:
        out["bucket"] = out[date_col].dt.floor("D")
    return out

def ensure_bucket_and_label(dff: pd.DataFrame, date_col: str, granularity: str) -> pd.DataFrame:
    """Гарантируем наличие столбцов bucket (datetime) и bucket_label (подпись)."""
    if dff.empty:
        return dff.copy()
    out = dff.copy()
    if "bucket" not in out.columns or not pd.api.types.is_datetime64_any_dtype(out["bucket"]):
        out = add_bucket(out, date_col, granularity)

    if granularity == "День":
        fmt = "%Y-%m-%d"
    elif granularity == "Неделя":
        fmt = "W%W (%Y-%m-%d)"  # дата старта недели
    elif granularity == "Месяц":
        fmt = "%Y-%m"
    elif granularity == "Год":
        fmt = "%Y"
    else:
        fmt = "%Y-%m-%d"

    out["bucket_label"] = out["bucket"].dt.strftime(fmt)
    return out

def prep_distribution(df_f: pd.DataFrame, value_col: str, allowed_values: list, label_title: str):
    """
    Возвращает:
      out: DataFrame со столбцами [bucket, bucket_label, val, val_str, count, total, pct]
      bucket_order: порядок оси X по дате
      val_order: порядок легенды по значениям
      label_title: заголовок легенды (например, 'G' или 'I')
    """
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

    # порядок дат строго по bucket (datetime)
    bucket_order = (out[["bucket", "bucket_label"]]
                    .drop_duplicates()
                    .sort_values("bucket")["bucket_label"].tolist())

    val_order = [str(v) for v in allowed_values]
    return out, bucket_order, val_order, label_title

# ===== Аспекты урока: словарь и утилиты (считаем по датам A, текст в E) =====
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
    import unicodedata, re
    s = unicodedata.normalize("NFKD", s).casefold().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

_ASPECTS_NORM = [(_norm(es), es, en) for es, en in ASPECTS_ES_EN]

_BASIC_ES_EN = {
    "materia":"subject","explicacion":"explanation","explicación":"explanation",
    "profesor":"teacher","actividades":"activities","sala":"classroom",
    "tareas":"homework","casa":"home","forma":"way","comporto":"behaved",
    "comportó":"behaved","clase":"class","aclararon":"clarified","dudas":"doubts",
    "trataron":"treated","bien":"well","atencion":"attention","atención":"attention"
}
def _naive_translate_es_en(text: str) -> str:
    import unicodedata, re
    w = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", text)
    out = []
    for t in w:
        k = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii").lower()
        out.append(_BASIC_ES_EN.get(k, t))
    return " ".join(out) if out else ""

def build_aspects_counts_from_E_by_A(df: pd.DataFrame, granularity: str):
    """
    Берём дату из A, текст аспектов из E. Считаем упоминания по датам (с учётом granularity).
    Возвращает:
      counts: DataFrame [bucket(datetime), bucket_label, aspect, count]
      unknown: DataFrame [mention, en, total]
    """
    if df.empty or ("A" not in df.columns) or ("E" not in df.columns):
        return pd.DataFrame(columns=["bucket","bucket_label","aspect","count"]), pd.DataFrame(columns=["mention","en","total"])

    d = df[["A","E"]].copy()
    d["A"] = pd.to_datetime(d["A"], errors="coerce")
    d = d.dropna(subset=["A"])
    if d.empty:
        return pd.DataFrame(columns=["bucket","bucket_label","aspect","count"]), pd.DataFrame(columns=["mention","en","total"])

    # bucket & подпись по дате A
    d = add_bucket(d.rename(columns={"A":"A", "E":"E"}), "A", granularity)
    d = ensure_bucket_and_label(d, "A", granularity)

    rows, unknown = [], []
    import re
    for _, r in d.iterrows():
        text = r["E"]
        if pd.isna(text):
            continue
        txt = str(text).strip()
        if not txt:
            continue

        # допускаем несколько пунктов в одной ячейке
        parts = re.split(r"[;,/\n|]+", txt) if re.search(r"[;,/\n|]", txt) else [txt]
        for p in parts:
            p_clean = p.strip()
            t = _norm(p_clean)
            if not t:
                continue

            matched = False
            # точное попадание
            for es_norm, es, en in _ASPECTS_NORM:
                if t == es_norm:
                    rows.append((r["bucket"], r["bucket_label"], f"{es} (EN: {en})", 1))
                    matched = True
                    break
            if not matched:
                # по вхождению
                for es_norm, es, en in _ASPECTS_NORM:
                    if es_norm in t:
                        rows.append((r["bucket"], r["bucket_label"], f"{es} (EN: {en})", 1))
                        matched = True
                        break
            if not matched:
                unknown.append(p_clean)

    counts = pd.DataFrame(rows, columns=["bucket","bucket_label","aspect","count"])
    if not counts.empty:
        counts = counts.groupby(["bucket","bucket_label","aspect"], as_index=False)["count"].sum()

    unknown_df = pd.DataFrame(unknown, columns=["mention"])
    if not unknown_df.empty:
        unknown_df["mention"] = unknown_df["mention"].str.strip()
        unknown_df = (unknown_df.groupby("mention", as_index=False)
                      .size().rename(columns={"size":"total"}).sort_values("total", ascending=False))
        unknown_df["en"] = unknown_df["mention"].apply(_naive_translate_es_en)
        unknown_df = unknown_df[["mention","en","total"]]
    else:
        unknown_df = pd.DataFrame(columns=["mention","en","total"])

    return counts, unknown_df


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

# Курсы: объединяем множества из обеих таблиц
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

# Дата: общий диапазон по двум таблицам
min1, max1 = (df1["A"].min(), df1["A"].max()) if not df1.empty else (pd.NaT, pd.NaT)
min2, max2 = (df2["A"].min(), df2["A"].max()) if not df2.empty else (pd.NaT, pd.NaT)
glob_min, glob_max = safe_minmax(min1, max1, min2, max2)

if pd.isna(glob_min) or pd.isna(glob_max):
    date_range = st.sidebar.date_input("Дата фидбека (A)", [])
else:
    date_range = st.sidebar.date_input("Дата фидбека (A)", [glob_min.date(), glob_max.date()])

# Гранулярность + ширина столбиков
granularity = st.sidebar.selectbox("Гранулярность для распределения", ["День", "Неделя", "Месяц", "Год"])
BAR_SIZE = {"День": 18, "Неделя": 44, "Месяц": 56, "Год": 64}
bar_size = BAR_SIZE.get(granularity, 36)

# ==================== ПРИМЕНЕНИЕ ФИЛЬТРОВ ====================

# Верхние графики (средние)
agg1 = apply_filters_and_aggregate(df1, course_col="N", date_col="A",
                                   x_col="S", y_col="G",
                                   selected_courses=selected_courses,
                                   date_range=date_range)

agg2 = apply_filters_and_aggregate(df2, course_col="M", date_col="A",
                                   x_col="R", y_col="I",
                                   selected_courses=selected_courses,
                                   date_range=date_range)

# Нижние распределения: «сырые» значения + bucket/bucket_label
df1_f = filter_df(df1, course_col="N", date_col="A",
                  selected_courses=selected_courses, date_range=date_range)
df2_f = filter_df(df2, course_col="M", date_col="A",
                  selected_courses=selected_courses, date_range=date_range)

if not df1_f.empty:
    df1_f = df1_f.dropna(subset=["A", "G"])
    df1_f = add_bucket(df1_f, "A", granularity)
    df1_f = ensure_bucket_and_label(df1_f, "A", granularity)

if not df2_f.empty:
    df2_f = df2_f.dropna(subset=["A", "I"])
    df2_f = add_bucket(df2_f, "A", granularity)
    df2_f = ensure_bucket_and_label(df2_f, "A", granularity)

# Готовим данные для нижних графиков
fr1_allowed = [1, 2, 3, 4, 5]
fr1_out, fr1_bucket_order, fr1_val_order, fr1_title = prep_distribution(df1_f, "G", fr1_allowed, "G")

fr2_allowed = list(range(1, 11))
fr2_out, fr2_bucket_order, fr2_val_order, fr2_title = prep_distribution(df2_f, "I", fr2_allowed, "I")

# ==================== ОТРИСОВКА ====================

st.title("40 week courses")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Form Responses 1 — Average by S")
    if agg1.empty:
        st.info("Нет данных для выбранных фильтров.")
    else:
        chart1 = (
            alt.Chart(agg1)
              .mark_line(point=True)
              .encode(
                  x=alt.X("S:Q", title="S"),
                  y=alt.Y("avg_y:Q", title="Average G"),
                  tooltip=[
                      alt.Tooltip("S:Q", title="S"),
                      alt.Tooltip("avg_y:Q", title="Average G", format=".2f"),
                      alt.Tooltip("count:Q", title="Кол-во ответов")
                  ]
              )
              .properties(height=380)
        )
        st.altair_chart(chart1, use_container_width=True)

with col2:
    st.subheader("Form Responses 2 — Average by R")
    if agg2.empty:
        st.info("Нет данных для выбранных фильтров.")
    else:
        chart2 = (
            alt.Chart(agg2)
              .mark_line(point=True)
              .encode(
                  x=alt.X("R:Q", title="R"),
                  y=alt.Y("avg_y:Q", title="Average I"),
                  tooltip=[
                      alt.Tooltip("R:Q", title="R"),
                      alt.Tooltip("avg_y:Q", title="Average I", format=".2f"),
                      alt.Tooltip("count:Q", title="Кол-во ответов")
                  ]
              )
              .properties(height=380)
        )
        st.altair_chart(chart2, use_container_width=True)

# ---------- НИЖНИЙ РЯД: РАСПРЕДЕЛЕНИЯ (широкие stacked bars; высота = кол-во) ----------
st.markdown("---")
st.subheader(f"Распределение значений (гранулярность: {granularity.lower()})")

col3, col4 = st.columns([1, 1])

with col3:
    st.markdown("**Form Responses 1 — распределение G (1–5)**")
    if fr1_out.empty:
        st.info("Нет данных (FR1).")
    else:
        bars1 = (
            alt.Chart(fr1_out)
              .mark_bar(size=bar_size)
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
                  ]
              )
              .properties(height=420)
        )
        st.altair_chart(bars1, use_container_width=True)

with col4:
    st.markdown("**Form Responses 2 — распределение I (1–10)**")
    if fr2_out.empty:
        st.info("Нет данных (FR2).")
    else:
        bars2 = (
            alt.Chart(fr2_out)
              .mark_bar(size=bar_size)
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
                  ]
              )
              .properties(height=420)
        )
        st.altair_chart(bars2, use_container_width=True)

# ---------- НИЖЕ: Аспекты урока (E) по датам из A ----------
st.markdown("---")
st.subheader("Аспекты урока (по датам A, текст из E)")

# считаем отдельно по двум листам и объединяем
asp1, unk1 = build_aspects_counts_from_E_by_A(df1, granularity)
asp2, unk2 = build_aspects_counts_from_E_by_A(df2, granularity)

aspects_all = pd.concat([asp1, asp2], ignore_index=True)
unknown_all = pd.concat([unk1, unk2], ignore_index=True)

if aspects_all.empty:
    st.info("Не нашёл упоминаний аспектов в колонке E.")
else:
    # порядок по реальной дате
    bucket_order = (aspects_all[["bucket","bucket_label"]]
                    .drop_duplicates()
                    .sort_values("bucket")["bucket_label"].tolist())

    # какие аспекты реально встречаются (в нужном порядке из словаря)
    expected_labels = [f"{es} (EN: {en})" for es, en in ASPECTS_ES_EN]
    present = [lbl for lbl in expected_labels if lbl in aspects_all["aspect"].unique()]

    # ---------- stacked bar (визуал)
    bars = (
        alt.Chart(aspects_all)
          .mark_bar(size=max(40, bar_size))
          .encode(
              x=alt.X("bucket_label:N", title="Период (по A)", sort=bucket_order),
              y=alt.Y("sum(count):Q", title="Кол-во упоминаний"),
              color=alt.Color("aspect:N", title="Аспект (EN в скобках)", sort=present)
          )
    )

    # ---------- подготовка данных для «умного» тултипа (одна строка на период)
    wide = (aspects_all
            .pivot_table(index=["bucket","bucket_label"],
                         columns="aspect", values="count",
                         aggfunc="sum", fill_value=0))

    # добавим отсутствующие столбцы (чтобы всегда был фиксированный порядок)
    for lbl in expected_labels:
        if lbl not in wide.columns:
            wide[lbl] = 0

    # порядок колонок
    wide = wide[expected_labels]

    # итоги и проценты
    wide["total"] = wide.sum(axis=1)
    for i, lbl in enumerate(expected_labels):
        col_pct = f"pct_{i}"
        with pd.option_context('mode.use_inf_as_na', True):
            wide[col_pct] = (wide[lbl] / wide["total"]).fillna(0.0)

    tooltip_df = wide.reset_index()  # bucket, bucket_label, <аспекты...>, total, pct_*

    # поля тултипа: период, всего, затем пары «аспект — кол-во/процент»
    tooltip_fields = [
        alt.Tooltip(field="bucket_label", type="nominal", title="Период"),
        alt.Tooltip(field="total", type="quantitative", title="Всего упоминаний"),
    ]
    for i, lbl in enumerate(expected_labels):
        tooltip_fields.append(alt.Tooltip(field=lbl, type="quantitative", title=f"{lbl} — кол-во"))
        tooltip_fields.append(alt.Tooltip(field=f"pct_{i}", type="quantitative",
                                          title=f"{lbl} — %", format=".0%"))

    # прозрачный слой поверх столбца — один «большой» тултип по всему столбцу
    bubble = (
        alt.Chart(tooltip_df)
          .mark_bar(size=max(40, bar_size), opacity=0.001)
          .encode(
              x=alt.X("bucket_label:N", sort=bucket_order),
              y=alt.Y("total:Q"),
              tooltip=tooltip_fields
          )
    )

    st.altair_chart((bars + bubble).properties(height=460), use_container_width=True)

# неизвестные упоминания (не из шаблона)
st.markdown("#### Упоминания вне шаблона")
if unknown_all.empty:
    st.success("Все упоминания соответствуют шаблону.")
else:
    unknown_agg = (unknown_all.groupby(["mention","en"], as_index=False)
                              .agg(total=("total","sum"))
                              .sort_values("total", ascending=False))
    st.dataframe(unknown_agg, use_container_width=True)

