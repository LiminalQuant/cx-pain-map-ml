import io
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="CX Pain Map: NPS + CRM",
    page_icon="🧭",
    layout="wide",
)

NONE = "— не использовать —"
MIN_TEXT_LEN = 4

PAIN_TAXONOMY: dict[str, dict[str, Any]] = {
    "access": {
        "label": "Доступность / запись",
        "patterns": [
            "не дозвон", "дозвон", "не могу запис", "нет записи", "нет слотов",
            "слот", "записаться", "перезапис", "отменили запись", "отмена записи",
            "запись пропала", "не записали", "колл центр", "call", "звонок",
        ],
    },
    "waiting": {
        "label": "Ожидание / очередь",
        "patterns": [
            "ждал", "ждала", "ждать", "ожидание", "очеред", "задерж", "долго",
            "40 минут", "30 минут", "час", "время ожидания", "прием задержали",
        ],
    },
    "front_office": {
        "label": "Регистратура / администраторы",
        "patterns": [
            "регистрат", "администрат", "ресепш", "reception", "стойка", "касса",
            "хам", "груб", "не объяснили", "не подсказали", "не помогли",
        ],
    },
    "digital": {
        "label": "Цифровой путь / приложение",
        "patterns": [
            "прилож", "смартмед", "smartmed", "личный кабинет", "лк", "сайт",
            "ошибка", "не работает", "не открывается", "завис", "не груз", "оплата в приложении",
            "код", "смс", "sms", "авторизац", "пароль",
        ],
    },
    "payment": {
        "label": "Оплата / ДМС / финансы",
        "patterns": [
            "оплат", "дмс", "страхов", "страховая", "счет", "счёт", "возврат",
            "деньги", "касс", "чек", "стоимость", "цена", "платн", "гарантийное письмо",
        ],
    },
    "medical_process": {
        "label": "Медицинский процесс",
        "patterns": [
            "врач", "доктор", "назначен", "назначил", "лечение", "диагноз",
            "анализ", "результат", "заключение", "обследование", "прием", "приём",
            "невниматель", "осмотр", "консультац", "повторный прием", "повторный приём",
        ],
    },
    "documents": {
        "label": "Документы / справки",
        "patterns": [
            "справк", "выписк", "документ", "договор", "акт", "заключение",
            "печать", "подпись", "больничн", "лист нетрудоспособности", "направление",
        ],
    },
    "navigation": {
        "label": "Навигация / логистика",
        "patterns": [
            "не наш", "куда идти", "кабинет", "этаж", "адрес", "парков", "вход",
            "указател", "навигац", "добраться", "проход", "корпус",
        ],
    },
    "communication": {
        "label": "Коммуникация / информирование",
        "patterns": [
            "не сообщили", "не предупред", "не объясн", "информац", "уведомлен",
            "обратная связь", "перезвон", "не перезвони", "сказали", "никто не", "молчание",
        ],
    },
    "staff_attitude": {
        "label": "Отношение персонала",
        "patterns": [
            "хам", "груб", "равнодуш", "невеж", "неуваж", "тон", "разговаривали",
            "отношение", "некомпетент", "агрессив", "отказались помочь",
        ],
    },
}

TOPIC_LABELS = {code: spec["label"] for code, spec in PAIN_TAXONOMY.items()}
LABEL_TO_CODE = {label: code for code, label in TOPIC_LABELS.items()}


# =========================================================
# LOW-LEVEL HELPERS
# =========================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\xa0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", case=False, regex=True)]
    return df


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "нат", "нет", "-"}:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def meaningful_text(text: Any) -> bool:
    text = clean_text(text)
    if len(text) < MIN_TEXT_LEN:
        return False
    low = text.lower().strip()
    trash = {
        "да", "нет", "ок", "норм", "норма", "хорошо", "спасибо", "отлично",
        "плохо", "ужас", "комментариев нет", "без комментариев", "все хорошо",
    }
    if low in trash:
        return False
    if re.fullmatch(r"[-+]?\d+([.,]\d+)?", low):
        return False
    return True


def detect_column(columns: list[str], candidates: list[str], fallback: str | None = None) -> str | None:
    lower_map = {c.lower().strip(): c for c in columns}

    for candidate in candidates:
        key = candidate.lower().strip()
        if key in lower_map:
            return lower_map[key]

    for col in columns:
        low = col.lower().strip()
        for candidate in candidates:
            if candidate.lower().strip() in low:
                return col

    return fallback


def select_col(label: str, columns: list[str], candidates: list[str], *, optional: bool = False, key: str) -> str | None:
    options = [NONE] + columns if optional else columns
    detected = detect_column(columns, candidates)
    default_value = detected if detected in columns else (NONE if optional else columns[0])
    index = options.index(default_value) if default_value in options else 0
    selected = st.selectbox(label, options, index=index, key=key)
    if selected == NONE:
        return None
    return selected


def get_excel_sheets(uploaded_file) -> list[str]:
    uploaded_file.seek(0)
    xls = pd.ExcelFile(uploaded_file)
    uploaded_file.seek(0)
    return xls.sheet_names


def read_uploaded_table(uploaded_file, sheet_name: str | None = None) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()
    uploaded_file.seek(0)

    if file_name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine="python")
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
    elif file_name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name or 0)
    else:
        raise ValueError("Поддерживаются только CSV/XLSX/XLS")

    return normalize_columns(df)


def make_period(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "День":
        return df["date"].dt.strftime("%Y-%m-%d")
    if mode == "Неделя":
        iso = df["date"].dt.isocalendar()
        return iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    return df["date"].dt.strftime("%Y-%m")


def to_excel_bytes(dfs: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in dfs.items():
            safe_name = re.sub(r"[\[\]\*\:/\\\?]", "_", str(sheet_name))[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    buffer.seek(0)
    return buffer.getvalue()


# =========================================================
# PAIN TAGGING
# =========================================================

def tag_text(text: Any) -> list[str]:
    low = clean_text(text).lower()
    if not low:
        return []

    tags: list[str] = []
    for code, spec in PAIN_TAXONOMY.items():
        for pattern in spec["patterns"]:
            if pattern.lower() in low:
                tags.append(code)
                break
    return tags


def add_pain_tags(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    out = df.copy()
    out["pain_codes"] = out[text_col].apply(tag_text)
    out["pain_labels"] = out["pain_codes"].apply(lambda xs: [TOPIC_LABELS[x] for x in xs])
    out["pain_count"] = out["pain_codes"].apply(len)
    out["has_pain"] = out["pain_count"] > 0
    out["primary_pain_code"] = out["pain_codes"].apply(lambda xs: xs[0] if xs else "unclassified")
    out["primary_pain"] = out["primary_pain_code"].map({**TOPIC_LABELS, "unclassified": "Не классифицировано"})

    for code in PAIN_TAXONOMY:
        out[f"pain__{code}"] = out["pain_codes"].apply(lambda xs, c=code: c in xs)

    return out


def explode_pains(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=list(df.columns) + ["pain_code", "pain_label"])

    work = df.copy()
    work["pain_code"] = work["pain_codes"].apply(lambda xs: xs if xs else ["unclassified"])
    work = work.explode("pain_code")
    work["pain_label"] = work["pain_code"].map({**TOPIC_LABELS, "unclassified": "Не классифицировано"})
    return work


# =========================================================
# NORMALIZERS
# =========================================================

def prepare_nps_feedback(
    raw: pd.DataFrame,
    *,
    date_col: str,
    text_col: str,
    clinic_col: str | None,
    segment_col: str | None,
    score_col: str | None,
    question_col: str | None,
    answer_option_col: str | None,
) -> pd.DataFrame:
    df = raw.copy()
    out = pd.DataFrame()

    out["date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)

    main_text = df[text_col].apply(clean_text)
    if answer_option_col:
        option_text = df[answer_option_col].apply(clean_text)
        out["text"] = [
            " | ".join([part for part in [opt, txt] if meaningful_text(part)])
            for opt, txt in zip(option_text, main_text)
        ]
        out["answer_option"] = option_text
    else:
        out["text"] = main_text
        out["answer_option"] = ""

    out["clinic"] = df[clinic_col].apply(clean_text) if clinic_col else "Не указано"
    out["question"] = df[question_col].apply(clean_text) if question_col else "Не указано"
    out["score"] = pd.to_numeric(df[score_col], errors="coerce") if score_col else pd.NA

    if segment_col:
        out["segment"] = df[segment_col].apply(clean_text)
    else:
        out["segment"] = np.where(
            pd.to_numeric(out["score"], errors="coerce") <= 6,
            "Критик",
            np.where(pd.to_numeric(out["score"], errors="coerce") <= 8, "Нейтрал", "Промоутер"),
        )

    segment_map = {
        "detractor": "Критик", "critic": "Критик", "критик": "Критик",
        "neutral": "Нейтрал", "нейтрал": "Нейтрал",
        "promoter": "Промоутер", "промоутер": "Промоутер",
    }
    out["segment"] = out["segment"].apply(lambda x: segment_map.get(str(x).strip().lower(), str(x).strip()))

    out["source"] = "NPS/CX"
    out["channel"] = "NPS/CX"
    out["status"] = "Не указано"
    out["crm_category"] = "Не указано"
    out["responsible_unit"] = "Не указано"

    out = out[out["date"].notna()].copy()
    out = out[out["text"].apply(meaningful_text)].copy()
    out["period_month"] = out["date"].dt.strftime("%Y-%m")
    return add_pain_tags(out)


def prepare_crm_feedback(
    raw: pd.DataFrame,
    *,
    date_col: str,
    text_col: str,
    clinic_col: str | None,
    channel_col: str | None,
    status_col: str | None,
    category_col: str | None,
    responsible_col: str | None,
) -> pd.DataFrame:
    df = raw.copy()
    out = pd.DataFrame()

    out["date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    out["text"] = df[text_col].apply(clean_text)
    out["clinic"] = df[clinic_col].apply(clean_text) if clinic_col else "Не указано"
    out["channel"] = df[channel_col].apply(clean_text) if channel_col else "Не указано"
    out["status"] = df[status_col].apply(clean_text) if status_col else "Не указано"
    out["crm_category"] = df[category_col].apply(clean_text) if category_col else "Не указано"
    out["responsible_unit"] = df[responsible_col].apply(clean_text) if responsible_col else "Не указано"

    out["source"] = "CRM"
    out["segment"] = "CRM"
    out["score"] = pd.NA
    out["question"] = "CRM обращение"
    out["answer_option"] = ""

    out = out[out["date"].notna()].copy()
    out = out[out["text"].apply(meaningful_text)].copy()
    out["period_month"] = out["date"].dt.strftime("%Y-%m")
    return add_pain_tags(out)


# =========================================================
# DEMO DATA
# =========================================================

def make_demo_data(seed: int = 42, rows: int = 260) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", "2026-05-31", freq="D")
    clinics = ["Белорусская", "Красная Пресня", "Мичуринский", "Отрадное", "Марьино"]
    channels = ["Телефон", "Приложение", "Сайт", "Личный визит", "Email"]

    pain_texts = [
        "Ждал прием почти час, очередь у регистратуры, никто не предупредил о задержке",
        "Не смог записаться через приложение, ошибка оплаты и зависание личного кабинета",
        "Страховая не подтвердила ДМС, на кассе долго разбирались со счетом",
        "Администратор грубо разговаривала и не объяснила куда идти",
        "Не пришли результаты анализов, врач не оставил нормальное заключение",
        "Не нашел кабинет, нет указателей, парковка неудобная",
        "Не дозвонился в колл центр, запись отменили без предупреждения",
        "Нужна была справка, документы готовили слишком долго",
        "Врач внимательно провел прием, но ожидание и очередь испортили впечатление",
        "СМС-код не приходит, приложение не открывается, оплатить невозможно",
    ]
    neutral_texts = [
        "В целом нормально, но ожидание было долгим",
        "К врачу претензий нет, но запись через сайт неудобная",
        "Не хватило информации по документам и дальнейшим действиям",
        "Все решили, но пришлось несколько раз звонить",
    ]
    good_texts = [
        "Все хорошо, врач помог", "Быстро приняли, спасибо", "Нормально", "Отличный прием",
    ]

    nps_rows = []
    for i in range(rows):
        score = int(rng.choice([2, 4, 5, 6, 7, 8, 9, 10], p=[0.05, 0.08, 0.09, 0.12, 0.16, 0.16, 0.18, 0.16]))
        if score <= 6:
            segment = "Критик"
            comment = rng.choice(pain_texts)
        elif score <= 8:
            segment = "Нейтрал"
            comment = rng.choice(neutral_texts)
        else:
            segment = "Промоутер"
            comment = rng.choice(good_texts)
        nps_rows.append({
            "Дата талона": rng.choice(dates),
            "Название клиники": rng.choice(clinics),
            "Тип респондента": segment,
            "Оценка": score,
            "Вопрос": rng.choice(["Оцените визит", "Оцените запись", "Оцените приложение", "Оцените ожидание"]),
            "Комментарий": comment,
            "Опция ответа": rng.choice(["", "Долго ждать", "Неудобная запись", "Проблема с оплатой", "Персонал"]),
        })

    crm_rows = []
    statuses = ["Закрыто", "В работе", "Повторно", "Эскалация"]
    categories = ["Жалоба", "Вопрос", "Претензия", "Благодарность", "Техническая проблема"]
    units = ["Регистратура", "Контакт-центр", "ДМС", "IT", "Врачи", "Документы"]
    for i in range(rows):
        text = rng.choice(pain_texts + neutral_texts)
        crm_rows.append({
            "Дата обращения": rng.choice(dates),
            "Клиника": rng.choice(clinics),
            "Канал": rng.choice(channels),
            "Статус": rng.choice(statuses, p=[0.55, 0.25, 0.12, 0.08]),
            "Категория": rng.choice(categories, p=[0.38, 0.22, 0.18, 0.07, 0.15]),
            "Ответственный блок": rng.choice(units),
            "Текст обращения": text,
        })

    return pd.DataFrame(nps_rows), pd.DataFrame(crm_rows)


# =========================================================
# ANALYTICS RENDERERS
# =========================================================

def filter_frame(
    df: pd.DataFrame,
    *,
    clinic: str,
    source: str | None = None,
    topic_label: str | None = None,
    channel: str | None = None,
    status: str | None = None,
    segment: list[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if clinic != "Все":
        out = out[out["clinic"] == clinic]
    if source and source != "Все":
        out = out[out["source"] == source]
    if channel and channel != "Все":
        out = out[out["channel"] == channel]
    if status and status != "Все":
        out = out[out["status"] == status]
    if segment:
        out = out[out["segment"].isin(segment)]
    if topic_label and topic_label != "Все":
        code = LABEL_TO_CODE.get(topic_label)
        if code:
            out = out[out["pain_codes"].apply(lambda xs: code in xs)]
        elif topic_label == "Не классифицировано":
            out = out[out["pain_codes"].apply(lambda xs: len(xs) == 0)]
    return out


def render_kpis(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Записей", f"{len(df):,}".replace(",", " "))
    c2.metric("Клиник", df["clinic"].nunique() if not df.empty else 0)
    c3.metric("С болью", f"{int(df['has_pain'].sum()) if not df.empty else 0:,}".replace(",", " "))
    share = (df["has_pain"].mean() * 100) if not df.empty else 0
    c4.metric("Доля классифицированных", f"{share:.1f}%")


def render_topic_chart(df: pd.DataFrame, title: str) -> pd.DataFrame:
    exploded = explode_pains(df)
    topic_counts = (
        exploded.groupby("pain_label")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    fig = px.bar(topic_counts.head(20), x="pain_label", y="count", title=title)
    st.plotly_chart(fig, use_container_width=True)
    return topic_counts


def render_timeline(df: pd.DataFrame, period_mode: str, title: str) -> pd.DataFrame:
    if df.empty:
        st.warning("Нет данных после фильтров")
        return pd.DataFrame()
    work = df.copy()
    work["period"] = make_period(work, period_mode)
    tl = (
        work.groupby(["period", "source"])
        .size()
        .reset_index(name="count")
        .sort_values("period")
    )
    fig = px.line(tl, x="period", y="count", color="source", markers=True, title=title)
    st.plotly_chart(fig, use_container_width=True)
    return tl


def render_heatmap(df: pd.DataFrame, row: str, title: str) -> pd.DataFrame:
    exploded = explode_pains(df)
    if exploded.empty:
        return pd.DataFrame()
    pivot = (
        exploded.pivot_table(index=row, columns="pain_label", values="text", aggfunc="count", fill_value=0)
        .reset_index()
    )
    long_df = exploded.groupby([row, "pain_label"]).size().reset_index(name="count")
    fig = px.density_heatmap(long_df, x="pain_label", y=row, z="count", title=title)
    st.plotly_chart(fig, use_container_width=True)
    return pivot


def render_evidence(df: pd.DataFrame, limit: int = 50) -> None:
    cols = [
        "date", "source", "clinic", "channel", "status", "segment", "question",
        "crm_category", "responsible_unit", "primary_pain", "pain_labels", "text",
    ]
    existing = [c for c in cols if c in df.columns]
    st.dataframe(
        df.sort_values("date", ascending=False)[existing].head(limit),
        use_container_width=True,
        height=460,
    )


def make_export_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    exploded = explode_pains(df)

    topic_summary = (
        exploded.groupby(["source", "pain_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["source", "count"], ascending=[True, False])
    )

    clinic_topic = (
        exploded.groupby(["source", "clinic", "pain_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["source", "clinic", "count"], ascending=[True, True, False])
    )

    monthly = df.copy()
    monthly["period"] = monthly["date"].dt.strftime("%Y-%m")
    monthly = monthly.groupby(["period", "source"]).size().reset_index(name="count").sort_values("period")

    return {
        "feedback_tagged": df.copy(),
        "feedback_exploded": exploded.copy(),
        "topic_summary": topic_summary,
        "clinic_topic": clinic_topic,
        "monthly": monthly,
    }


# =========================================================
# UI
# =========================================================

st.title("🧭 CX Pain Map: NPS + CRM")
st.caption("Единый слой анализа клиентской боли: NPS/CX + CRM обращения. Без LLM на первом тесте — только проверяем механику и управленческий контур.")

with st.sidebar:
    st.header("Режим")
    demo_mode = st.toggle("Использовать демо-данные", value=True)
    period_mode_global = st.selectbox("Период агрегации", ["Месяц", "Неделя", "День"], index=0)
    st.divider()
    st.write("**Таксономия болей**")
    st.dataframe(pd.DataFrame([
        {"code": code, "label": spec["label"], "patterns": ", ".join(spec["patterns"][:5]) + "..."}
        for code, spec in PAIN_TAXONOMY.items()
    ]), use_container_width=True, height=300)

if demo_mode:
    demo_nps, demo_crm = make_demo_data()
else:
    demo_nps, demo_crm = None, None

if "nps_processed" not in st.session_state:
    st.session_state["nps_processed"] = pd.DataFrame()
if "crm_processed" not in st.session_state:
    st.session_state["crm_processed"] = pd.DataFrame()


tab_nps, tab_crm, tab_combined, tab_evidence = st.tabs([
    "NPS / CX слой",
    "CRM слой",
    "Объединённая карта",
    "Evidence / Export",
])


# =========================================================
# TAB NPS
# =========================================================
with tab_nps:
    st.header("NPS / CX слой")

    if demo_mode:
        raw_nps = demo_nps.copy()
        st.info("Загружены демо-данные NPS/CX")
    else:
        uploaded_nps = st.file_uploader("Загрузите NPS/CX файл", type=["csv", "xlsx", "xls"], key="nps_uploader")
        raw_nps = None
        if uploaded_nps:
            sheet = None
            if uploaded_nps.name.lower().endswith((".xlsx", ".xls")):
                sheets = get_excel_sheets(uploaded_nps)
                sheet = st.selectbox("Лист Excel", sheets, key="nps_sheet")
            raw_nps = read_uploaded_table(uploaded_nps, sheet)

    if raw_nps is not None:
        st.subheader("Маппинг колонок")
        cols = raw_nps.columns.tolist()
        c1, c2 = st.columns(2)
        with c1:
            nps_date_col = select_col("Дата", cols, ["Дата талона", "Дата", "date", "created", "timestamp"], key="nps_date")
            nps_text_col = select_col("Текст / комментарий", cols, ["Комментарий", "Ответ", "Текст", "comment", "text"], key="nps_text")
            nps_clinic_col = select_col("Клиника", cols, ["Название клиники", "Клиника", "clinic", "branch"], optional=True, key="nps_clinic")
            nps_question_col = select_col("Вопрос", cols, ["Вопрос", "question"], optional=True, key="nps_question")
        with c2:
            nps_segment_col = select_col("Сегмент", cols, ["Тип респондента", "Сегмент", "segment"], optional=True, key="nps_segment")
            nps_score_col = select_col("Оценка", cols, ["Оценка", "score", "rating", "nps"], optional=True, key="nps_score")
            nps_option_col = select_col("Опция ответа", cols, ["Опция ответа", "answer_option", "option"], optional=True, key="nps_option")

        try:
            nps_df = prepare_nps_feedback(
                raw_nps,
                date_col=nps_date_col,
                text_col=nps_text_col,
                clinic_col=nps_clinic_col,
                segment_col=nps_segment_col,
                score_col=nps_score_col,
                question_col=nps_question_col,
                answer_option_col=nps_option_col,
            )
            st.session_state["nps_processed"] = nps_df
        except Exception as exc:
            st.error(f"Ошибка подготовки NPS: {exc}")
            nps_df = pd.DataFrame()

        if not nps_df.empty:
            st.subheader("Срез")
            clinics = ["Все"] + sorted(nps_df["clinic"].dropna().unique().tolist())
            segments = sorted(nps_df["segment"].dropna().unique().tolist())
            f1, f2 = st.columns(2)
            with f1:
                selected_clinic = st.selectbox("Клиника", clinics, key="nps_filter_clinic")
            with f2:
                selected_segments = st.multiselect("Сегменты", segments, default=[x for x in segments if x in ["Критик", "Нейтрал", "CRM"]] or segments, key="nps_filter_segment")

            nps_filtered = filter_frame(nps_df, clinic=selected_clinic, segment=selected_segments)
            render_kpis(nps_filtered)

            st.subheader("Динамика")
            nps_timeline = render_timeline(nps_filtered, period_mode_global, "Динамика NPS/CX записей")

            st.subheader("Темы болей")
            nps_topic_counts = render_topic_chart(nps_filtered, "Топ болей NPS/CX")

            st.subheader("Клиника × боль")
            nps_heatmap = render_heatmap(nps_filtered, "clinic", "NPS/CX: распределение болей по клиникам")

            with st.expander("Примеры / evidence"):
                render_evidence(nps_filtered)

            with st.expander("Сырые нормализованные данные"):
                st.dataframe(nps_df.head(200), use_container_width=True)
        else:
            st.warning("После подготовки NPS-слоя данных нет. Проверь дату и текстовую колонку.")


# =========================================================
# TAB CRM
# =========================================================
with tab_crm:
    st.header("CRM слой")
    st.write("CRM здесь не притворяется NPS. Это отдельный операционный сигнал: обращения, статусы, каналы, категории и фактические боли.")

    if demo_mode:
        raw_crm = demo_crm.copy()
        st.info("Загружены демо-данные CRM")
    else:
        uploaded_crm = st.file_uploader("Загрузите CRM файл", type=["csv", "xlsx", "xls"], key="crm_uploader")
        raw_crm = None
        if uploaded_crm:
            sheet = None
            if uploaded_crm.name.lower().endswith((".xlsx", ".xls")):
                sheets = get_excel_sheets(uploaded_crm)
                sheet = st.selectbox("Лист Excel", sheets, key="crm_sheet")
            raw_crm = read_uploaded_table(uploaded_crm, sheet)

    if raw_crm is not None:
        st.subheader("Маппинг колонок")
        cols = raw_crm.columns.tolist()
        c1, c2 = st.columns(2)
        with c1:
            crm_date_col = select_col("Дата обращения", cols, ["Дата обращения", "Дата", "created", "date", "timestamp"], key="crm_date")
            crm_text_col = select_col("Текст обращения", cols, ["Текст обращения", "Описание", "Комментарий", "Текст", "message", "text"], key="crm_text")
            crm_clinic_col = select_col("Клиника / филиал", cols, ["Клиника", "Филиал", "Название клиники", "clinic", "branch"], optional=True, key="crm_clinic")
            crm_channel_col = select_col("Канал", cols, ["Канал", "channel", "source"], optional=True, key="crm_channel")
        with c2:
            crm_status_col = select_col("Статус", cols, ["Статус", "status"], optional=True, key="crm_status")
            crm_category_col = select_col("Категория CRM", cols, ["Категория", "Тема", "Тип", "category", "topic"], optional=True, key="crm_category")
            crm_responsible_col = select_col("Ответственный блок", cols, ["Ответственный блок", "Ответственный", "Подразделение", "unit", "owner"], optional=True, key="crm_responsible")

        try:
            crm_df = prepare_crm_feedback(
                raw_crm,
                date_col=crm_date_col,
                text_col=crm_text_col,
                clinic_col=crm_clinic_col,
                channel_col=crm_channel_col,
                status_col=crm_status_col,
                category_col=crm_category_col,
                responsible_col=crm_responsible_col,
            )
            st.session_state["crm_processed"] = crm_df
        except Exception as exc:
            st.error(f"Ошибка подготовки CRM: {exc}")
            crm_df = pd.DataFrame()

        if not crm_df.empty:
            st.subheader("Срез")
            clinics = ["Все"] + sorted(crm_df["clinic"].dropna().unique().tolist())
            channels = ["Все"] + sorted(crm_df["channel"].dropna().unique().tolist())
            statuses = ["Все"] + sorted(crm_df["status"].dropna().unique().tolist())
            f1, f2, f3 = st.columns(3)
            with f1:
                selected_clinic = st.selectbox("Клиника", clinics, key="crm_filter_clinic")
            with f2:
                selected_channel = st.selectbox("Канал", channels, key="crm_filter_channel")
            with f3:
                selected_status = st.selectbox("Статус", statuses, key="crm_filter_status")

            crm_filtered = filter_frame(
                crm_df,
                clinic=selected_clinic,
                channel=selected_channel,
                status=selected_status,
            )
            render_kpis(crm_filtered)

            st.subheader("Динамика")
            crm_timeline = render_timeline(crm_filtered, period_mode_global, "Динамика CRM-обращений")

            st.subheader("Темы болей")
            crm_topic_counts = render_topic_chart(crm_filtered, "Топ болей CRM")

            st.subheader("Канал × боль")
            crm_channel_heatmap = render_heatmap(crm_filtered, "channel", "CRM: распределение болей по каналам")

            st.subheader("Ответственный блок × боль")
            crm_unit_heatmap = render_heatmap(crm_filtered, "responsible_unit", "CRM: распределение болей по ответственным блокам")

            with st.expander("Примеры / evidence"):
                render_evidence(crm_filtered)

            with st.expander("Сырые нормализованные данные"):
                st.dataframe(crm_df.head(200), use_container_width=True)
        else:
            st.warning("После подготовки CRM-слоя данных нет. Проверь дату и текстовую колонку.")


# =========================================================
# TAB COMBINED
# =========================================================
with tab_combined:
    st.header("Объединённая карта боли")

    frames = []
    if not st.session_state["nps_processed"].empty:
        frames.append(st.session_state["nps_processed"])
    if not st.session_state["crm_processed"].empty:
        frames.append(st.session_state["crm_processed"])

    if frames:
        combined = pd.concat(frames, ignore_index=True)

        clinics = ["Все"] + sorted(combined["clinic"].dropna().unique().tolist())
        sources = ["Все"] + sorted(combined["source"].dropna().unique().tolist())
        topics = ["Все"] + [TOPIC_LABELS[x] for x in PAIN_TAXONOMY] + ["Не классифицировано"]

        f1, f2, f3 = st.columns(3)
        with f1:
            selected_clinic = st.selectbox("Клиника", clinics, key="combined_clinic")
        with f2:
            selected_source = st.selectbox("Источник", sources, key="combined_source")
        with f3:
            selected_topic = st.selectbox("Боль", topics, key="combined_topic")

        combined_filtered = filter_frame(
            combined,
            clinic=selected_clinic,
            source=selected_source,
            topic_label=selected_topic,
        )

        render_kpis(combined_filtered)

        st.subheader("Источник × боль")
        source_topic = render_heatmap(combined_filtered, "source", "Распределение болей по источникам")

        st.subheader("Динамика по источникам")
        combined_timeline = render_timeline(combined_filtered, period_mode_global, "NPS/CX + CRM: динамика сигналов")

        st.subheader("Клиника × боль")
        combined_clinic = render_heatmap(combined_filtered, "clinic", "Общая карта болей по клиникам")

        st.subheader("Топ болей")
        combined_topics = render_topic_chart(combined_filtered, "Топ болей во всех источниках")

        st.subheader("Что совпадает между NPS и CRM")
        exploded = explode_pains(combined_filtered)
        overlap = (
            exploded.groupby(["pain_label", "source"])
            .size()
            .reset_index(name="count")
        )
        overlap_pivot = overlap.pivot_table(index="pain_label", columns="source", values="count", fill_value=0).reset_index()
        if {"NPS/CX", "CRM"}.issubset(set(overlap_pivot.columns)):
            overlap_pivot["total"] = overlap_pivot["NPS/CX"] + overlap_pivot["CRM"]
            overlap_pivot["crm_share"] = overlap_pivot["CRM"] / overlap_pivot["total"].replace(0, np.nan)
        st.dataframe(overlap_pivot.sort_values(overlap_pivot.columns[-1], ascending=False), use_container_width=True)

        with st.expander("Evidence по объединённой карте"):
            render_evidence(combined_filtered, limit=100)
    else:
        st.warning("Сначала загрузите или сгенерируйте NPS/CRM данные.")


# =========================================================
# TAB EVIDENCE / EXPORT
# =========================================================
with tab_evidence:
    st.header("Evidence / Export")

    frames = []
    if not st.session_state["nps_processed"].empty:
        frames.append(st.session_state["nps_processed"])
    if not st.session_state["crm_processed"].empty:
        frames.append(st.session_state["crm_processed"])

    if frames:
        combined = pd.concat(frames, ignore_index=True)

        clinics = ["Все"] + sorted(combined["clinic"].dropna().unique().tolist())
        sources = ["Все"] + sorted(combined["source"].dropna().unique().tolist())
        topics = ["Все"] + [TOPIC_LABELS[x] for x in PAIN_TAXONOMY] + ["Не классифицировано"]

        f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
        with f1:
            selected_clinic = st.selectbox("Клиника", clinics, key="evidence_clinic")
        with f2:
            selected_source = st.selectbox("Источник", sources, key="evidence_source")
        with f3:
            selected_topic = st.selectbox("Боль", topics, key="evidence_topic")
        with f4:
            limit = st.number_input("Лимит примеров", min_value=10, max_value=500, value=100, step=10)

        evidence_df = filter_frame(
            combined,
            clinic=selected_clinic,
            source=selected_source,
            topic_label=selected_topic,
        )

        st.subheader("Примеры")
        render_evidence(evidence_df, limit=int(limit))

        st.subheader("Экспорт")
        export_frames = make_export_frames(combined)
        excel_bytes = to_excel_bytes(export_frames)
        st.download_button(
            "Скачать Excel с анализом",
            data=excel_bytes,
            file_name="cx_pain_map_nps_crm.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        with st.expander("Состав выгрузки"):
            st.write(list(export_frames.keys()))
    else:
        st.warning("Нет данных для evidence/export.")
