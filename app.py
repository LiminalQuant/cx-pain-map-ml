import io
import re
import json
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================

API_KEY = st.secrets["OLLAMA_API_KEY"].strip()
URL = st.secrets["OLLAMA_URL"].strip()
MODEL = "gpt-oss:120b"

st.set_page_config(
    page_title="CX Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# TABS
# =========================================================

tab1, tab2 = st.tabs(["📊 Root Cause Analytics", "🎯 Pain Map ML"])

# =========================================================
# COMMON HELPERS
# =========================================================

def to_excel_bytes(dfs: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in dfs.items():
            safe_name = re.sub(r"[\[\]\*\:/\\\?]", "_", str(sheet_name))[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    buffer.seek(0)
    return buffer.getvalue()


# =========================================================
# TAB 1: ROOT CAUSE ANALYTICS
# =========================================================

with tab1:
    # =====================================================
    # SCHEMAS / CONSTANTS
    # =====================================================

    CX_SCHEMA_FULL = {
        "Дата талона": "date",
        "ЭМК пациента": "patient_id",
        "Название клиники": "clinic",
        "Категория клиники": "clinic_category",
        "Регоион клиники": "clinic_region",
        "Тип респондента": "segment",
        "Вопрос": "question",
        "Ответ": "answer",
        "Опция ответа": "answer_option",
    }

    CX_SCHEMA_COMMENT = {
        "Дата талона": "date",
        "ЭМК пациента": "patient_id",
        "Название клиники": "clinic",
        "Тип респондента": "segment",
        "Вопрос": "question",
        "Оценка": "score",
        "Комментарий": "answer",
    }

    TARGET_SEGMENTS = ["Критик", "Нейтрал"]

    # =====================================================
    # HELPERS
    # =====================================================

    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = (
            df.columns.astype(str)
            .str.replace("\xa0", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        return df

    def detect_cx_schema(df: pd.DataFrame) -> str:
        cols = set(df.columns.astype(str).str.strip())

        full_required = set(CX_SCHEMA_FULL.keys())
        comment_required = set(CX_SCHEMA_COMMENT.keys())

        if full_required.issubset(cols):
            return "full"

        if comment_required.issubset(cols):
            return "comment"

        return "unknown"

    def clean_text_for_llm(x) -> str:
        if x is None:
            return ""
        text = str(x).strip()
        if text.lower() in {"nan", "none", "null"}:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def build_llm_text(answer: str, answer_option: str) -> str:
        answer = clean_text_for_llm(answer)
        answer_option = clean_text_for_llm(answer_option)

        parts = []

        if answer_option and answer_option.lower() not in {"-", "без ответа", "nan", "none"}:
            parts.append(f"Опция ответа: {answer_option}")

        if answer and answer.lower() not in {"-", "без ответа", "nan", "none"}:
            parts.append(f"Комментарий: {answer}")

        return "\n".join(parts).strip()

    def is_meaningful_text(value: str) -> bool:
        if value is None:
            return False

        text = str(value).strip()
        if not text:
            return False

        low = text.lower()

        trash = {
            "да", "нет", "ок", "норм", "норма", "хорошо", "спасибо",
            "отлично", "плохо", "ужас", "комментариев нет"
        }
        if low in trash:
            return False

        if re.fullmatch(r"[-+]?\d+([.,]\d+)?", text):
            return False

        words = re.findall(r"\w+", low, flags=re.UNICODE)

        if len(words) < 2:
            return False

        if len(text) < 12:
            return False

        return True

    def is_meaningful_combined_text(answer: str, answer_option: str) -> bool:
        combined = build_llm_text(answer, answer_option)

        if not combined:
            return False

        low = combined.lower()

        trash = {
            "да", "нет", "ок", "норм", "норма", "хорошо", "спасибо",
            "отлично", "плохо", "ужас", "комментариев нет"
        }
        if low in trash:
            return False

        if re.fullmatch(r"[-+]?\d+([.,]\d+)?", combined):
            return False

        words = re.findall(r"\w+", low, flags=re.UNICODE)

        if len(words) < 2:
            return False

        if len(combined) < 10:
            return False

        return True

    def read_uploaded_table(uploaded_file) -> pd.DataFrame:
        """
        Пробуем header=1 и header=0, выбираем тот вариант, где лучше распознается схема.
        Для CSV тоже делаем fallback.
        """
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".csv"):
            uploaded_file.seek(0)
            df0 = pd.read_csv(uploaded_file)
            df0 = normalize_columns(df0)

            if detect_cx_schema(df0) != "unknown":
                return df0

            uploaded_file.seek(0)
            df1 = pd.read_csv(uploaded_file, header=1)
            df1 = normalize_columns(df1)

            if detect_cx_schema(df1) != "unknown":
                return df1

            return df0

        candidates = []

        for header_row in [1, 0]:
            try:
                uploaded_file.seek(0)
                dfx = pd.read_excel(uploaded_file, header=header_row)
                dfx = normalize_columns(dfx)
                dfx = dfx.loc[:, ~dfx.columns.str.contains(r"^Unnamed", case=False, regex=True)]
                schema = detect_cx_schema(dfx)
                score = 1 if schema != "unknown" else 0
                candidates.append((score, header_row, dfx))
            except Exception:
                continue

        if not candidates:
            raise ValueError("Не удалось прочитать файл как Excel/CSV")

        candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        return candidates[0][2]

    def load_file(uploaded_file) -> pd.DataFrame:
        """
        Поддержка двух CX-схем:

        1) Полная:
           Дата талона / ЭМК пациента / Название клиники / Категория клиники /
           Регоион клиники / Тип респондента / Вопрос / Ответ / Опция ответа

        2) Упрощенная:
           Дата талона / Название клиники / ЭМК пациента / Тип респондента /
           Вопрос / Оценка / Комментарий

        На выходе:
        date, patient_id, clinic, clinic_category, clinic_region,
        segment, question, answer, answer_option, score, llm_text, source_schema
        """
        df = read_uploaded_table(uploaded_file)
        df = normalize_columns(df)
        df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, regex=True)]

        schema_type = detect_cx_schema(df)

        if schema_type == "full":
            df = df.rename(columns=CX_SCHEMA_FULL)
            if "score" not in df.columns:
                df["score"] = pd.NA

        elif schema_type == "comment":
            df = df.rename(columns=CX_SCHEMA_COMMENT)

            if "clinic_category" not in df.columns:
                df["clinic_category"] = "Не указано"

            if "clinic_region" not in df.columns:
                df["clinic_region"] = "Не указано"

            if "answer_option" not in df.columns:
                df["answer_option"] = ""

        else:
            raise ValueError(
                "Формат файла не распознан.\n\n"
                "Ожидался один из двух CX-форматов:\n"
                "1) Полный: Дата талона, ЭМК пациента, Название клиники, Категория клиники, "
                "Регоион клиники, Тип респондента, Вопрос, Ответ, Опция ответа\n"
                "2) Упрощенный: Дата талона, Название клиники, ЭМК пациента, Тип респондента, "
                "Вопрос, Оценка, Комментарий\n\n"
                f"Фактические колонки: {list(df.columns)}"
            )

        required_internal = [
            "date", "patient_id", "clinic", "clinic_category", "clinic_region",
            "segment", "question", "answer", "answer_option", "score"
        ]

        for col in required_internal:
            if col not in df.columns:
                if col == "answer_option":
                    df[col] = ""
                elif col == "score":
                    df[col] = pd.NA
                elif col in ["clinic_category", "clinic_region"]:
                    df[col] = "Не указано"
                else:
                    df[col] = ""

        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        df["patient_id"] = df["patient_id"].astype(str).str.strip()
        df["clinic"] = df["clinic"].astype(str).str.strip()
        df["clinic_category"] = df["clinic_category"].astype(str).str.strip()
        df["clinic_region"] = df["clinic_region"].astype(str).str.strip()
        df["segment"] = df["segment"].astype(str).str.strip()
        df["question"] = df["question"].astype(str).str.strip()
        df["answer"] = df["answer"].astype(str).str.strip()
        df["answer_option"] = df["answer_option"].astype(str).str.strip()
        df["score"] = pd.to_numeric(df["score"], errors="coerce")

        df = df[df["date"].notna()].copy()

        segment_map = {
            "detractor": "Критик",
            "neutral": "Нейтрал",
            "promoter": "Промоутер",
            "критик": "Критик",
            "нейтрал": "Нейтрал",
            "промоутер": "Промоутер",
        }

        df["segment"] = df["segment"].apply(
            lambda x: segment_map.get(str(x).strip().lower(), str(x).strip())
        )

        df["llm_text"] = df.apply(
            lambda row: build_llm_text(row["answer"], row["answer_option"]),
            axis=1
        )

        iso = df["date"].dt.isocalendar()
        df["year"] = df["date"].dt.year
        df["month_num"] = df["date"].dt.month
        df["month_name"] = df["date"].dt.strftime("%Y-%m")
        df["day"] = df["date"].dt.date.astype(str)
        df["iso_week"] = iso.week.astype(int)
        df["iso_year"] = iso.year.astype(int)
        df["year_week"] = (
            df["iso_year"].astype(str) + "-W" + df["iso_week"].astype(str).str.zfill(2)
        )

        df["source_schema"] = schema_type

        return df

    def ask_llm_json(prompt: str) -> dict:
        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты CX-аналитик. "
                        "Отвечай строго JSON без markdown и без пояснений вне JSON."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        r = requests.post(URL, headers=headers, json=payload, timeout=120)

        if r.status_code != 200:
            st.error(f"STATUS: {r.status_code}")
            st.error(r.text)
            raise Exception("LLM request failed")

        data = r.json()
        content = data["message"]["content"].strip()

        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            return {
                "root_causes": [],
                "summary": "LLM не вернула валидный JSON",
                "confidence_note": content[:1000]
            }

        try:
            return json.loads(match.group())
        except Exception:
            return {
                "root_causes": [],
                "summary": "Не удалось распарсить JSON",
                "confidence_note": content[:1000]
            }

    def build_negative_base(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["segment"].isin(TARGET_SEGMENTS)].copy()

    def segment_summary(df_neg: pd.DataFrame) -> pd.DataFrame:
        out = (
            df_neg.groupby(["clinic", "segment"])
            .size()
            .reset_index(name="count")
            .pivot(index="clinic", columns="segment", values="count")
            .fillna(0)
            .reset_index()
        )

        for col in TARGET_SEGMENTS:
            if col not in out.columns:
                out[col] = 0

        out["total_negative"] = out[TARGET_SEGMENTS].sum(axis=1)
        out = out.sort_values("total_negative", ascending=False)
        return out

    def question_summary(df_neg: pd.DataFrame) -> pd.DataFrame:
        grp = (
            df_neg.groupby(["clinic", "question", "segment"])
            .size()
            .reset_index(name="count")
        )

        totals = (
            grp.groupby(["clinic", "segment"])["count"]
            .sum()
            .reset_index(name="segment_total")
        )

        grp = grp.merge(totals, on=["clinic", "segment"], how="left")
        grp["share_in_segment"] = grp["count"] / grp["segment_total"]
        grp = grp.sort_values(["clinic", "segment", "count"], ascending=[True, True, False])
        return grp

    def timeline_summary(df_neg: pd.DataFrame, freq: str) -> pd.DataFrame:
        time_col = "year_week" if freq == "Неделя" else "month_name"

        grp = (
            df_neg.groupby([time_col, "clinic", "segment"])
            .size()
            .reset_index(name="count")
            .sort_values([time_col, "clinic", "segment"])
        )
        return grp

    def day_breakdown_for_month(df_neg: pd.DataFrame, month_value: str) -> pd.DataFrame:
        sub = df_neg[df_neg["month_name"] == month_value].copy()
        grp = (
            sub.groupby(["day", "clinic", "segment"])
            .size()
            .reset_index(name="count")
            .sort_values(["day", "clinic", "segment"])
        )
        return grp

    def patient_transition_analysis(df_neg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        x = df_neg[["patient_id", "date", "clinic", "segment"]].copy()
        x = x.sort_values(["patient_id", "date"])

        seq = (
            x.groupby("patient_id")
            .agg(
                first_date=("date", "min"),
                last_date=("date", "max"),
                first_segment=("segment", "first"),
                last_segment=("segment", "last"),
                touchpoints=("segment", "size"),
                clinic_n=("clinic", "nunique"),
            )
            .reset_index()
        )

        seq["changed_segment"] = seq["first_segment"] != seq["last_segment"]

        def label_transition(row):
            a = row["first_segment"]
            b = row["last_segment"]
            if a == "Нейтрал" and b == "Критик":
                return "Нейтрал → Критик"
            if a == "Критик" and b == "Нейтрал":
                return "Критик → Нейтрал"
            if a == b == "Критик":
                return "Стабильно Критик"
            if a == b == "Нейтрал":
                return "Стабильно Нейтрал"
            return f"{a} → {b}"

        seq["transition"] = seq.apply(label_transition, axis=1)

        transition_summary = (
            seq.groupby("transition")
            .size()
            .reset_index(name="patient_count")
            .sort_values("patient_count", ascending=False)
        )

        return seq, transition_summary

    def build_root_cause_payload(
        df_neg: pd.DataFrame,
        clinic_scope: str,
        question_limit: int = 8,
        sample_per_question: int = 20
    ) -> pd.DataFrame:
        counts = (
            df_neg.groupby(["clinic", "question"])
            .size()
            .reset_index(name="negative_count")
        )

        counts = counts.sort_values(["clinic", "negative_count"], ascending=[True, False])

        payload_rows = []

        clinics = counts["clinic"].unique().tolist()
        for clinic in clinics:
            top_q = counts[counts["clinic"] == clinic].head(question_limit)

            for _, row in top_q.iterrows():
                clinic_name = row["clinic"]
                question = row["question"]
                cnt = int(row["negative_count"])

                mask = (
                    (df_neg["clinic"] == clinic_name) &
                    (df_neg["question"] == question)
                )

                comments = df_neg.loc[mask].copy()
                comments = comments[
                    comments.apply(
                        lambda x: is_meaningful_combined_text(x["answer"], x["answer_option"]),
                        axis=1
                    )
                ]["llm_text"].dropna().astype(str).tolist()

                comments = comments[:sample_per_question]

                payload_rows.append({
                    "clinic": clinic_name,
                    "question": question,
                    "negative_count": cnt,
                    "text_count": len(comments),
                    "comments": comments
                })

        return pd.DataFrame(payload_rows)

    def run_root_cause_llm(
        df_neg: pd.DataFrame,
        selected_clinic: str,
        max_questions_per_clinic: int = 6,
        sample_per_question: int = 15
    ) -> pd.DataFrame:
        if selected_clinic != "Все клиники":
            df_work_local = df_neg[df_neg["clinic"] == selected_clinic].copy()
        else:
            df_work_local = df_neg.copy()

        payload_df = build_root_cause_payload(
            df_work_local,
            clinic_scope=selected_clinic,
            question_limit=max_questions_per_clinic,
            sample_per_question=sample_per_question
        )

        results = []

        for clinic, sub in payload_df.groupby("clinic"):
            blocks = []

            for _, row in sub.iterrows():
                if row["text_count"] == 0:
                    continue

                comments_block = json.dumps(row["comments"], ensure_ascii=False)
                blocks.append(
                    f"Вопрос: {row['question']}\n"
                    f"Количество негативных ответов: {row['negative_count']}\n"
                    f"Комментарии (JSON массив строк):\n{comments_block}\n"
                )

            if not blocks:
                results.append({
                    "clinic": clinic,
                    "summary": "Нет достаточного объема текстовых комментариев для root cause extraction.",
                    "root_causes_json": json.dumps({"root_causes": []}, ensure_ascii=False)
                })
                continue

            prompt = f"""
Ниже данные по негативным ответам пациентов для клиники.

Клиника: {clinic}

Для каждого вопроса есть набор текстовых комментариев.
Важно: каждый комментарий уже может включать и "Опцию ответа", и свободный "Комментарий".

Нужно выявить root causes.

Требования:
1. Не пересказывай комментарии.
2. Объедини похожие причины.
3. Формулируй причины управленчески, коротко и точно.
4. Укажи, к какому вопросу относится каждая причина.
5. Не выдумывай того, чего нет в данных.
6. Если причина слабая или мало подтверждений — отметь это.

Верни строго JSON формата:
{{
  "summary": "краткий общий вывод по клинике",
  "root_causes": [
    {{
      "question": "название вопроса",
      "cause": "корневая причина",
      "evidence": ["краткий паттерн 1", "краткий паттерн 2"],
      "severity": "high|medium|low",
      "confidence": "high|medium|low"
    }}
  ]
}}

Данные:
{chr(10).join(blocks)}
            """.strip()

            parsed = ask_llm_json(prompt)

            results.append({
                "clinic": clinic,
                "summary": parsed.get("summary", ""),
                "root_causes_json": json.dumps(parsed, ensure_ascii=False)
            })

        return pd.DataFrame(results)

    def build_global_llm_summary(llm_df: pd.DataFrame) -> dict:
        if llm_df.empty:
            return {"management_summary": "Нет данных"}

        blocks = []

        for _, row in llm_df.iterrows():
            try:
                parsed = json.loads(row["root_causes_json"])
            except Exception:
                continue

            clinic = row["clinic"]
            summary = parsed.get("summary", "")
            causes = parsed.get("root_causes", [])

            causes_text = "\n".join([
                f"- {c.get('question')} → {c.get('cause')} "
                f"(severity={c.get('severity')}, confidence={c.get('confidence')})"
                for c in causes
            ])

            blocks.append(
                f"""
Клиника: {clinic}
Вывод: {summary}
Причины:
{causes_text}
"""
            )

        prompt = f"""
Ты руководитель CX аналитики сети клиник.

У тебя есть root cause анализ по клиникам.

Задача:
1. Найти системные проблемы (повторяются)
2. Найти различия между клиниками
3. Найти локальные аномалии
4. Определить приоритет проблем
5. Дать четкий управленческий вывод

Правила:
- не пересказывай
- не пиши воду
- игнорируй слабые сигналы
- опирайся на повторяемость причин
- думай как операционный директор

Верни JSON:
{{
  "system_problems": [],
  "clinic_differences": [],
  "local_anomalies": [],
  "management_summary": ""
}}

Данные:
{chr(10).join(blocks)}
"""
        return ask_llm_json(prompt)

    def build_final_llm_insight(df_neg: pd.DataFrame) -> dict:
        if df_neg.empty:
            return {"summary": "Нет данных"}

        question_segment = (
            df_neg.groupby(["question", "segment"])
            .size()
            .reset_index(name="count")
            .sort_values(["question", "count"], ascending=[True, False])
        )

        question_totals = (
            df_neg.groupby("question")
            .size()
            .reset_index(name="total_count")
            .sort_values("total_count", ascending=False)
        )

        blocks = []

        for _, qrow in question_totals.iterrows():
            q = qrow["question"]
            q_total = int(qrow["total_count"])

            seg_rows = question_segment[question_segment["question"] == q]
            seg_text = ", ".join(
                [f"{r['segment']}: {int(r['count'])}" for _, r in seg_rows.iterrows()]
            )

            texts = df_neg[
                (df_neg["question"] == q)
            ].copy()

            texts = texts[
                texts.apply(
                    lambda x: is_meaningful_combined_text(x["answer"], x["answer_option"]),
                    axis=1
                )
            ]["llm_text"].dropna().astype(str).tolist()

            if not texts:
                continue

            texts = texts[:80]
            comments_block = "\n".join([f"- {t}" for t in texts])

            blocks.append(
                f"""
ВОПРОС: {q}
ВСЕГО НЕГАТИВНЫХ ОТВЕТОВ ПО ВОПРОСУ: {q_total}
РАСПРЕДЕЛЕНИЕ ПО СЕГМЕНТАМ: {seg_text}

КОММЕНТАРИИ:
{comments_block}
""".strip()
            )

        prompt = f"""
Ты работаешь как сильный CX-аналитик, а не как копирайтер.

Ниже даны реальные негативные ответы пациентов:
- по вопросам
- с распределением по сегментам (Критик / Нейтрал)
- с текстовыми комментариями

Важно: внутри комментариев уже могут быть объединены "Опция ответа" и свободный "Ответ".

Твоя задача: сделать ПОЛНОЦЕННЫЙ аналитический разбор, а не короткую выжимку.

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
1. НЕЛЬЗЯ схлопывать всё в 1-2 общие темы, если внутри есть разные самостоятельные проблемы.
2. Нужно выделять ОТДЕЛЬНЫЕ паттерны проблем внутри каждого вопроса.
3. Если внутри одного вопроса есть 3 разные причины негатива — покажи все 3.
4. Не теряй редкие, но устойчивые проблемы, если они повторяются в нескольких комментариях.
5. Не пиши воду.
6. Не выдумывай причины, которых нет в текстах.
7. Если проблема больше характерна для критиков — укажи это. Если больше для нейтралов — тоже укажи.
8. Для каждой проблемы покажи:
   - к какому вопросу она относится
   - в чем суть проблемы
   - кто чаще пишет: критики / нейтралы / смешанный сигнал
   - примерные текстовые паттерны / evidence
   - относительную значимость: high / medium / low
   - примерный охват
9. После детального разбора сделай итог:
   - какие вопросы дают основной негатив
   - какие проблемы наиболее системные
   - какие рекомендации следуют из данных

Верни СТРОГО JSON следующего формата:
{{
  "question_analysis": [
    {{
      "question": "название вопроса",
      "total_negative_count": 0,
      "segment_split": {{
        "Критик": 0,
        "Нейтрал": 0
      }},
      "problems": [
        {{
          "problem": "конкретная проблема",
          "dominant_segment": "Критик|Нейтрал|Смешанный",
          "evidence_patterns": ["краткий паттерн 1", "краткий паттерн 2"],
          "impact": "high|medium|low",
          "coverage_estimate": "например: ~8 из 20 комментариев по вопросу"
        }}
      ]
    }}
  ],
  "cross_question_patterns": [
    {{
      "pattern": "системная проблема",
      "related_questions": ["вопрос 1", "вопрос 2"]
    }}
  ],
  "top_problem_areas": [
    {{
      "question": "название вопроса",
      "why_important": "почему это один из главных драйверов негатива"
    }}
  ],
  "recommendations": [
    "рекомендация 1",
    "рекомендация 2",
    "рекомендация 3"
  ],
  "management_summary": "итоговый вывод"
}}

ДАННЫЕ:
{chr(10).join(blocks)}
""".strip()

        return ask_llm_json(prompt)

    def build_question_pain_analysis(df_neg: pd.DataFrame, selected_question: str) -> dict:
        if selected_question != "Все вопросы":
            df_q = df_neg[df_neg["question"] == selected_question].copy()
        else:
            df_q = df_neg.copy()

        if df_q.empty:
            return {"summary": "Нет данных по выбранному вопросу"}

        option_stats = (
            df_q[
                df_q["answer_option"].astype(str).str.strip().ne("") &
                df_q["answer_option"].astype(str).str.strip().ne("nan")
            ]
            .groupby("answer_option")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

        option_block = "\n".join(
            [f"- {row['answer_option']}: {int(row['count'])}" for _, row in option_stats.head(15).iterrows()]
        )

        text_rows = df_q[
            df_q.apply(lambda x: is_meaningful_combined_text(x["answer"], x["answer_option"]), axis=1)
        ].copy()

        comments = text_rows["llm_text"].dropna().astype(str).tolist()[:120]

        if not comments:
            return {
                "question": selected_question,
                "summary": "Недостаточно текстовых данных по выбранному вопросу",
                "main_pain": {},
                "secondary_pains": [],
                "top_answer_options": option_stats.head(10).to_dict(orient="records"),
                "operational_recommendations": []
            }

        comments_block = "\n".join([f"- {x}" for x in comments])

        prompt = f"""
Ты сильный CX-аналитик сети клиник.

Ниже данные только по одному выбранному вопросу опроса.

ВОПРОС:
{selected_question}

РАСПРЕДЕЛЕНИЕ ОПЦИЙ ОТВЕТА:
{option_block if option_block else "Нет выраженных опций ответа"}

ТЕКСТЫ ОТВЕТОВ:
{comments_block}

Задача:
1. Выявить основную боль по этому вопросу.
2. Не смешивать разные причины в одну кашу.
3. Отдельно учитывать:
   - что люди выбирают в "Опция ответа"
   - что они дописывают в свободном поле "Ответ"
4. Выделить 3-7 ключевых паттернов боли.
5. Показать, какая боль основная, а какие вторичные.
6. Не выдумывать ничего вне текста.

Верни строго JSON:
{{
  "question": "название вопроса",
  "summary": "краткий вывод по основной боли",
  "main_pain": {{
    "pain": "главная проблема",
    "why_main": "почему это основной драйвер",
    "related_answer_options": ["..."],
    "evidence_patterns": ["..."],
    "severity": "high|medium|low"
  }},
  "secondary_pains": [
    {{
      "pain": "вторичная проблема",
      "related_answer_options": ["..."],
      "evidence_patterns": ["..."],
      "severity": "high|medium|low"
    }}
  ],
  "top_answer_options": [
    {{
      "answer_option": "текст опции",
      "count": 0,
      "interpretation": "что это означает"
    }}
  ],
  "operational_recommendations": [
    "рекомендация 1",
    "рекомендация 2"
  ]
}}
        """.strip()

        parsed = ask_llm_json(prompt)

        if "question" not in parsed:
            parsed["question"] = selected_question
        if "top_answer_options" not in parsed:
            parsed["top_answer_options"] = option_stats.head(10).to_dict(orient="records")
        if "secondary_pains" not in parsed:
            parsed["secondary_pains"] = []
        if "operational_recommendations" not in parsed:
            parsed["operational_recommendations"] = []

        return parsed

    # =====================================================
    # UI
    # =====================================================

    st.header("📊 CX / NPS Root Cause Analytics")

    uploaded1 = st.file_uploader(
        "Загрузите файл Excel/CSV",
        type=["xlsx", "csv"],
        key="uploader1"
    )

    if uploaded1:
        try:
            df = load_file(uploaded1)
        except Exception as e:
            st.error(str(e))
            st.stop()

        st.success("Файл загружен")

        if "source_schema" in df.columns and not df.empty:
            schema_name = df["source_schema"].iloc[0]
            if schema_name == "full":
                st.info("Обнаружен формат CX: полный (Ответ + Опция ответа)")
            elif schema_name == "comment":
                st.info("Обнаружен формат CX: упрощенный (Комментарий без Опции ответа)")

        with st.expander("Проверка данных"):
            st.write("Колонки после нормализации:")
            st.write(df.columns.tolist())
            st.dataframe(df.head(10), use_container_width=True)

        clinics = sorted(df["clinic"].dropna().unique().tolist())
        clinic_options = ["Все клиники"] + clinics

        col1, col2, col3 = st.columns(3)

        with col1:
            selected_clinic = st.selectbox("Актив / клиника", clinic_options)

        with col2:
            period_mode = st.selectbox("Таймлайн", ["Неделя", "Месяц"])

        with col3:
            selected_segments = st.multiselect(
                "Типы респондентов",
                options=TARGET_SEGMENTS,
                default=TARGET_SEGMENTS
            )

        df_work = df.copy()
        df_work = df_work[df_work["segment"].isin(selected_segments)]

        if selected_clinic != "Все клиники":
            df_work = df_work[df_work["clinic"] == selected_clinic]

        if df_work.empty:
            st.warning("После фильтров данных не осталось.")
            st.stop()

        # -------------------------------------------------
        # KPI
        # -------------------------------------------------
        st.subheader("1. Общая картина")

        seg_sum = segment_summary(df_work)

        a, b, c = st.columns(3)
        a.metric("Клиник в выборке", df_work["clinic"].nunique())
        b.metric("Уникальных ЭМК", df_work["patient_id"].nunique())
        c.metric("Негативных ответов", len(df_work))

        st.dataframe(seg_sum, use_container_width=True)

        # -------------------------------------------------
        # Questions
        # -------------------------------------------------
        st.subheader("2. Какие вопросы сильнее негативят")

        q_sum = question_summary(df_work)

        q_pivot = (
            q_sum.pivot_table(
                index=["clinic", "question"],
                columns="segment",
                values="count",
                fill_value=0
            )
            .reset_index()
        )

        q_pivot["total_negative"] = q_pivot.get("Критик", 0) + q_pivot.get("Нейтрал", 0)
        q_pivot = q_pivot.sort_values(["clinic", "total_negative"], ascending=[True, False])

        st.dataframe(q_pivot, use_container_width=True)

        top_chart = q_pivot.copy()
        if selected_clinic == "Все клиники":
            top_chart = top_chart.groupby("question", as_index=False)["total_negative"].sum()
            fig_top = px.bar(
                top_chart.sort_values("total_negative", ascending=False).head(15),
                x="question",
                y="total_negative",
                title="Топ вопросов по негативу",
            )
        else:
            fig_top = px.bar(
                top_chart.head(15),
                x="question",
                y="total_negative",
                title=f"Топ вопросов по негативу — {selected_clinic}",
            )

        st.plotly_chart(fig_top, use_container_width=True)

        # -------------------------------------------------
        # Timeline
        # -------------------------------------------------
        st.subheader("3. Таймлайн")

        tl = timeline_summary(df_work, period_mode)
        time_col = "year_week" if period_mode == "Неделя" else "month_name"

        fig_tl = px.line(
            tl,
            x=time_col,
            y="count",
            color="segment",
            line_group="clinic",
            facet_row="clinic" if selected_clinic == "Все клиники" and df_work["clinic"].nunique() <= 6 else None,
            markers=True,
            title=f"Динамика негатива по {period_mode.lower()}м"
        )
        st.plotly_chart(fig_tl, use_container_width=True)
        st.dataframe(tl, use_container_width=True)

        # -------------------------------------------------
        # Day drill-down
        # -------------------------------------------------
        if period_mode == "Месяц":
            st.subheader("4. Drill-down по дням внутри месяца")

            months = sorted(df_work["month_name"].dropna().unique().tolist())
            selected_month = st.selectbox("Выберите месяц для разбивки по дням", months)

            day_df = day_breakdown_for_month(df_work, selected_month)

            fig_day = px.bar(
                day_df,
                x="day",
                y="count",
                color="segment",
                barmode="group",
                facet_row="clinic" if selected_clinic == "Все клиники" and df_work["clinic"].nunique() <= 6 else None,
                title=f"Разбивка по дням — {selected_month}"
            )
            st.plotly_chart(fig_day, use_container_width=True)
            st.dataframe(day_df, use_container_width=True)

        # -------------------------------------------------
        # Patient transitions
        # -------------------------------------------------
        st.subheader("5. Динамика по ЭМК: меняется ли сегмент")

        patient_seq, patient_transitions = patient_transition_analysis(df_work)

        left, right = st.columns([1, 2])

        with left:
            st.write("Сводка переходов")
            st.dataframe(patient_transitions, use_container_width=True)

        with right:
            fig_trans = px.bar(
                patient_transitions,
                x="transition",
                y="patient_count",
                title="Переходы между сегментами"
            )
            st.plotly_chart(fig_trans, use_container_width=True)

        with st.expander("Подробно по ЭМК"):
            st.dataframe(
                patient_seq.sort_values(["changed_segment", "touchpoints"], ascending=[False, False]),
                use_container_width=True
            )

            # ---------------------------------------------
            # LLM block
            # ---------------------------------------------
            st.subheader("6. Root Cause Extraction по текстовым ответам")

            if "llm_df" not in st.session_state:
                st.session_state.llm_df = pd.DataFrame()

            if "global_summary" not in st.session_state:
                st.session_state.global_summary = None

            if "final_insight" not in st.session_state:
                st.session_state.final_insight = None

            if "question_pain_result" not in st.session_state:
                st.session_state.question_pain_result = None

            text_candidates = df_work[
                df_work.apply(
                    lambda x: is_meaningful_combined_text(x["answer"], x["answer_option"]),
                    axis=1
                )
            ].copy()

            x1, x2, x3 = st.columns(3)
            x1.metric("Текстовых ответов для LLM", len(text_candidates))
            x2.metric("Уникальных вопросов", text_candidates["question"].nunique())
            x3.metric("Клиник", text_candidates["clinic"].nunique())

            if len(text_candidates) == 0:
                st.info("Нет текстовых данных")
            else:
                question_options = ["Все вопросы"] + sorted(df_work["question"].dropna().unique().tolist())
                selected_question_llm = st.selectbox(
                    "Вопрос для LLM-анализа",
                    question_options,
                    key="selected_question_llm"
                )

                col1, col2, col3, col4 = st.columns(4)

                run_llm = col1.button("Root cause")
                run_global = col2.button("Сеть")
                run_final = col3.button("Итог")
                run_question_pain = col4.button("Боль по вопросу")

                if run_llm:
                    with st.spinner("LLM..."):
                        st.session_state.llm_df = run_root_cause_llm(
                            text_candidates,
                            selected_clinic=selected_clinic,
                            max_questions_per_clinic=6,
                            sample_per_question=30
                        )

                llm_df = st.session_state.llm_df

                if not llm_df.empty:
                    for _, row in llm_df.iterrows():
                        st.markdown(f"### {row['clinic']}")
                        st.write(row["summary"])

                        parsed = json.loads(row["root_causes_json"])
                        st.dataframe(pd.DataFrame(parsed.get("root_causes", [])), use_container_width=True)

                if run_global and not llm_df.empty:
                    with st.spinner("LLM global..."):
                        st.session_state.global_summary = build_global_llm_summary(llm_df)

                if st.session_state.global_summary:
                    st.subheader("7. Сводка сети")
                    st.json(st.session_state.global_summary)

                if run_final:
                    with st.spinner("LLM final..."):
                        st.session_state.final_insight = build_final_llm_insight(text_candidates)

                final = st.session_state.final_insight

                if final:
                    st.subheader("8. Итоговый анализ")

                    for q in final.get("question_analysis", []):
                        st.markdown(f"### {q['question']}")
                        st.dataframe(pd.DataFrame(q.get("problems", [])), use_container_width=True)

                    st.markdown("### Системные паттерны")
                    st.dataframe(pd.DataFrame(final.get("cross_question_patterns", [])), use_container_width=True)

                    st.markdown("### Рекомендации")
                    for r in final.get("recommendations", []):
                        st.write(f"- {r}")

                    st.markdown("### Итог")
                    st.write(final.get("management_summary", ""))

                if run_question_pain:
                    with st.spinner("LLM question pain..."):
                        st.session_state.question_pain_result = build_question_pain_analysis(
                            text_candidates,
                            selected_question_llm
                        )

                question_pain = st.session_state.question_pain_result

                if question_pain:
                    st.subheader("9. Основная боль по выбранному вопросу")

                    st.markdown(f"### Вопрос: {question_pain.get('question', selected_question_llm)}")
                    st.write(question_pain.get("summary", ""))

                    main_pain = question_pain.get("main_pain", {})
                    if main_pain:
                        st.markdown("### Главная боль")
                        st.json(main_pain)

                    secondary = question_pain.get("secondary_pains", [])
                    if secondary:
                        st.markdown("### Вторичные боли")
                        st.dataframe(pd.DataFrame(secondary), use_container_width=True)

                    top_opts = question_pain.get("top_answer_options", [])
                    if top_opts:
                        st.markdown("### Что чаще выбирают в 'Опция ответа'")
                        st.dataframe(pd.DataFrame(top_opts), use_container_width=True)

                    recs = question_pain.get("operational_recommendations", [])
                    if recs:
                        st.markdown("### Операционные рекомендации")
                        for r in recs:
                            st.write(f"- {r}")

            # ---------------------------------------------
            # Export
            # ---------------------------------------------
            st.subheader("10. Экспорт")

            export_dict = {
                "raw_filtered": df_work,
                "segment_summary": seg_sum,
                "question_summary": q_sum,
                "question_pivot": q_pivot,
                "timeline": tl,
                "patient_sequences": patient_seq,
                "patient_transitions": patient_transitions,
            }

            if period_mode == "Месяц":
                export_dict["day_breakdown"] = day_df

            if "llm_df" in st.session_state and not st.session_state.llm_df.empty:
                export_dict["llm_root_causes"] = st.session_state.llm_df

            if "question_pain_result" in st.session_state and st.session_state.question_pain_result:
                qp = st.session_state.question_pain_result

                main_pain_df = pd.DataFrame([qp.get("main_pain", {})]) if qp.get("main_pain") else pd.DataFrame()
                secondary_df = pd.DataFrame(qp.get("secondary_pains", []))
                top_opts_df = pd.DataFrame(qp.get("top_answer_options", []))
                recs_df = pd.DataFrame({"recommendation": qp.get("operational_recommendations", [])})
                summary_df = pd.DataFrame([{
                    "question": qp.get("question", ""),
                    "summary": qp.get("summary", "")
                }])

                export_dict["question_pain_summary"] = summary_df
                export_dict["question_main_pain"] = main_pain_df
                export_dict["question_secondary_pains"] = secondary_df
                export_dict["question_top_options"] = top_opts_df
                export_dict["question_recommendations"] = recs_df

            excel_bytes = to_excel_bytes(export_dict)

            st.download_button(
                "⬇️ Скачать Excel с анализом",
                data=excel_bytes,
                file_name="cx_root_cause_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# =========================================================
# TAB 2: PAIN MAP ML
# =========================================================

with tab2:
    st.header("🎯 CX Pain Map — ML driven")

    uploaded2 = st.file_uploader(
        "Загрузите файл (xlsx / csv)",
        type=["xlsx", "csv"],
        key="uploader2"
    )

    if uploaded2:
        try:
            from ml import predict_topics
            from logic import build_pain_matrix
            ml_available = True
        except Exception as e:
            st.error(f"Ошибка загрузки ML модулей: {e}")
            st.info("Проверьте файлы ml.py и logic.py")
            ml_available = False

        if ml_available:
            if uploaded2.name.endswith(".xlsx"):
                df2 = pd.read_excel(uploaded2)
            else:
                df2 = pd.read_csv(uploaded2)

            st.subheader("Настройка колонок")

            text_col = st.selectbox("Колонка с текстом", df2.columns, key="text_col")
            date_col = st.selectbox("Колонка с датой", df2.columns, key="date_col")
            segment_col = st.selectbox("Колонка с типом респондента", df2.columns, key="segment_col")

            threshold = st.slider(
                "Порог вероятности темы",
                0.1, 0.9, 0.5, 0.05,
                key="threshold"
            )

            if st.button("▶ Определить темы и построить pain-map", key="run_ml"):
                with st.spinner("ML-классификация..."):
                    df_ml = predict_topics(df2, text_col)

                weekly, pivots = build_pain_matrix(
                    df_ml,
                    date_col=date_col,
                    topic_col="ml_topics",
                    segment_col=segment_col,
                    text_col=text_col
                )

                st.success("Готово")

                if pivots:
                    fig, axes = plt.subplots(1, len(pivots), figsize=(18, 6), sharey=True)
                    if len(pivots) == 1:
                        axes = [axes]

                    for ax, (seg, pv) in zip(axes, pivots.items()):
                        im = ax.imshow(pv.T.values, aspect="auto", cmap="Reds")
                        ax.set_title(seg)
                        ax.set_xticks(range(len(pv.index)))
                        ax.set_xticklabels(pv.index, rotation=45, ha="right")
                        ax.set_yticks(range(len(pv.columns)))
                        ax.set_yticklabels(pv.columns)

                    plt.colorbar(im, ax=axes)
                    st.pyplot(fig)

                export_dict_ml = {
                    "with_ml": df_ml,
                    "weekly_long": weekly
                }

                for seg, pv in pivots.items():
                    export_dict_ml[f"counts_{seg}"] = pv

                excel_bytes_ml = to_excel_bytes(export_dict_ml)

                st.download_button(
                    "⬇️ Скачать Excel",
                    data=excel_bytes_ml,
                    file_name="pain_map.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
