from pathlib import Path
import json
import re
import requests
import pandas as pd
import streamlit as st

URL = st.secrets["OLLAMA_URL"].strip()
API_KEY = st.secrets["OLLAMA_API_KEY"].strip()
MODEL = "gpt-oss:120b"


def ask_llm_json(prompt: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты аналитик CX. "
                    "Классифицируй жалобы пациентов по темам. "
                    "Отвечай строго JSON без markdown и без пояснений."
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

    r = requests.post(URL, headers=headers, json=payload, timeout=180)

    if r.status_code != 200:
        raise RuntimeError(f"LLM request failed: {r.status_code} {r.text}")

    data = r.json()
    content = data["message"]["content"].strip()

    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        raise RuntimeError(f"LLM did not return valid JSON: {content[:1000]}")

    return json.loads(match.group())


def normalize_text(x) -> str:
    if x is None:
        return ""
    x = str(x).strip()
    if x.lower() in {"nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", x)


def build_topic_prompt(texts: list[str], allowed_topics: list[str]) -> str:
    topic_list = "\n".join([f"- {t}" for t in allowed_topics])

    items = []
    for i, txt in enumerate(texts, start=1):
        items.append(f"{i}. {txt}")

    items_block = "\n".join(items)

    return f"""
Ниже комментарии пациентов.

Нужно для каждого комментария определить 1-3 наиболее подходящие темы из фиксированного списка.

СПИСОК ДОПУСТИМЫХ ТЕМ:
{topic_list}

ПРАВИЛА:
1. Используй только темы из списка.
2. Если тема не подходит — верни пустой список.
3. Не выдумывай новые темы.
4. Если комментарий слишком общий — выбери самые близкие темы по смыслу.
5. На выходе строго JSON формата:

{{
  "items": [
    {{
      "id": 1,
      "topics": ["тема1", "тема2"]
    }}
  ]
}}

КОММЕНТАРИИ:
{items_block}
""".strip()


def chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def predict_topics(df, text_col, threshold=0.5):
    """
    Совместимая сигнатура, чтобы app.py почти не трогать.
    threshold здесь не используется, но оставлен для совместимости интерфейса.
    """
    df = df.copy().reset_index(drop=True)
    df["_text_norm"] = df[text_col].apply(normalize_text)

    # Можно потом вынести в UI, но для теста зафиксируем список
    allowed_topics = [
        "Ожидание / очереди",
        "Запись / перенос приема",
        "Коммуникация персонала",
        "Навигация / организация процесса",
        "Документы / оформление",
        "Стоимость / оплата",
        "Мобильное приложение / сайт",
        "Контакт-центр / дозвон",
        "Прием врача",
        "Диагностика / обследование",
        "Повторные визиты",
        "Результаты / обратная связь",
    ]

    texts = df["_text_norm"].tolist()
    batch_size = 20

    all_topics = [""] * len(df)

    indexed_texts = list(enumerate(texts))

    for batch in chunk_list(indexed_texts, batch_size):
        batch_indices = [idx for idx, _ in batch]
        batch_texts = [txt for _, txt in batch]

        prompt = build_topic_prompt(batch_texts, allowed_topics)
        parsed = ask_llm_json(prompt)

        result_items = parsed.get("items", [])

        local_topics_map = {}
        for item in result_items:
            item_id = item.get("id")
            topics = item.get("topics", [])
            if not isinstance(topics, list):
                topics = []
            topics = [str(t).strip() for t in topics if str(t).strip() in allowed_topics]
            local_topics_map[item_id] = ",".join(topics)

        for local_pos, global_idx in enumerate(batch_indices, start=1):
            all_topics[global_idx] = local_topics_map.get(local_pos, "")

    df["ml_topics"] = all_topics

    # Для совместимости с экспортом можно добавить пустые prob-колонки не надо.
    return df.drop(columns=["_text_norm"])
