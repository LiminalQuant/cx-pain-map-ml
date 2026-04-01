import pandas as pd
from dateutil import parser


def parse_date(val):
    if pd.isna(val):
        return pd.NaT
    s = str(val).strip().replace('"', "").replace("'", "")
    try:
        return parser.parse(s, dayfirst=True)
    except Exception:
        return pd.NaT


def build_pain_matrix(
    df,
    date_col,
    topic_col,
    segment_col,
    text_col,
    segments=("Netral", "Detractor")
):
    df = df.copy()

    # --- нормализация текста ---
    df["_comment_norm"] = (
        df[text_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("\n", " ", regex=False)
    )

    df = df[df["_comment_norm"] != ""]

    # --- дедупликация по комментарию ---
    df = df.sort_values(date_col)
    df = df.drop_duplicates(subset=["_comment_norm"], keep="first")

    # --- дата ---
    df["bill_date"] = df[date_col].apply(parse_date)
    df["week"] = df["bill_date"].dt.to_period("W").astype(str)

    rows = []

    for _, r in df.iterrows():
        seg_value = str(r.get(segment_col, "")).strip()

        if seg_value not in segments:
            continue

        if pd.isna(r.get(topic_col)) or str(r.get(topic_col)).strip() == "":
            continue

        if pd.isna(r.get("bill_date")):
            continue

        topics = {
            t.strip()
            for t in str(r[topic_col]).split(",")
            if t.strip()
        }

        for t in topics:
            rows.append({
                "week": r["week"],
                "topic": t,
                "segment": seg_value
            })

    # --- если после всех фильтров ничего не осталось ---
    if not rows:
        empty_weekly = pd.DataFrame(columns=["week", "topic", "segment", "count"])
        empty_pivots = {
            seg: pd.DataFrame()
            for seg in segments
        }
        return empty_weekly, empty_pivots

    pain = pd.DataFrame(rows, columns=["week", "topic", "segment"])

    weekly = (
        pain
        .groupby(["week", "topic", "segment"])
        .size()
        .reset_index(name="count")
    )

    pivots = {}
    for seg in segments:
        seg_df = weekly[weekly["segment"] == seg].copy()

        if seg_df.empty:
            pivots[seg] = pd.DataFrame()
        else:
            pivots[seg] = (
                seg_df
                .pivot(index="week", columns="topic", values="count")
                .fillna(0)
                .sort_index()
            )

    return weekly, pivots
