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
        if r.get(segment_col) not in segments:
            continue
        if pd.isna(r[topic_col]) or str(r[topic_col]).strip() == "":
            continue
        if pd.isna(r["week"]):
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
                "segment": r[segment_col]
            })

    pain = pd.DataFrame(rows)

    weekly = (
        pain
        .groupby(["week", "topic", "segment"])
        .size()
        .reset_index(name="count")
    )

    pivots = {
        seg: (
            weekly[weekly["segment"] == seg]
            .pivot(index="week", columns="topic", values="count")
            .fillna(0)
            .sort_index()
        )
        for seg in segments
    }

    return weekly, pivots

