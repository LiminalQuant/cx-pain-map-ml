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
    segments=("Netral", "Detractor")
):
    df = df.copy()
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

        for t in str(r[topic_col]).split(","):
            rows.append({
                "week": r["week"],
                "topic": t.strip(),
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
