import joblib
import pandas as pd

MODEL_PATH = "model/nps_sbert_model.pkl"
ENCODER_PATH = "model/nps_mlb.pkl"

bundle = joblib.load(MODEL_PATH)
clf = bundle["clf"]
embedder = bundle["embedder"]

encoder = joblib.load(ENCODER_PATH)


def predict_topics(df, text_col):
    texts = df[text_col].astype(str).fillna("").tolist()

    X = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False
    )

    # multi-label → probabilities
    probs = clf.predict_proba(X)

    topic_cols = encoder.classes_

    prob_df = pd.DataFrame(
        probs,
        columns=[f"prob_{t}" for t in topic_cols]
    )

    # порог — можно вынести в UI
    threshold = 0.5
    topics = []

    for row in probs:
        labels = [
            topic_cols[i]
            for i, p in enumerate(row)
            if p >= threshold
        ]
        topics.append(",".join(labels))

    df = df.reset_index(drop=True)
    df["ml_topics"] = topics

    return pd.concat([df, prob_df], axis=1)
