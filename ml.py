import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "model/nps_sbert_model.pkl"
ENCODER_PATH = "model/nps_mlb.pkl"


@st.cache_resource(show_spinner=False)
def load_ml_artifacts():
    bundle = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    clf = bundle["clf"]
    embedder = bundle["embedder"]
    return clf, embedder, encoder


def predict_topics(df, text_col, threshold=0.5):
    clf, embedder, encoder = load_ml_artifacts()

    texts = df[text_col].astype(str).fillna("").tolist()

    X = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False
    )

    probs = clf.predict_proba(X)
    topic_cols = encoder.classes_

    prob_df = pd.DataFrame(
        probs,
        columns=[f"prob_{t}" for t in topic_cols]
    )

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
