from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "nps_sbert_model.pkl"
ENCODER_PATH = BASE_DIR / "model" / "nps_mlb.pkl"


# ---------------------------------------------------------
# PICKLE COMPAT SHIM
# ---------------------------------------------------------
# Старый pickle ищет:
# transformers.models.bert.modeling_bert.BertSdpaSelfAttention
# В текущем transformers этого класса может уже не быть.
# Подкладываем алиас на BertSelfAttention, чтобы joblib.load не падал.

def patch_transformers_pickle_compat():
    try:
        import transformers.models.bert.modeling_bert as modeling_bert

        if not hasattr(modeling_bert, "BertSdpaSelfAttention"):
            if hasattr(modeling_bert, "BertSelfAttention"):
                class BertSdpaSelfAttention(modeling_bert.BertSelfAttention):
                    pass

                modeling_bert.BertSdpaSelfAttention = BertSdpaSelfAttention

        return True, None
    except Exception as e:
        return False, e


# ---------------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_ml_artifacts():
    ok, patch_error = patch_transformers_pickle_compat()
    if not ok:
        raise RuntimeError(
            f"Не удалось применить compatibility patch для transformers: "
            f"{type(patch_error).__name__}: {patch_error}"
        )

    bundle = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    clf = bundle["clf"]
    embedder = bundle["embedder"]

    return clf, embedder, encoder


# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------

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
