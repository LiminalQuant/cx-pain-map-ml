import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ml import predict_topics
from logic import build_pain_matrix

st.set_page_config(layout="wide")
st.title("CX Pain Map — ML driven")

uploaded = st.file_uploader(
    "Загрузите файл (xlsx / csv)",
    type=["xlsx", "csv"]
)

if uploaded:
    if uploaded.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)

    st.subheader("Настройка колонок")

    text_col = st.selectbox("Колонка с текстом", df.columns)
    date_col = st.selectbox("Колонка с датой", df.columns)
    segment_col = st.selectbox("Колонка с типом респондента", df.columns)

    threshold = st.slider(
        "Порог вероятности темы",
        0.1, 0.9, 0.5, 0.05
    )

    if st.button("▶ Определить темы и построить pain-map"):
        with st.spinner("ML-классификация..."):
            df_ml = predict_topics(df, text_col)

        weekly, pivots = build_pain_matrix(
            df_ml,
            date_col=date_col,
            topic_col=topic_col,
            segment_col=segment_col,
            text_col=text_col
        )

        st.success("Готово")

        # === VISUAL ===
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

        for ax, (seg, pv) in zip(axes, pivots.items()):
            ax.imshow(pv.T.values, aspect="auto", cmap="Reds")
            ax.set_title(seg)
            ax.set_xticks(range(len(pv.index)))
            ax.set_xticklabels(pv.index, rotation=45, ha="right")
            ax.set_yticks(range(len(pv.columns)))
            ax.set_yticklabels(pv.columns)

        st.pyplot(fig)

        # === DOWNLOAD ===
        with pd.ExcelWriter("pain_map.xlsx", engine="openpyxl") as writer:
            df_ml.to_excel(writer, sheet_name="with_ml", index=False)
            weekly.to_excel(writer, sheet_name="weekly_long", index=False)
            for seg, pv in pivots.items():
                pv.to_excel(writer, sheet_name=f"counts_{seg}")

        with open("pain_map.xlsx", "rb") as f:
            st.download_button(
                "⬇️ Скачать Excel",
                f,
                file_name="pain_map.xlsx"
            )
