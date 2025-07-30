import streamlit as st
import pandas as pd
from automl_engine import detect_task_type, run_automl

st.title("AutoML-as-a-Service")

st.markdown("Upload your dataset (CSV format).")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    target_col = st.selectbox("Select the target column", df.columns)

    if target_col:
        task = detect_task_type(df[target_col])
        st.markdown(f"**Detected Task Type:** `{task}`")

        if st.button("Run AutoML"):
            st.info("Training in progress... please wait ⏳")

            results = run_automl(df, target_col, task)

            st.success("Training complete! 🎉")

            st.subheader("Model Performance Leaderboard")
            st.table(pd.DataFrame(results, columns=["Model", "Score"]))
