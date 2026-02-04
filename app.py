import os

import pandas as pd
import streamlit as st

from src.utils import load_object


def get_artifact_path(filename):
    return os.path.join(os.path.dirname(__file__), "artifacts", filename)


st.set_page_config(page_title="Insurance Charges Predictor", page_icon="🩺")

st.title("Insurance Charges Predictor")
st.write(
    "Provide the details below to estimate insurance charges using the trained model."
)

model_path = get_artifact_path("model.pkl")
preprocessor_path = get_artifact_path("preprocessor.pkl")

try:
    model = load_object(model_path)
    preprocessor = load_object(preprocessor_path)
except Exception as exc:
    st.error(
        "Unable to load model artifacts. Ensure 'artifacts/model.pkl' and "
        "'artifacts/preprocessor.pkl' exist before running the app."
    )
    st.exception(exc)
    st.stop()

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

    with col2:
        children = st.number_input("Children", min_value=0, max_value=10, value=0)
        sex = st.selectbox("Sex", options=["female", "male"])

    with col3:
        smoker = st.selectbox("Smoker", options=["yes", "no"])
        region = st.selectbox(
            "Region", options=["southeast", "southwest", "northwest", "northeast"]
        )

    submitted = st.form_submit_button("Predict Charges")

if submitted:
    input_df = pd.DataFrame(
        {
            "age": [age],
            "bmi": [bmi],
            "children": [children],
            "sex": [sex],
            "smoker": [smoker],
            "region": [region],
        }
    )

    transformed = preprocessor.transform(input_df)
    prediction = model.predict(transformed)[0]

    st.success(f"Estimated Insurance Charges: ${prediction:,.2f}")
