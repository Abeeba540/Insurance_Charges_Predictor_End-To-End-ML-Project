import streamlit as st
import pandas as pd
import numpy as np
import os

from src.utils import load_object

# Page config
st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")

st.title("💰 Insurance Charges Prediction")
st.write("Enter customer details to predict insurance charges")

# Load artifacts
MODEL_PATH = os.path.join("artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")

model = load_object(MODEL_PATH)
preprocessor = load_object(PREPROCESSOR_PATH)

# User input form
age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)

sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Predict button
if st.button("Predict Insurance Charges"):
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex": [sex],
        "smoker": [smoker],
        "region": [region]
    })

    # Preprocess input
    transformed_data = preprocessor.transform(input_data)

    # Prediction
    prediction = model.predict(transformed_data)[0]

    st.success(f"💵 Estimated Insurance Charges: ₹ {prediction:,.2f}")
