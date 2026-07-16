from __future__ import annotations

import streamlit as st

from src.predict import StrokePredictor


@st.cache_resource
def load_predictor() -> StrokePredictor:
    return StrokePredictor()


st.set_page_config(page_title="Stroke Prediction", layout="centered")
st.title("Stroke Prediction")

predictor = load_predictor()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", min_value=0, max_value=120, value=45, step=1)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart disease", ["No", "Yes"])
        ever_married = st.selectbox("Ever married", ["Yes", "No"])

    with col2:
        work_type = st.selectbox(
            "Work type",
            ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
        )
        residence_type = st.selectbox("Residence type", ["Urban", "Rural"])
        avg_glucose_level = st.number_input(
            "Average glucose level",
            min_value=0.0,
            value=100.0,
            step=1.0,
        )
        bmi = st.number_input("BMI", min_value=0.0, value=28.0, step=0.1)
        smoking_status = st.selectbox(
            "Smoking status",
            ["never smoked", "formerly smoked", "smokes"],
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    record = {
        "gender": gender,
        "age": float(age),
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status,
    }

    result = predictor.predict_one(record)
    probability = result["stroke_probability"]
    prediction = result["prediction"]

    st.metric("Stroke probability", f"{probability:.2%}")

    if prediction == 1:
        st.error(f"High risk detected. Threshold: {result['threshold']:.3f}")
    else:
        st.success(f"No high-risk flag. Threshold: {result['threshold']:.3f}")

    st.caption("Educational project output. Not medical advice.")
