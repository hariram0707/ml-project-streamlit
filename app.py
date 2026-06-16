import streamlit as st
import pandas as pd
import joblib

model = joblib.load("xgboost_model.pkl")

st.title("Student Performance Predictor")

tool_diversity = st.number_input("Tool Diversity")
weekly_genai_hours = st.number_input("Weekly GenAI Hours")
traditional_study_hours = st.number_input("Traditional Study Hours")

if st.button("Predict"):

    data = pd.DataFrame(
        [[tool_diversity,
          weekly_genai_hours,
          traditional_study_hours]],
        columns=[
            "Tool_Diversity",
            "Weekly_GenAI_Hours",
            "Traditional_Study_Hours"
        ]
    )

    prediction = model.predict(data)

    st.success(f"Prediction: {prediction[0]}")