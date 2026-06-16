import streamlit as st
import pandas as pd
import joblib

model = joblib.load("naive_bayes_model.pkl")

x1 = st.number_input("Tool Diversity")
x2 = st.number_input("Weekly GenAI Hours")
x3 = st.number_input("Traditional Study Hours")

if st.button("Predict"):
    data = pd.DataFrame(
        [[x1, x2, x3]],
        columns=[
            'Tool_Diversity',
            'Weekly_GenAI_Hours',
            'Traditional_Study_Hours'
        ]
    )

    prediction = model.predict(data)
    st.success(f"Predicted GPA: {prediction[0]:.2f}")