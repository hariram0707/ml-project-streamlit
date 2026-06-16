import streamlit as st
import joblib
import pandas as pd

model = joblib.load("naive_bayes_model.pkl")

st.title("ML Prediction App")

st.write("Enter values:")

# CHANGE THESE INPUTS BASED ON YOUR DATASET
x1 = st.number_input("Feature 1")
x2 = st.number_input("Feature 2")
x3 = st.number_input("Feature 3")

if st.button("Predict"):
    data = pd.DataFrame([[x1, x2, x3]])
    prediction = model.predict(data)
    st.success(f"Prediction: {prediction[0]}")