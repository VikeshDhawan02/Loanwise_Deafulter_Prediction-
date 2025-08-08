# Home.py
import streamlit as st

st.set_page_config(page_title="Loan Default Prediction App", layout="centered")

st.title("LoanWise Defaulter Prediction Project")

st.markdown("""
Welcome to the LoanWise Default Prediction application. This project aims to develop an accurate and interpretable machine learning model for predicting loan defaults.

### Project Overview
* **Objective**: To build a robust machine learning model that predicts loan default, helping financial institutions make informed decisions.
* **Dataset**: The model is trained on a comprehensive dataset containing various features such as Age, Income, Loan Amount, Credit Score, Employment details, and a new `DebtAmount` feature.

### How to Use
* Navigate to the "Prediction UI" page using the sidebar to input customer details and get a real-time loan default prediction.
* The "Tableau Dashboard" page provides an interactive visualization of historical data and model performance (if applicable).
""")

st.markdown("---")
st.markdown("Developed as a machine learning project.")

