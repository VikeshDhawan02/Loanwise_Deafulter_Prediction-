# home.py
import streamlit as st

st.set_page_config(page_title="Loan Default Prediction App", layout="centered")

st.title("LoanWise Defaulter Prediction Project")

st.markdown("""
Welcome to the Loan Default Prediction application. This project aims to predict whether a loan applicant is likely to default on their loan based on various financial and personal attributes.

### Project Overview
* **Objective**: To build a robust machine learning model that predicts loan default, helping financial institutions make informed decisions.
* **Technology Stack**:
    * **Frontend**: Streamlit for interactive web application development.
    * **Backend (Data Storage)**: MySQL database to store prediction results and input data.
    * **Machine Learning**: Python with the XGBoost library for model training and prediction, using one-hot encoded features and a `StandardScaler` for preprocessing.
    * **Data Visualization**: Tableau for interactive dashboards.
* **Dataset**: The model is trained on a comprehensive dataset containing various features such as Age, Income, Loan Amount, Credit Score, Employment details, and a new `DebtAmount` feature.

### How to Use
Navigate to the "Prediction UI" page using the sidebar to input customer details and get a real-time loan default prediction.
The "Tableau Dashboard" page provides an interactive visualization of historical data and model performance (if applicable).

### Model Details
The prediction model used in this application is a pre-trained machine learning model (XGBoost) which has been serialized using `pickle`. A `StandardScaler` is also used to preprocess numerical features. Note that the model expects one-hot encoded categorical features, which are handled internally by the application.

**Please ensure the `Xgboost.pkl` and `scaler.pkl` files are present in the root directory alongside this application.**
""")

st.markdown("---")
st.markdown("Developed as a machine learning project.")
