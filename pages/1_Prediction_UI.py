# pages/1_Prediction_UI.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import mysql.connector
from mysql.connector import Error


import streamlit as st
import base64

# Function to encode a local image to a Base64 string
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set the background
def set_background(image_file):
    bin_str = get_base64_of_bin_file(image_file)
    page_bg_img = f'''
    <style>
    .stApp {{
    background-image: url("data:image/jpeg;base64,{bin_str}");
    background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Call the function with your image file name
# Make sure 'front.jpeg' is in the same directory as this script.
set_background('front.jpeg')




# --- Configuration ---
# MySQL connection config
# Update these with your actual MySQL credentials
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'testing'
}

# --- Updated field lists to match spark_cleaned_data.csv ---
# These are the columns the model was trained on, with a total of 26 features
model_input_fields = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'DebtAmount',
    'EmploymentType', 'HasMortgage', 'HasDependents', 'HasCoSigner',
    'LoanPurpose_Education', 'LoanPurpose_Home', 'LoanPurpose_Other',
    'LoanPurpose_Auto', 'LoanPurpose_Business', 'MaritalStatus_Married',
    'MaritalStatus_Single', 'MaritalStatus_Divorced', 'Education_HighSchool',
    'Education_Bachelor', 'Education_Master', 'Education_PhD'
]

# Numerical columns for scaling
numerical_cols = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'DebtAmount'
]

# Categorical mappings (from your balanced_data.py or similar)
# This is crucial for displaying user-friendly options and converting back to numerical
label_map = {
    'Education': {"Bachelor's": 'Bachelor', "Master's": 'Master', 'High School': 'HighSchool', 'PhD': 'PhD'},
    'EmploymentType': {'Full-time': 4, 'Unemployed': 0, 'Self-employed': 1, 'Part-time': 2},
    'MaritalStatus': {'Divorced': 'Divorced', 'Married': 'Married', 'Single': 'Single'},
    'HasMortgage': {'Yes': 1, 'No': 0},
    'HasDependents': {'Yes': 1, 'No': 0},
    'LoanPurpose': {'Other': 'Other', 'Auto': 'Auto', 'Business': 'Business', 'Home': 'Home', 'Education': 'Education'},
    'HasCoSigner': {'Yes': 1, 'No': 0},
}


# --- Function to load model and scaler ---
@st.cache_resource
@st.cache_resource
def load_model():
    """
    Loads the pre-trained LGBM model.
    """
    try:
        with open('lgbm_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Please ensure 'lgbm_model.pkl' is in the root directory.")
        return None

model = load_model()

# --- Function to insert prediction into MySQL database ---
def insert_prediction_into_db(input_data, prediction_result):
    """
    Inserts the prediction data and result into the MySQL predictions table.
    """
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # The new table has more columns due to one-hot encoding
        # You must have run the new CREATE TABLE query in MySQL for this to work
        columns = [
            'prediction_timestamp', 'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
            'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'DebtAmount', 'Education',
            'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents',
            'LoanPurpose', 'HasCoSigner', 'PredictionResult'
        ]
        
        # Prepare the data for insertion
        placeholders = ', '.join(['%s'] * len(columns))
        insert_query = f"INSERT INTO predictions_log ({', '.join(columns)}) VALUES ({placeholders})"

        # Create a list of values to insert
        values_to_insert = [
            pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            input_data['Age'], input_data['Income'], input_data['LoanAmount'],
            input_data['CreditScore'], input_data['MonthsEmployed'], input_data['NumCreditLines'],
            input_data['InterestRate'], input_data['LoanTerm'], input_data['DTIRatio'],
            input_data['DebtAmount'], input_data['Education'], input_data['EmploymentType'],
            input_data['MaritalStatus'], input_data['HasMortgage'], input_data['HasDependents'],
            input_data['LoanPurpose'], input_data['HasCoSigner'],
            int(prediction_result)  # Explicitly cast to a Python integer
        ]

        cursor.execute(insert_query, values_to_insert)
        conn.commit()
        st.success("Prediction logged to database successfully!")

    except Error as e:
        st.error(f"Error while connecting to MySQL or inserting data: {e}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

# --- Streamlit UI ---
st.set_page_config(page_title="Loan Default Prediction", layout="centered")

st.title("Loan Default Prediction Interface")

st.markdown("""
Enter the details below to get a real-time prediction on the likelihood of a loan default.
""")

# Store user inputs
input_data = {}

# --- User Input Widgets ---
st.subheader("Financial & Personal Details")

# Input fields for numerical features, including the new 'DebtAmount'
col1, col2, col3 = st.columns(3)
with col1:
    input_data['Age'] = st.number_input("Age", min_value=18, max_value=100, value=35)
    input_data['Income'] = st.number_input("Income", min_value=0, value=60000)
    input_data['LoanAmount'] = st.number_input("Loan Amount", min_value=0, value=250000)

with col2:
    input_data['CreditScore'] = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    input_data['MonthsEmployed'] = st.number_input("Months Employed", min_value=0, value=48)
    input_data['NumCreditLines'] = st.number_input("Number of Credit Lines", min_value=0, value=2)

with col3:
    input_data['InterestRate'] = st.number_input("Interest Rate (%)", min_value=0.0, value=7.5, format="%.2f")
    input_data['LoanTerm'] = st.number_input("Loan Term (months)", min_value=12, value=36, step=12)
    input_data['DTIRatio'] = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.4, format="%.2f")
    input_data['DebtAmount'] = st.number_input("Debt Amount", min_value=0, value=20000)

st.subheader("Other Attributes")
col4, col5 = st.columns(2)
with col4:
    input_data['Education'] = st.selectbox("Education", options=['High School', "Bachelor's", "Master's", "PhD"])
    input_data['EmploymentType'] = st.selectbox("Employment Type", options=['Full-time', 'Unemployed', 'Self-employed', 'Part-time'])
    input_data['MaritalStatus'] = st.selectbox("Marital Status", options=['Married', 'Single', 'Divorced'])

with col5:
    input_data['LoanPurpose'] = st.selectbox("Loan Purpose", options=['Other', 'Auto', 'Business', 'Home', 'Education'])
    input_data['HasMortgage'] = st.selectbox("Has Mortgage?", options=['No', 'Yes'])
    input_data['HasDependents'] = st.selectbox("Has Dependents?", options=['No', 'Yes'])
    input_data['HasCoSigner'] = st.selectbox("Has Co-Signer?", options=['No', 'Yes'])


# --- Prediction Button ---
if st.button("Predict Loan Default"):
    if model is None:
        st.error("Cannot make prediction: Model or scaler not loaded.")
    else:
        # 1. Create a DataFrame for user input
        user_input_df = pd.DataFrame([input_data])

        # 2. Convert categorical inputs to the format the model expects
        
        # Initialize a dictionary for one-hot encoded and numerical values
        model_input = {field: 0 for field in model_input_fields}

        # Transfer numerical values directly
        for col in numerical_cols:
            model_input[col] = user_input_df[col].iloc[0]

        # Use label mapping for the single categorical EmploymentType column
        model_input['EmploymentType'] = label_map['EmploymentType'][user_input_df['EmploymentType'].iloc[0]]

        # Convert other categorical choices to one-hot encoded format
        education_suffix = user_input_df['Education'].iloc[0].replace(' ', '').replace("'", '')
        model_input[f"Education_{education_suffix}"] = 1
        
        marital_suffix = user_input_df['MaritalStatus'].iloc[0]
        model_input[f"MaritalStatus_{marital_suffix}"] = 1
        
        loan_purpose_suffix = user_input_df['LoanPurpose'].iloc[0]
        model_input[f"LoanPurpose_{loan_purpose_suffix}"] = 1
        
        # Handle binary inputs
        model_input['HasMortgage'] = label_map['HasMortgage'][user_input_df['HasMortgage'].iloc[0]]
        model_input['HasDependents'] = label_map['HasDependents'][user_input_df['HasDependents'].iloc[0]]
        model_input['HasCoSigner'] = label_map['HasCoSigner'][user_input_df['HasCoSigner'].iloc[0]]


        # Create the final DataFrame for the model
        model_input_df = pd.DataFrame([model_input], columns=model_input_fields)
    

        # 3. Make prediction
        try:
            # The model expects a 2D array, even for a single prediction
            prediction_result = model.predict(model_input_df.values)[0]

            if prediction_result == 1:
                st.error("Prediction: Customer is likely to **DEFAULT** on the loan.")
            else:
                st.success("Prediction: Customer is likely **NOT** to default on the loan.")

            # 4. Insert prediction into MySQL
            insert_prediction_into_db(input_data, prediction_result)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.info("Please check if the model expects scaled input and if all required features are present.")
