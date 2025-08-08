# pages/1_Prediction_UI.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import mysql.connector
from mysql.connector import Error

# --- Configuration ---
# MySQL connection config
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'testing'
}

# --- Updated field lists to match spark_cleaned_data.csv ---
model_input_fields = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'DebtAmount',
    'EmploymentType', 'HasMortgage', 'HasDependents', 'HasCoSigner',
    'LoanPurpose_Education', 'LoanPurpose_Home', 'LoanPurpose_Other',
    'LoanPurpose_Auto', 'LoanPurpose_Business', 'MaritalStatus_Married',
    'MaritalStatus_Single', 'MaritalStatus_Divorced', 'Education_HighSchool',
    'Education_Bachelor', 'Education_Master', 'Education_PhD'
]

numerical_cols = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'DebtAmount'
]

label_map = {
    'Education': {"Bachelor's": 'Bachelor', "Master's": 'Master', 'High School': 'HighSchool', 'PhD': 'PhD'},
    'EmploymentType': {'Full-time': 4, 'Unemployed': 0, 'Self-employed': 1, 'Part-time': 2},
    'MaritalStatus': {'Divorced': 'Divorced', 'Married': 'Married', 'Single': 'Single'},
    'HasMortgage': {'Yes': 1, 'No': 0},
    'HasDependents': {'Yes': 1, 'No': 0},
    'LoanPurpose': {'Other': 'Other', 'Auto': 'Auto', 'Business': 'Business', 'Home': 'Home', 'Education': 'Education'},
    'HasCoSigner': {'Yes': 1, 'No': 0},
}

# Risk categories with thresholds
RISK_CATEGORIES = {
    'High Risk': (0.7, 1.0),
    'Medium Risk': (0.4, 0.7),
    'Low Risk': (0.1, 0.4),
    'Minimal Risk': (0.0, 0.1)
}

# --- Function to load model and scaler ---
@st.cache_resource
def load_model():
    try:
        with open('loan_default_lgbm_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Please ensure 'loan_default_lgbm_model.pkl' is in the root directory.")
        return None

model = load_model()

# --- Function to insert prediction into MySQL database ---
def insert_prediction_into_db(input_data, prediction_prob, risk_category):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        columns = [
            'prediction_timestamp', 'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
            'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'DebtAmount', 'Education',
            'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents',
            'LoanPurpose', 'HasCoSigner', 'PredictionProbability', 'RiskCategory'
        ]
        
        placeholders = ', '.join(['%s'] * len(columns))
        insert_query = f"INSERT INTO predictions_log ({', '.join(columns)}) VALUES ({placeholders})"

        values_to_insert = [
            pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            input_data['Age'], input_data['Income'], input_data['LoanAmount'],
            input_data['CreditScore'], input_data['MonthsEmployed'], input_data['NumCreditLines'],
            input_data['InterestRate'], input_data['LoanTerm'], input_data['DTIRatio'],
            input_data['DebtAmount'], input_data['Education'], input_data['EmploymentType'],
            input_data['MaritalStatus'], input_data['HasMortgage'], input_data['HasDependents'],
            input_data['LoanPurpose'], input_data['HasCoSigner'],
            float(prediction_prob),
            risk_category
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

# --- Function to determine risk category ---
def get_risk_category(probability):
    for category, (lower, upper) in RISK_CATEGORIES.items():
        if lower <= probability < upper:
            return category
    return 'Unknown Risk'

# --- Streamlit UI ---
st.set_page_config(page_title="Loan Default Prediction", layout="centered")

st.title("Loan Default Risk Assessment")

st.markdown("""
Enter the details below to assess the risk of loan default with probability scores.
""")

# Store user inputs
input_data = {}

# --- User Input Widgets ---
st.subheader("Financial & Personal Details")

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
if st.button("Assess Default Risk"):
    if model is None:
        st.error("Cannot make prediction: Model not loaded.")
    else:
        # 1. Create a DataFrame for user input
        user_input_df = pd.DataFrame([input_data])

        # 2. Convert categorical inputs to the format the model expects
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
            # Get probability scores
            probability = model.predict_proba(model_input_df.values)[0][1]
            risk_category = get_risk_category(probability)
            
            # Display results with color coding
            st.subheader("Risk Assessment Results")
            
            # Create a progress bar for visualization
            st.progress(probability)
            
            # Display probability and risk category
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Default Probability", f"{probability*100:.2f}%")
            with col2:
                if risk_category == 'High Risk':
                    st.error(f"Risk Category: {risk_category}")
                elif risk_category == 'Medium Risk':
                    st.warning(f"Risk Category: {risk_category}")
                else:
                    st.success(f"Risk Category: {risk_category}")
            
            # Provide interpretation
            st.markdown("### Interpretation")
            if risk_category == 'High Risk':
                st.error("""
                🚨 **High Risk Warning**: This applicant has a high probability of default. 
                - Consider additional collateral or higher interest rates
                - Recommend thorough manual review
                - May require cosigner or additional guarantees
                """)
            elif risk_category == 'Medium Risk':
                st.warning("""
                ⚠️ **Medium Risk**: This applicant has moderate default risk.
                - Standard underwriting procedures recommended
                - May benefit from slightly higher interest rates
                - Consider additional verification of income/employment
                """)
            elif risk_category == 'Low Risk':
                st.info("""
                ℹ️ **Low Risk**: This applicant has acceptable default risk.
                - Standard loan terms appropriate
                - Normal underwriting procedures
                - Generally favorable candidate
                """)
            else:  # Minimal Risk
                st.success("""
                ✅ **Minimal Risk**: This applicant has very low default probability.
                - Consider preferential rates/terms
                - Fast-track approval possible
                - Excellent candidate for loan
                """)
            
            # 4. Insert prediction into MySQL
            insert_prediction_into_db(input_data, probability, risk_category)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.info("Please check if the model expects all required features to be present.")