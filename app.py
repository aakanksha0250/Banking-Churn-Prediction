import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the trained model (ensure you save your trained model using joblib or pickle)
# For example: joblib.dump(model, 'rf_model.pkl')
model = joblib.load('rf_model.pkl')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Define the features
features = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
            'Geography_Germany', 'Geography_Spain']

def main():
    st.title("Customer Churn Prediction")
    st.header("Enter customer details to predict if they will churn")

    # User input
    CreditScore = st.number_input('Credit Score', min_value=0, max_value=1000, value=600)
    Gender = st.selectbox('Gender', ('Male', 'Female'))
    Age = st.number_input('Age', min_value=18, max_value=100, value=30)
    Tenure = st.number_input('Tenure', min_value=0, max_value=10, value=3)
    Balance = st.number_input('Balance', min_value=0.0, max_value=1000000.0, value=1000.0)
    NumOfProducts = st.number_input('Number of Products', min_value=1, max_value=10, value=1)
    HasCrCard = st.selectbox('Has Credit Card?', ('Yes', 'No'))
    IsActiveMember = st.selectbox('Is Active Member?', ('Yes', 'No'))
    EstimatedSalary = st.number_input('Estimated Salary', min_value=0.0, max_value=1000000.0, value=50000.0)
    Geography = st.selectbox('Geography', ('France', 'Germany', 'Spain'))

    # Convert categorical data to numeric
    Gender = 1 if Gender == 'Male' else 0
    HasCrCard = 1 if HasCrCard == 'Yes' else 0
    IsActiveMember = 1 if IsActiveMember == 'Yes' else 0
    Geography_Germany = 1 if Geography == 'Germany' else 0
    Geography_Spain = 1 if Geography == 'Spain' else 0

    # Create a numpy array of the features
    user_data = np.array([[CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, 
                           HasCrCard, IsActiveMember, EstimatedSalary, 
                           Geography_Germany, Geography_Spain]])

    # Scale the user data
    user_data = scaler.transform(user_data)

    # Predict
    if st.button('Predict'):
        prediction = model.predict(user_data)
        if prediction[0] == 1:
            st.warning("The customer is likely to churn.")
        else:
            st.success("The customer is not likely to churn.")

if __name__ == '__main__':
    main()
