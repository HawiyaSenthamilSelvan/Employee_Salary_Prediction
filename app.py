import streamlit as st
import pandas as pd
import pickle

# Load saved encoders, scaler, and model
with open("encoders_salary.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("scaler_salary.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("emp_sal_pred_model", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
feature_names = model_data["Features_names"]

st.title("👨‍💼 Employee Salary Predictor")

st.write("Enter the following details to predict whether the salary is >50K or <=50K")

# Input widgets
age = st.slider("Age", 18, 90, 30)
workclass = st.selectbox("Workclass", encoders['workclass'].classes_)
education = st.selectbox("Education", encoders['education'].classes_)
education_num = st.slider("Education Num", 1, 16, 9)
marital_status = st.selectbox("Marital Status", encoders['marital.status'].classes_)
occupation = st.selectbox("Occupation", encoders['occupation'].classes_)
relationship = st.selectbox("Relationship", encoders['relationship'].classes_)
race = st.selectbox("Race", encoders['race'].classes_)
sex = st.selectbox("Sex", encoders['sex'].classes_)
capital_gain = st.number_input("Capital Gain", min_value=0, step=1)
capital_loss = st.number_input("Capital Loss", min_value=0, step=1)
hours_per_week = st.slider("Hours per Week", 1, 100, 40)
native_country = st.selectbox("Native Country", encoders['native.country'].classes_)

# Predict button
if st.button("Predict Salary"):
    # Prepare input
    input_dict = {
        'age': age,
        'workclass': workclass,
        'education': education,
        'education.num': education_num,
        'marital.status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital.gain': capital_gain,
        'capital.loss': capital_loss,
        'hours.per.week': hours_per_week,
        'native.country': native_country
    }

    input_df = pd.DataFrame([input_dict])

    # Encode categorical values
    for col, encoder in encoders.items():
        input_df[col] = encoder.transform(input_df[col])

    # Scale input
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_names)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]

    # Display result
    st.success(f"Prediction: {'High Salary (>50K)' if prediction == 1 else 'Low Salary (<=50K)'}")
    st.info(f"Prediction Probability: {prob}")
