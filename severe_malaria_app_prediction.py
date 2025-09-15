import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('severe_malaria_prediction_app.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Malaria Prediction App")

# Collect inputs from user
age = st.number_input("Age", 1, 100)
sex = st.selectbox("Gender", ["Female", "Male"])
fever = st.selectbox("Fever", ["No", "Yes"])
cold = st.selectbox("Cold", ["No", "Yes"])
rigor = st.selectbox("Rigor", ["No", "Yes"])
fatigue = st.selectbox("Fatigue", ["No", "Yes"])
headace = st.selectbox("Headache", ["No", "Yes"])
bitter_tongue = st.selectbox("Bitter Tongue", ["No", "Yes"])
vomitting = st.selectbox("Vomiting", ["No", "Yes"])
diarrhea = st.selectbox("Diarrhea", ["No", "Yes"])
Convulsion = st.selectbox("Convulsion", ["No", "Yes"])
Anemia = st.selectbox("Anemia", ["No", "Yes"])
jundice = st.selectbox("Jaundice", ["No", "Yes"])
cocacola_urine = st.selectbox("Coca-Cola Urine", ["No", "Yes"])
hypoglycemia = st.selectbox("Hypoglycemia", ["No", "Yes"])
prostraction = st.selectbox("Prostration", ["No", "Yes"])
hyperpyrexia = st.selectbox("Hyperpyrexia", ["No", "Yes"])

if st.button("Predict"):
    # Convert categorical inputs to numeric
    sex = 0 if sex == "Female" else 1
    fever = 1 if fever == "Yes" else 0
    cold = 1 if cold == "Yes" else 0
    rigor = 1 if rigor == "Yes" else 0
    fatigue = 1 if fatigue == "Yes" else 0
    headace = 1 if headace == "Yes" else 0
    bitter_tongue = 1 if bitter_tongue == "Yes" else 0
    vomitting = 1 if vomitting == "Yes" else 0
    diarrhea = 1 if diarrhea == "Yes" else 0
    Convulsion = 1 if Convulsion == "Yes" else 0
    Anemia = 1 if Anemia == "Yes" else 0
    jundice = 1 if jundice == "Yes" else 0
    cocacola_urine = 1 if cocacola_urine == "Yes" else 0
    hypoglycemia = 1 if hypoglycemia == "Yes" else 0
    prostraction = 1 if prostraction == "Yes" else 0
    hyperpyrexia = 1 if hyperpyrexia == "Yes" else 0


    # Arrange features in correct order by copying our result from print(X.columns.tolist())
    features = np.array([[age, sex, fever, cold, rigor, fatigue, headace,
                          bitter_tongue, vomitting, diarrhea, Convulsion, Anemia,
                          jundice, cocacola_urine, hypoglycemia, prostraction,
                          hyperpyrexia]])
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    # Display result
    st.success("✅ Positive for Severe Malaria" if prediction == 1 else "❌ Negative for Severe Malaria")