import streamlit as st
import numpy as np
import joblib

# -------------------------
# Load model and scaler
# -------------------------
model = joblib.load('severe_malaria_prediction_app.pkl')
scaler = joblib.load('scaler.pkl')

# -------------------------
# Page Title
# -------------------------
st.title("Malaria Prediction App (Severe Malaria Risk)")

# Display model used
st.markdown("**Model Used:** Random Forest Classifier")

# -------------------------
# User Inputs
# -------------------------
age = st.number_input("Age", min_value=0, max_value=100, value=20)
sex = st.selectbox("Gender", ["Female", "Male"])
sex_encoded = 1 if sex == "Male" else 0

# Symptoms (1 = Yes, 0 = No)
symptom_names = [
    "Fever", "Cold", "Rigor", "Fatigue", "Headache", "Bitter Tongue",
    "Vomiting", "Diarrhea", "Convulsion", "Anemia", "Jaundice",
    "Coca-cola Urine", "Hypoglycemia", "Prostration", "Hyperpyrexia"
]

symptom_values = {}
for symptom in symptom_names:
    symptom_values[symptom] = st.selectbox(symptom, [0, 1], index=0)

# -------------------------
# Explain Confidence Interval
# -------------------------
st.markdown("""
**Confidence Interval:** In this context, it shows how confident the model is that a patient has severe malaria. 
For example, a probability of 0.75 means the model predicts a 75% chance of severe malaria based on the symptoms provided.
""")

# -------------------------
# Prediction Button
# -------------------------
if st.button("Predict Severe Malaria"):
    
    # -------------------------
    # Feature Engineering
    # -------------------------
    fever = symptom_values["Fever"]
    cold = symptom_values["Cold"]
    rigor = symptom_values["Rigor"]
    fatigue = symptom_values["Fatigue"]
    headache = symptom_values["Headache"]
    bitter_tongue = symptom_values["Bitter Tongue"]
    vomiting = symptom_values["Vomiting"]
    diarrhea = symptom_values["Diarrhea"]
    convulsion = symptom_values["Convulsion"]
    anemia = symptom_values["Anemia"]
    jaundice = symptom_values["Jaundice"]
    cocacola_urine = symptom_values["Coca-cola Urine"]
    hypoglycemia = symptom_values["Hypoglycemia"]
    prostration = symptom_values["Prostration"]
    hyperpyrexia = symptom_values["Hyperpyrexia"]

    severity_score = sum([
        cold, cocacola_urine, prostration, convulsion,
        hypoglycemia, anemia, jaundice, hyperpyrexia
    ])
    age_group = 0 if age <= 12 else 1 if age <= 30 else 2
    symptom_count = sum(symptom_values.values())
    gi_symptoms = vomiting + diarrhea + bitter_tongue
    neuro_symptoms = headache + convulsion
    age_symptom_interaction = age * symptom_count
    fever_severity = fever * severity_score

    # -------------------------
    # Create feature array in exact order
    # -------------------------
    features = np.array([[ 
        age, sex_encoded, fever, cold, rigor, fatigue, headache, bitter_tongue,
        vomiting, diarrhea, convulsion, anemia, jaundice, cocacola_urine,
        hypoglycemia, prostration, hyperpyrexia, severity_score, age_group,
        symptom_count, gi_symptoms, neuro_symptoms, age_symptom_interaction,
        fever_severity
    ]])

    # -------------------------
    # Scale features
    # -------------------------
    features_scaled = scaler.transform(features)

    # -------------------------
    # Make prediction
    # -------------------------
    threshold = 0.4  # fixed threshold
    probability = model.predict_proba(features_scaled)[0][1]
    prediction = 1 if probability >= threshold else 0

    # -------------------------
    # Display result
    # -------------------------
    st.subheader("Prediction Result")
    st.write(f"Predicted Severe Malaria Status: **{'Positive' if prediction == 1 else 'Negative'}**")
    st.write(f"Probability of Severe Malaria (Confidence Interval): **{probability:.2%}**")
    st.write(f"Threshold used: **{threshold:.0%}**")
