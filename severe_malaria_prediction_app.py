import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -------------------------
# Load model and scaler
# -------------------------
model = joblib.load('severe_malaria_prediction_app.pkl')
scaler = joblib.load('scaler.pkl')

# -------------------------
# Page Title
# -------------------------
st.set_page_config(
    page_title="Malaria Prediction System",
    layout="wide"
)

st.markdown("<h1 style='text-align: center; color: #2E8B57;'>Malaria Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Early Detection Based on Patient Symptoms</p>", unsafe_allow_html=True)

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("Patient Information")

age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=20)
sex = st.sidebar.selectbox("Gender", ["Female", "Male"])
sex_encoded = 1 if sex == "Male" else 0

st.sidebar.header("Symptoms (Yes/No)")
symptom_names = [
    "Fever", "Cold", "Rigor", "Fatigue", "Headache", "Bitter Tongue",
    "Vomiting", "Diarrhea", "Convulsion", "Anemia", "Jaundice",
    "Coca-cola Urine", "Hypoglycemia", "Prostration", "Hyperpyrexia"
]

symptom_values = {}
for symptom in symptom_names:
    symptom_values[symptom] = st.sidebar.selectbox(symptom, ["No", "Yes"], index=0)

# Convert Yes/No to 1/0
symptom_values = {k: 1 if v=="Yes" else 0 for k,v in symptom_values.items()}

# -------------------------
# Prediction Button
# -------------------------
if st.button("Predict Severe Malaria"):

    # Feature Engineering
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

    # Engineered features
    severity_score = sum([
        fatigue, jaundice, prostration, convulsion, cocacola_urine
    ])
    age_group = 0 if age <= 12 else 1 if age <= 30 else 2
    symptom_count = sum(symptom_values.values())
    gi_symptoms = vomiting + diarrhea + bitter_tongue
    neuro_symptoms = headache + convulsion
    age_symptom_interaction = age * symptom_count
    fever_severity = fever * severity_score

    # Feature array in order
    features = np.array([[ 
        age, sex_encoded, fever, cold, rigor, fatigue, headache, bitter_tongue,
        vomiting, diarrhea, convulsion, anemia, jaundice, cocacola_urine,
        hypoglycemia, prostration, hyperpyrexia, severity_score, age_group,
        symptom_count, gi_symptoms, neuro_symptoms, age_symptom_interaction,
        fever_severity
    ]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Prediction
    threshold = 0.4
    probability = model.predict_proba(features_scaled)[0][1]
    prediction = 1 if probability >= threshold else 0

    # Display result
    if prediction == 1:
        st.markdown(f"<h2 style='color:red;'>Predicted Severe Malaria: Positive</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='color:green;'>Predicted Severe Malaria: Negative</h2>", unsafe_allow_html=True)

    # Engineered features summary
    st.markdown("### Engineered Features Summary")
    st.write(f"• Fever Severity: {fever_severity}")
    st.write(f"• Neurological Symptoms Count: {neuro_symptoms}")
    st.write(f"• Gastrointestinal (GI) Symptoms Count: {gi_symptoms}")
    st.write(f"• Severity Score: {severity_score}")
    st.write(f"• Total Symptoms Count: {symptom_count}")

# -------------------------
# Tabs: About Model and User Guide
# -------------------------
tab1, tab2 = st.tabs(["About the Model", "User Guide"])

with tab1:
    st.markdown("### Model Information")
    st.write("**Model Type:** Random Forest Classifier")
    st.write("**Number of Training Samples:** 337")
    st.write("**Number of Features:** 24 (including engineered features)")
    st.write("""
**Key Features Used:**
- Fever Severity
- Neurological Symptoms
- Gastrointestinal Symptoms
- Severity Score
- Total Symptoms Count
- Age Group
""")

with tab2:
    st.markdown("### How to Use This Tool")
    st.write("""
1. **Enter Patient Information:** Input age and select gender in the sidebar.
2. **Select Symptoms:** Choose 'Yes' for all symptoms present, 'No' otherwise.
3. **Predict:** Click the 'Predict Severe Malaria' button.
4. **Interpret Results:**  
   - **Positive:** Red, likely severe malaria → seek immediate medical care.  
   - **Negative:** Green, unlikely severe malaria → monitor symptoms.
5. **Check Engineered Features:** View counts for Fever Severity, Neuro, GI, Severity Score, and total symptoms.
""")
