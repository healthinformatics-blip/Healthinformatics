import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Severe Malaria Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; font-weight: bold; text-align: center; color: #2E8B57; margin-bottom: 2rem;}
    .sub-header {font-size: 1.5rem; color: #4682B4; margin-bottom: 1rem;}
    .metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #2E8B57;}
    .warning-box {background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('severe_malaria_prediction_app.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, scaler = load_model()

# Symptom labels
symptom_labels = {
    'fever': 'Fever',
    'cold': 'Cold',
    'rigor': 'Rigor/Chills',
    'fatigue': 'Fatigue',
    'headache': 'Headache',
    'bitter_tongue': 'Bitter Tongue',
    'vomiting': 'Vomiting',
    'diarrhea': 'Diarrhea',
    'convulsion': 'Convulsion',
    'anemia': 'Anemia',
    'jaundice': 'Jaundice',
    'cocacola_urine': 'Dark/Coca-Cola Urine',
    'hypoglycemia': 'Hypoglycemia',
    'prostration': 'Prostration',
    'hyperpyrexia': 'Hyperpyrexia'
}

# Sidebar: Inputs
st.sidebar.markdown('<h2 class="sub-header">Patient Information</h2>', unsafe_allow_html=True)
age = st.sidebar.slider("Age", 0, 100, 20)
sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
st.sidebar.markdown("### Symptoms (Check all present)")
symptoms = {k: int(st.sidebar.checkbox(v, key=k)) for k, v in symptom_labels.items()}

# Main App
st.markdown('<h1 class="main-header"> Severe Malaria Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Early Detection Based on Patient Symptoms</p>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
    
    if st.button("Analyze Patient"):
        if model is None or scaler is None:
            st.error("Model not loaded. Cannot perform prediction.")
        else:
            # Feature Engineering
            severity_score = sum([symptoms['cold'], symptoms['cocacola_urine'], symptoms['prostration'],
                                  symptoms['convulsion'], symptoms['hypoglycemia'], symptoms['anemia'],
                                  symptoms['jaundice'], symptoms['hyperpyrexia']])
            age_group = 0 if age <= 12 else 1 if age <= 30 else 2
            symptom_count = sum(symptoms.values())
            gi_symptoms = symptoms['vomiting'] + symptoms['diarrhea'] + symptoms['bitter_tongue']
            neuro_symptoms = symptoms['headache'] + symptoms['convulsion']
            age_symptom_interaction = age * symptom_count
            fever_severity = symptoms['fever'] * severity_score
            
            feature_array = np.array([[age, sex,
                                       symptoms['fever'], symptoms['cold'], symptoms['rigor'],
                                       symptoms['fatigue'], symptoms['headache'], symptoms['bitter_tongue'],
                                       symptoms['vomiting'], symptoms['diarrhea'], symptoms['convulsion'],
                                       symptoms['anemia'], symptoms['jaundice'], symptoms['cocacola_urine'],
                                       symptoms['hypoglycemia'], symptoms['prostration'], symptoms['hyperpyrexia'],
                                       severity_score, age_group, symptom_count, gi_symptoms,
                                       neuro_symptoms, age_symptom_interaction, fever_severity]])
            
            try:
                features_scaled = scaler.transform(feature_array)
                probability = model.predict_proba(features_scaled)[0][1]

                # Fixed threshold
                threshold = 0.4
                prediction = 1 if probability >= threshold else 0

                # Risk display and color
                if probability < threshold:
                    risk_color = "green"
                    risk_text = "LOW RISK"
                else:
                    risk_color = "red"
                    risk_text = "HIGH RISK"

                if prediction == 1:
                    st.error(f"{risk_text}: Severe Malaria Positive (Confidence: {probability:.1%})")
                else:
                    st.success(f"{risk_text}: Severe Malaria Negative (Confidence: {(1-probability):.1%})")

                # Probability gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability*100,
                    domain = {'x':[0,1], 'y':[0,1]},
                    title={'text': "Severe Malaria Risk (%)"},
                    gauge={
                        'axis': {'range': [0,100]},
                        'bar': {'color': risk_color},
                        'steps': [
                            {'range':[0,40], 'color':'green'},
                            {'range':[40,100], 'color':'red'}
                        ],
                        'threshold': {'line': {'color':"black",'width':4}, 'thickness':0.75, 'value': threshold*100}
                    }
                ))
                fig_gauge.update_layout(height=400)
                st.plotly_chart(fig_gauge, use_container_width=True)

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

    # Disclaimer
    st.markdown("""<div class="warning-box"><strong> Medical Disclaimer:</strong> This tool is for educational purposes only. Always consult healthcare professionals for diagnosis and treatment.</div>""", unsafe_allow_html=True)

with col2:
    st.markdown('<h2 class="sub-header">Patient Summary</h2>', unsafe_allow_html=True)
    st.markdown("### Demographics")
    st.write(f"**Age:** {age} years")
    st.write(f"**Sex:** {'Female' if sex==0 else 'Male'}")
    
    st.markdown("### Symptoms Present")
    active_symptoms = [v for k,v in symptom_labels.items() if symptoms[k]==1]
    if active_symptoms:
        for s in active_symptoms:
            st.write(f"• {s}")
        st.metric("Total Symptoms", len(active_symptoms))
        fig_symptoms = px.bar(
            x=list(symptom_labels.values()),
            y=[symptoms[k] for k in symptom_labels.keys()],
            labels={'x':'Symptoms','y':'Present'},
            title="Symptom Profile"
        )
        fig_symptoms.update_layout(height=300, showlegend=False)
        fig_symptoms.update_traces(marker_color=['#ff6b6b' if v==1 else '#e0e0e0' for v in symptoms.values()])
        st.plotly_chart(fig_symptoms, use_container_width=True)
    else:
        st.info("No symptoms selected")

# Tabs
st.markdown("---")
tab_about, tab_more = st.tabs(["About the Model", "Additional Info"])

with tab_about:
    st.markdown("### Model Information")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Model Type", "Random Forest")
    with col2: st.metric("Training Samples", "337")
    with col3: st.metric("Features Used", "24 (with engineered)")
    st.markdown("""
    **Key Features:** 
    - Severity score
    - Age group
    - Symptom count
    - GI and neurological symptoms
    - Individual symptoms
    """)

with tab_more:
    st.markdown("### Notes")
    st.markdown("""
    - This model is intended for educational purposes only.
    - Do not use for actual clinical diagnosis.
    - Always consult healthcare professionals for proper evaluation.
    """)

# Footer
st.markdown("---")
st.markdown("""<div style='text-align:center;color:#666;padding:20px;'>Severe Malaria Prediction System | Educational Use Only</div>""", unsafe_allow_html=True)
