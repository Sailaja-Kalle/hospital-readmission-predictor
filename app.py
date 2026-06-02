import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agent import analyze_patient

st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e0f7ff, #f9e6ff, #fff9e6);
}
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: 800;
    background: linear-gradient(90deg, #0099cc, #cc66ff, #ffcc00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding: 10px 0;
}
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #7a5fa0;
    margin-bottom: 20px;
}
.risk-high {
    background: linear-gradient(135deg, #ffe0e0, #ffcccc);
    border-left: 6px solid #ff4444;
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
}
.risk-low {
    background: linear-gradient(135deg, #e0ffe0, #ccffcc);
    border-left: 6px solid #44bb44;
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
}
.agent-box {
    background: linear-gradient(135deg, #e6f9ff, #f9e6ff);
    border-left: 6px solid #cc66ff;
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
}
.stButton > button {
    background: linear-gradient(90deg, #0099cc, #cc66ff, #ffcc00);
    color: white;
    font-size: 18px;
    font-weight: 700;
    border: none;
    border-radius: 30px;
    padding: 12px 40px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏥 Hospital Readmission Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">✨ Enter patient details and AI will predict readmission risk</div>', unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Patient Information")
    age = st.slider("Age", 20, 90, 50)
    gender = st.selectbox("Gender", ["M", "F"])
    length_of_stay = st.slider("Length of Stay (days)", 1, 30, 5)
    num_prev_admissions = st.slider("Previous Admissions", 0, 10, 1)
    diabetic = st.selectbox("Diabetic", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    st.subheader("🔬 Clinical Details")
    num_medications = st.slider("Number of Medications", 1, 20, 5)
    num_diagnoses = st.slider("Number of Diagnoses", 1, 10, 3)
    num_lab_procedures = st.slider("Lab Procedures", 1, 50, 15)
    insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
    heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.markdown("---")

if st.button("🔍 Predict Readmission Risk"):
    with st.spinner("🤖 AI Agent is analyzing patient data..."):
        patient_data = {
            "age": age,
            "gender": gender,
            "length_of_stay": length_of_stay,
            "num_prev_admissions": num_prev_admissions,
            "num_medications": num_medications,
            "num_diagnoses": num_diagnoses,
            "num_lab_procedures": num_lab_procedures,
            "insulin": insulin,
            "diabetic": diabetic,
            "heart_disease": heart_disease
        }

        result = analyze_patient(patient_data)

    col3, col4 = st.columns(2)
    with col3:
        if result['risk_level'] == 'HIGH':
            st.markdown(f"""
            <div class="risk-high">
                <h2>🔴 HIGH RISK</h2>
                <h3>Readmission Probability: {result['probability']}%</h3>
                <p>This patient has a HIGH risk of being readmitted within 30 days.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
                <h2>🟢 LOW RISK</h2>
                <h3>Readmission Probability: {result['probability']}%</h3>
                <p>This patient has a LOW risk of being readmitted within 30 days.</p>
            </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="agent-box">
            <h3>🤖 AI Agent Analysis</h3>
            <p>{result['agent_analysis']}</p>
        </div>""", unsafe_allow_html=True)

    st.warning("⚠ This is an AI-assisted prediction only. Always consult a qualified doctor.")