import os
import pickle
import numpy as np
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

with open('data/model.pkl', 'rb') as f:
    model = pickle.load(f)

def analyze_patient(patient_data: dict) -> dict:
    features = [
        patient_data['age'],
        1 if patient_data['gender'] == 'M' else 0,
        patient_data['length_of_stay'],
        patient_data['num_prev_admissions'],
        patient_data['num_medications'],
        patient_data['num_diagnoses'],
        patient_data['num_lab_procedures'],
        {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}[patient_data['insulin']],
        patient_data['diabetic'],
        patient_data['heart_disease']
    ]

    prob = model.predict_proba([features])[0][1]
    risk = "HIGH" if prob >= 0.5 else "LOW"

    prompt = f"""You are a clinical AI assistant helping doctors.
A patient has been analyzed with the following details:
- Age: {patient_data['age']}
- Gender: {patient_data['gender']}
- Length of Stay: {patient_data['length_of_stay']} days
- Previous Admissions: {patient_data['num_prev_admissions']}
- Number of Medications: {patient_data['num_medications']}
- Number of Diagnoses: {patient_data['num_diagnoses']}
- Lab Procedures: {patient_data['num_lab_procedures']}
- Insulin: {patient_data['insulin']}
- Diabetic: {'Yes' if patient_data['diabetic'] else 'No'}
- Heart Disease: {'Yes' if patient_data['heart_disease'] else 'No'}

ML Model Prediction: {risk} RISK of readmission ({prob*100:.1f}% probability)

Please provide:
1. Brief clinical reasoning for this risk level
2. Top 3 risk factors for this patient
3. Recommended actions before discharge
Keep it concise and professional."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    return {
        "risk_level": risk,
        "probability": round(prob * 100, 1),
        "agent_analysis": response.choices[0].message.content
    }