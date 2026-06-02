import json
import pickle
import pandas as pd
from datetime import datetime
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Hospital Readmission Server")

with open('data/model.pkl', 'rb') as f:
    model = pickle.load(f)

prediction_logs = []

@mcp.tool()
def predict_readmission(
    age: int,
    gender: str,
    length_of_stay: int,
    num_prev_admissions: int,
    num_medications: int,
    num_diagnoses: int,
    num_lab_procedures: int,
    insulin: str,
    diabetic: int,
    heart_disease: int
) -> dict:
    """Predict hospital readmission risk for a patient"""
    features = [
        age,
        1 if gender == 'M' else 0,
        length_of_stay,
        num_prev_admissions,
        num_medications,
        num_diagnoses,
        num_lab_procedures,
        {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}[insulin],
        diabetic,
        heart_disease
    ]

    prob = model.predict_proba([features])[0][1]
    risk = "HIGH" if prob >= 0.5 else "LOW"

    result = {
        "patient_age": age,
        "risk_level": risk,
        "readmission_probability": round(prob * 100, 1),
        "timestamp": datetime.now().isoformat()
    }

    prediction_logs.append(result)
    return result

@mcp.tool()
def get_prediction_logs() -> list:
    """Get all past prediction logs"""
    return prediction_logs

@mcp.tool()
def get_high_risk_patients() -> list:
    """Get only high risk patients from logs"""
    return [p for p in prediction_logs if p['risk_level'] == 'HIGH']

@mcp.tool()
def get_statistics() -> dict:
    """Get statistics of all predictions made"""
    if not prediction_logs:
        return {"message": "No predictions yet"}
    
    total = len(prediction_logs)
    high_risk = len([p for p in prediction_logs if p['risk_level'] == 'HIGH'])
    low_risk = total - high_risk
    avg_prob = sum(p['readmission_probability'] for p in prediction_logs) / total

    return {
        "total_predictions": total,
        "high_risk_count": high_risk,
        "low_risk_count": low_risk,
        "average_probability": round(avg_prob, 1)
    }

if __name__ == "__main__":
    mcp.run()