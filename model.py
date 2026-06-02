import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import lightgbm as lgb
import pickle

def generate_sample_data(n=2000):
    np.random.seed(42)
    age = np.random.randint(20, 90, n)
    num_prev_admissions = np.random.randint(0, 10, n)
    length_of_stay = np.random.randint(1, 30, n)
    num_medications = np.random.randint(1, 20, n)
    num_diagnoses = np.random.randint(1, 10, n)
    num_lab_procedures = np.random.randint(1, 50, n)
    diabetic = np.random.choice([0, 1], n)
    heart_disease = np.random.choice([0, 1], n)
    gender = np.random.choice(['M', 'F'], n)
    insulin = np.random.choice(['No', 'Steady', 'Up', 'Down'], n)

    # Create realistic readmission logic
    risk_score = (
        (age > 65).astype(int) * 2 +
        (num_prev_admissions > 3).astype(int) * 3 +
        (length_of_stay < 3).astype(int) * 2 +
        (num_medications > 10).astype(int) * 1 +
        (num_diagnoses > 5).astype(int) * 2 +
        diabetic * 2 +
        heart_disease * 2 +
        (num_lab_procedures > 30).astype(int) * 1
    )

    # Readmitted if risk score is high
    readmitted = (risk_score >= 6).astype(int)

    # Add small noise
    noise = np.random.choice([0, 1], n, p=[0.97, 0.03])
    readmitted = np.clip(readmitted + noise, 0, 1)

    data = {
        'age': age,
        'gender': gender,
        'length_of_stay': length_of_stay,
        'num_prev_admissions': num_prev_admissions,
        'num_medications': num_medications,
        'num_diagnoses': num_diagnoses,
        'num_lab_procedures': num_lab_procedures,
        'insulin': insulin,
        'diabetic': diabetic,
        'heart_disease': heart_disease,
        'readmitted': readmitted
    }
    return pd.DataFrame(data)

def train_model():
    print("Generating dataset...")
    df = generate_sample_data(2000)
    df.to_csv('data/patient_data.csv', index=False)
    print("Dataset saved to data/patient_data.csv")

    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    df['insulin'] = le.fit_transform(df['insulin'])

    X = df.drop('readmitted', axis=1)
    y = df['readmitted']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training LightGBM model...")
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=50,
        min_child_samples=10,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n--- Model Performance ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    with open('data/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\nModel saved to data/model.pkl")

    feature_names = X.columns.tolist()
    with open('data/features.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

    return model

if __name__ == "__main__":
    train_model()