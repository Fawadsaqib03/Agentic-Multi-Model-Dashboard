import pickle
import numpy as np
import json
import re
from langchain.tools import tool

# Load models at startup
with open("models/churn_model.pkl", "rb") as f: churn_bundle = pickle.load(f)
with open("models/diabetes_model.pkl", "rb") as f: diab_bundle = pickle.load(f)
with open("models/spam_model.pkl", "rb") as f: spam_bundle = pickle.load(f)

# Feature names for reference
CHURN_FEATURES = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                  'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                  'Geography_Germany', 'Geography_Spain', 'Gender_Male']

DIABETES_FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Median defaults (from real dataset distributions)
CHURN_DEFAULTS = [650.0, 39.0, 5.0, 76485.0, 1.0, 1.0, 1.0, 100090.0, 0.0, 0.0, 1.0]
DIABETES_DEFAULTS = [3.0, 117.0, 72.0, 23.0, 30.5, 32.0, 0.3725, 29.0]


def _parse_features(raw_input, expected_len, defaults):
    """Robustly parse feature input from the LLM.
    Handles: list, JSON string, comma-separated string, or partial values."""
    features = None
    
    # If already a list, use directly
    if isinstance(raw_input, list):
        features = [float(x) for x in raw_input]
    elif isinstance(raw_input, str):
        cleaned = raw_input.strip()
        # Try JSON array
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                features = [float(x) for x in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Try extracting all numbers from string
        if features is None:
            numbers = re.findall(r"[-+]?\d*\.?\d+", cleaned)
            if numbers:
                features = [float(x) for x in numbers]
    
    if features is None:
        features = []
    
    # Pad with defaults if too short
    if len(features) < expected_len:
        for i in range(len(features), expected_len):
            features.append(defaults[i])
    
    return features[:expected_len]


@tool
def predict_churn(features: list[float]) -> str:
    """Predict if a bank customer will CHURN (leave) or STAY using a Neural Network.
    Input MUST be a JSON object with a single key 'features' containing a list of exactly 11 numeric values:
    [CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard(0/1), IsActiveMember(0/1), EstimatedSalary, Geography_Germany(0/1), Geography_Spain(0/1), Gender_Male(0/1)]
    
    Median defaults if missing: CreditScore=650, Age=39, Tenure=5, Balance=76485, NumOfProducts=1, HasCrCard=1, IsActiveMember=1, EstimatedSalary=100090, Geography_Germany=0, Geography_Spain=0, Gender_Male=1
    
    Example: {"features": [650, 42, 5, 75000, 1, 1, 1, 60000, 1, 0, 1]}"""
    parsed = _parse_features(features, 11, CHURN_DEFAULTS)
    X = np.array(parsed).reshape(1, -1)
    X_sc = churn_bundle["scaler"].transform(X)
    pred = churn_bundle["model"].predict(X_sc)[0]
    proba = churn_bundle["model"].predict_proba(X_sc)[0]
    
    label = "CHURN (Customer will leave)" if pred == 1 else "STAY (Customer will remain)"
    feature_report = ", ".join([f"{n}={v}" for n, v in zip(CHURN_FEATURES, parsed)])
    
    return (f"Model: Neural Network (MLP) | Prediction: {label} | "
            f"Churn Probability: {proba[1]*100:.1f}% | Stay Probability: {proba[0]*100:.1f}% | "
            f"Features Used: [{feature_report}]")


@tool
def predict_diabetes(features: list[float]) -> str:
    """Predict if a patient is DIABETIC or HEALTHY using Logistic Regression.
    Input MUST be a JSON object with a single key 'features' containing a list of exactly 8 numeric values:
    [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    
    Median defaults if missing: Pregnancies=3, Glucose=117, BloodPressure=72, SkinThickness=23, Insulin=30.5, BMI=32, Pedigree=0.3725, Age=29
    
    Example: {"features": [6, 148, 72, 35, 0, 33.6, 0.627, 50]}"""
    parsed = _parse_features(features, 8, DIABETES_DEFAULTS)
    X = np.array(parsed).reshape(1, -1)
    X_sc = diab_bundle["scaler"].transform(X)
    pred = diab_bundle["model"].predict(X_sc)[0]
    proba = diab_bundle["model"].predict_proba(X_sc)[0]
    
    label = "DIABETIC (Positive)" if pred == 1 else "HEALTHY (Negative)"
    feature_report = ", ".join([f"{n}={v}" for n, v in zip(DIABETES_FEATURES, parsed)])
    
    return (f"Model: Logistic Regression | Prediction: {label} | "
            f"Diabetes Probability: {proba[1]*100:.1f}% | Healthy Probability: {proba[0]*100:.1f}% | "
            f"Features Used: [{feature_report}]")


@tool
def detect_spam(text: str) -> str:
    """Detect if a text message or email is SPAM or HAM (safe) using SVM.
    Input MUST be a JSON object with a single key 'text' containing the string to classify.
    Example: {"text": "Congratulations! You have won a free iPhone. Call now!"}"""
    X_vec = spam_bundle["vectorizer"].transform([text])
    pred = spam_bundle["model"].predict(X_vec)[0]
    proba = spam_bundle["model"].predict_proba(X_vec)[0]
    
    label = "SPAM (Junk/Malicious)" if pred == 1 else "HAM (Safe/Legitimate)"
    
    return (f"Model: SVM (Support Vector Machine) | Prediction: {label} | "
            f"Spam Probability: {proba[1]*100:.1f}% | Ham Probability: {proba[0]*100:.1f}% | "
            f"Text Analyzed: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")


ALL_TOOLS = [predict_churn, predict_diabetes, detect_spam]