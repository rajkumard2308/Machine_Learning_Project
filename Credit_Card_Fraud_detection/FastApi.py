from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Load model & scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize FastAPI
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Detect fraudulent credit card transactions",
    version="1.0"
)

# -------------------------------
# Input Schema
# -------------------------------
class Transaction(BaseModel):
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

# -------------------------------
# Health Check
# -------------------------------
@app.get("/")
def health_check():
    return {"status": "Fraud Detection API is running"}

# -------------------------------
# Prediction Endpoint
# -------------------------------
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    
    # Convert input to DataFrame
    input_data = pd.DataFrame([transaction.dict()])

    # Scale Time
    input_data[['Amount']] = scaler.transform(input_data[['Amount']])

    # FORCE EXACT TRAINING ORDER
    input_data = input_data[model.feature_names_in_]

    # Predict
    fraud_prob = model.predict_proba(input_data)[0][1]
    prediction = "FRAUD" if fraud_prob > 0.3 else "LEGIT"

    return {
        "prediction": prediction,
        "fraud_probability": round(float(fraud_prob), 4)
    }
