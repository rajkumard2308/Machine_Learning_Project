import streamlit as st
import joblib
import os
import pandas as pd

# ----------------------------------
# Page config (MUST be first)
# ----------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Card Fraud Detection")
st.write("Predict whether a credit card transaction is **FRAUD** or **LEGIT**.")
st.divider()

# ----------------------------------
# Load model & scaler safely
# ----------------------------------
# Get absolute path of current file (app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    st.error(f"Failed to load model files: {e}")
    st.stop()

# ----------------------------------
# Correct feature order (VERY IMPORTANT)
# ----------------------------------
FEATURE_ORDER = [
    "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

# ----------------------------------
# Sidebar Inputs
# ----------------------------------
st.sidebar.header("🧾 Transaction Inputs")

# Amount = st.sidebar.number_input("Amount", value=100.0)

# V = {}
# for i in range(1, 29):
#     V[f"V{i}"] = st.sidebar.number_input(
#         f"V{i}", value=0.0
#     )

Amount = st.sidebar.number_input("Amount", value=2450.75)

default_V = {
    "V1": -4.12,  "V2": 2.56,   "V3": -3.89, "V4": 1.75,
    "V5": -2.44,  "V6": -1.98,  "V7": -3.21, "V8": 0.88,
    "V9": -2.65,  "V10": -4.77, "V11": 3.10, "V12": -5.60,
    "V13": 0.45,  "V14": -6.25, "V15": 0.12, "V16": -3.85,
    "V17": -4.10, "V18": -1.95, "V19": 0.78, "V20": 1.35,
    "V21": 0.42,  "V22": -0.55, "V23": -0.21, "V24": -0.68,
    "V25": 0.32,  "V26": 0.18,  "V27": 0.95,  "V28": 0.22
}

V = {}
for i in range(1, 29):
    key = f"V{i}"
    V[key] = st.sidebar.number_input(
        key,
        value=default_V[key]
    )

predict_btn = st.sidebar.button("🔍 Predict Fraud")

# ----------------------------------
# Prediction Logic
# ----------------------------------
if predict_btn:
    try:
        # Create input DataFrame
        input_data = {"Amount": Amount}
        input_data.update(V)
        input_df = pd.DataFrame([input_data])

        # Scale Amount
        input_df[["Amount"]] = scaler.transform(input_df[["Amount"]])

        # Enforce training feature order
        input_df = input_df[FEATURE_ORDER]

        # Predict
        fraud_prob = model.predict_proba(input_df)[0][1]
        prediction = "FRAUD" if fraud_prob > 0.3 else "LEGIT"

        st.subheader("📊 Prediction Result")

        if prediction == "FRAUD":
            st.error(f"🚨 **FRAUD DETECTED**")
        else:
            st.success(f"✅ **LEGIT TRANSACTION**")

        st.metric(
            label="Fraud Probability",
            value=f"{fraud_prob:.4f}"
        )

    except Exception as e:
        st.error(f" Prediction failed: {e}")
