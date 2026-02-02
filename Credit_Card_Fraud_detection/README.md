# 💳 Credit Card Fraud Detection – End‑to‑End ML Project

This project demonstrates a **complete real‑world Machine Learning pipeline** for detecting fraudulent credit card transactions. It covers **EDA → Model Training → Handling Imbalanced Data → Model Evaluation → API Deployment (FastAPI) → UI (Streamlit)**.

---

## 📁 Project Structure

```
Machine Learning Project/
│
├── .venv/                          # Virtual environment
│
├── Credit_Card_Fraud_detection/
│   ├── credit_card_fraud_detection.ipynb   # EDA + Model training notebook
│   ├── creditcard.csv                      # Dataset
│   ├── FastApi.py                          # FastAPI inference service
│   ├── Streamlit_App.py                    # Streamlit web app
│   ├── fraud_model.pkl                     # Trained ML model
│   ├── scaler.pkl                          # Feature scaler
│   ├── requirements.txt                    # Project dependencies
```

---

## 🚀 What I Did in This Project

### 1️⃣ Problem Understanding

* Goal: **Identify fraudulent credit card transactions**
* Challenge: **Highly imbalanced dataset** (very few fraud cases)
* Focused on Recall and ROC-AUC, not accuracy
* Designed as a cost-sensitive classification problem
---

### 2️⃣ Exploratory Data Analysis (EDA)

* Checked class imbalance between **fraud vs non‑fraud**
* Analyzed feature distributions
* Identified that `Amount` needs scaling
* Verified no missing values

---

### 3️⃣ Data Preprocessing

* Removed duplicated values
* Dropped unnecessary features (if any)
* Performed **train‑test split BEFORE SMOTE** (best practice)
* Applied **StandardScaler** on `Amount`
* Saved the fitted scaler as `scaler.pkl`

---

### 4️⃣ Handling Imbalanced Data

* Used **SMOTE (Synthetic Minority Over‑sampling Technique)** only on **training data**
* Ensured test data remains untouched

---

### 5️⃣ Model Training

* Trained a **classification model** (Logistic Regression / Tree‑based)
* Stored feature order using `model.feature_names_in_`
* Tuned threshold to prioritize **high recall** (fraud detection)

---

### 6️⃣ Model Evaluation

* Evaluated using:

  * Accuracy
  * Precision
  * Recall
  * F1‑Score
  * ROC‑AUC Curve

👉 Focused on **Recall & ROC‑AUC**, not accuracy (because fraud detection is cost‑sensitive)

---

### 7️⃣ Model Serialization

* Saved trained artifacts using `joblib`:

  * `fraud_model.pkl`
  * `scaler.pkl`

---

### 8️⃣ FastAPI Deployment

* Built a REST API using **FastAPI**
* Defined input schema using **Pydantic**
* Implemented `/predict` endpoint
* Ensured **exact feature order matching training**
* Returns:

  * Fraud probability
  * Final prediction (FRAUD / LEGIT)

Run API:

```bash
uvicorn FastApi:app --reload
```
Swagger UI:
http://127.0.0.1:8000/docs
---

### 9️⃣ Streamlit Web App

* Built a user‑friendly UI using **Streamlit**
* Sidebar input for all transaction features
* Default values provided for easy testing
* Displays:

  * Prediction
  * Fraud probability

Run app:

```bash
streamlit run Streamlit_App.py
```
Streamlit Inference Flow
```
User Input
   ↓
Load scaler.pkl
   ↓
Load fraud_model.pkl
   ↓
Scale Amount
   ↓
Align feature order
   ↓
Predict probability
   ↓
Apply threshold
   ↓
Display result
```
---

## 🧠 Key ML Best Practices Followed

✔ Train‑test split before SMOTE
✔ Feature scaling only on training data
✔ Saved preprocessing objects
✔ Threshold tuning for recall
✔ End‑to‑end deployment mindset
✔ Production‑ready folder structure

---

## 🛠 Tech Stack

* Python
* Pandas, NumPy
* Scikit‑learn
* SMOTE (imbalanced‑learn)
* FastAPI
* Streamlit
* Joblib

---

## 📌 Use Case

This project is suitable for:

* Data Scientist / ML Engineer portfolios
* Interview discussion (end‑to‑end ML system)
* Real‑world fraud detection systems

---

## ✨ Author

**Rajkumar**
Aspiring Data Scientist | ML Engineer

---

If you want:

* Dockerization 🐳
* Cloud deployment (AWS / Azure)
* Model monitoring
* CI/CD pipeline

👉 Just ask 😄
