import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("Customer Churn Prediction (Demo)")

model = joblib.load("model.joblib")
threshold = float(json.load(open("threshold.json"))["threshold"])
schema = json.load(open("input_schema.json"))["columns"]

# Helper: pick the first column containing a keyword
def pick(keyword):
    for c in schema:
        if keyword in c.lower():
            return c
    return None

# Try to auto-detect your column names
TENURE   = pick("tenure")
MONTHLY  = pick("monthly")
TOTAL    = pick("total charges") or pick("total")
CONTRACT = pick("contract")
PAYMENT  = pick("payment")
INTERNET = pick("internet")

# These will be required if found
required = [c for c in [TENURE, MONTHLY, CONTRACT] if c is not None]

st.caption(f"Threshold: {threshold:.3f}")
st.write("Required fields:", required)

# Initialize row with missing values for all columns
row = {c: None for c in schema}

# Build demo form
with st.form("demo"):
    if TENURE:
        row[TENURE] = st.number_input(TENURE, min_value=0.0, max_value=200.0, value=12.0, step=1.0)
    if MONTHLY:
        row[MONTHLY] = st.number_input(MONTHLY, min_value=0.0, max_value=500.0, value=70.0, step=1.0)
    if TOTAL:
        row[TOTAL] = st.number_input(TOTAL, min_value=0.0, max_value=100000.0, value=1000.0, step=10.0)

    if CONTRACT:
        row[CONTRACT] = st.selectbox(CONTRACT, options=["Month-to-month", "One year", "Two year"])

    if INTERNET:
        row[INTERNET] = st.selectbox(INTERNET, options=["Yes", "No"])

    if PAYMENT:
        row[PAYMENT] = st.text_input(PAYMENT, "")

    submitted = st.form_submit_button("Predict")

if submitted:
    # Validate required
    missing = [c for c in required if row.get(c) is None or row.get(c) == ""]
    if missing:
        st.error(f"Please fill: {missing}")
        st.stop()

    df = pd.DataFrame([row], columns=schema)
    prob = float(model.predict_proba(df)[:, 1][0])
    pred = int(prob >= threshold)

    st.metric("Churn probability", f"{prob:.3f}")
    st.write("Prediction:", "CHURN" if pred else "NO CHURN")
