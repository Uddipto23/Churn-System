from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib, json

app = FastAPI(title="Churn Prediction API", version="1.0")

model = joblib.load("model.joblib")
threshold = json.load(open("threshold.json"))["threshold"]
schema = json.load(open("input_schema.json"))["columns"]

class PredictRequest(BaseModel):
    data: dict  # must contain all schema columns

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    row = req.data

    missing = [c for c in schema if c not in row]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing[:25]}{'...' if len(missing)>25 else ''}"
        )

    df = pd.DataFrame([row], columns=schema)
    prob = float(model.predict_proba(df)[:, 1][0])
    pred = int(prob >= threshold)

    return {
        "churn_probability": prob,
        "churn_prediction": pred,
        "threshold_used": float(threshold)
    }
