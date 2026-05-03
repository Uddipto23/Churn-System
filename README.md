# Churn-System — Intelligent Customer Churn Prediction

This project predicts whether a customer is likely to **churn** (leave/cancel a service) using supervised machine learning.  
It includes an end-to-end pipeline: model training, evaluation, explainability, a REST API, and an interactive dashboard.

## What is Customer Churn?
**Churn** means a customer stops using a service (e.g., cancels a subscription).  
Predicting churn helps businesses take proactive actions (discounts, support, retention campaigns).

## Key Features
- Data preprocessing (missing value handling, scaling, one-hot encoding)
- Multiple model training + comparison
  - Logistic Regression
  - Random Forest
  - XGBoost (best model)
  - LightGBM
- Hyperparameter tuning (cross-validation)
- Threshold selection for classification
- Explainability with SHAP
- **FastAPI** REST API (`/predict`)
- **Streamlit** dashboard (demo UI)
- Dockerized deployment (API + Dashboard)
- GitHub Actions CI (tests + docker builds)

## Model Summary (from training)
- Best model: **XGBoost**
- Test ROC-AUC: ~0.86 (varies by split/tuning)
- Tuned threshold stored in `models/threshold.json`

## Repository Structure
```
app/
  api.py              # FastAPI inference service
  streamlit_app.py    # Streamlit dashboard (calls the API)
models/
  model.joblib
  threshold.json
  input_schema.json
tests/
  test_api.py
Dockerfile.api
Dockerfile.streamlit
docker-compose.yml
requirements.txt
```

## Run with Docker (Recommended)
### 1) Build and start
```bash
docker compose up --build
```

### 2) Open
- API Swagger docs: http://localhost:8000/docs
- Streamlit Dashboard: http://localhost:8501

## API Usage
### Health
`GET /health`

### Predict
`POST /predict`

Request format:
```json
{
  "data": {
    "...": "..."
  }
}
```

Response format:
```json
{
  "churn_probability": 0.433,
  "churn_prediction": 1,
  "threshold_used": 0.349
}
```

## Notes / Limitations
- The model artifact is stored in `models/model.joblib`.
- Keep `scikit-learn` pinned (see `requirements.txt`) to avoid joblib/pickle version issues.
- This is a demo/portfolio project; for production, add authentication, monitoring, and stricter input validation.

## Author
GitHub: @Uddipto23
