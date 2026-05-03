from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_missing_cols_returns_400():
    r = client.post("/predict", json={"data": {"x": 1}})
    assert r.status_code == 400