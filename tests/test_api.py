
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "degraded (model not loaded)"]

def test_predict_endpoint_validation():
    # Missing fields
    response = client.post("/predict", json={})
    assert response.status_code == 422

def test_predict_valid_payload():
    # Note: If model is not loaded (e.g. CI environment without trained model),
    # this might return 503. Handled gracefully.
    payload = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.9,
        "TotalCharges": 1078.8
    }
    response = client.post("/predict", json=payload)
    
    if response.status_code == 503:
        assert response.json()["detail"] == "Model not loaded. Please train the model first."
    else:
        assert response.status_code == 200
        data = response.json()
        assert "churn_probability" in data
        assert "marketing_offer" in data
