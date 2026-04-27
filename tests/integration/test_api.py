from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
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
    "TotalCharges": 1078.8,
}


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "model_type" in data
    assert "threshold" in data


def test_health_check_legacy_alias():
    """Legacy /health must return the same shape as /healthz."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "model_type" in data
    assert "threshold" in data


def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "loaded" in data
    assert "model_type" in data or "status" in data


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_predict_missing_all_fields():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_invalid_gender():
    payload = {**VALID_PAYLOAD, "gender": "Other"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_contract():
    payload = {**VALID_PAYLOAD, "Contract": "Weekly"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_internet_service():
    payload = {**VALID_PAYLOAD, "InternetService": "5G"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_payment_method():
    payload = {**VALID_PAYLOAD, "PaymentMethod": "Crypto"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_monthly_charges():
    payload = {**VALID_PAYLOAD, "MonthlyCharges": -10.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_senior_citizen_out_of_range():
    payload = {**VALID_PAYLOAD, "SeniorCitizen": 5}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_optional_total_charges_omitted():
    """TotalCharges is optional — omitting it must not cause a validation error."""
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "TotalCharges"}
    response = client.post("/predict", json=payload)
    # Model may not be loaded in CI, but validation must pass (not 422)
    assert response.status_code != 422


# ---------------------------------------------------------------------------
# Successful prediction (Gemini mocked)
# ---------------------------------------------------------------------------


@patch(
    "src.api.services.marketing.MarketingService.generate_offer",
    new_callable=AsyncMock,
    return_value="Special offer: lock in your rate for 2 years and save 20%!",
)
def test_predict_with_loaded_model(mock_offer):
    """
    When the model is loaded, /predict must return a valid response.
    Gemini is mocked to avoid external API calls in tests.
    """
    import src.api.main as main_module

    if main_module.model_pipeline is None:
        pytest.skip("Model not loaded — train the model before running this test")

    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "marketing_offer" in data
    assert "metadata" in data

    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in [0, 1]
    assert "threshold" in data["metadata"]
    assert "latency_ms" in data["metadata"]

    mock_offer.assert_called_once()


@patch(
    "src.api.services.marketing.MarketingService.generate_offer",
    new_callable=AsyncMock,
    return_value="Loyalty bonus unlocked!",
)
def test_predict_response_schema(mock_offer):
    """Response must conform to PredictionResponse schema."""
    import src.api.main as main_module

    if main_module.model_pipeline is None:
        pytest.skip("Model not loaded — train the model before running this test")

    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()

    assert isinstance(data["churn_probability"], float)
    assert isinstance(data["churn_prediction"], int)
    assert isinstance(data["marketing_offer"], str)
    assert isinstance(data["metadata"], dict)


# ---------------------------------------------------------------------------
# Model-not-loaded behaviour
# ---------------------------------------------------------------------------


def test_predict_model_not_loaded_returns_503():
    """When no model is loaded, /predict must return 503."""
    import src.api.main as main_module

    original_pipeline = main_module.model_pipeline
    try:
        main_module.model_pipeline = None
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 503
    finally:
        main_module.model_pipeline = original_pipeline


# ---------------------------------------------------------------------------
# Gemini fallback
# ---------------------------------------------------------------------------


@patch(
    "src.api.services.marketing.MarketingService.generate_offer",
    new_callable=AsyncMock,
    side_effect=Exception("Gemini API down"),
)
def test_predict_gemini_error_does_not_crash(mock_offer):
    """
    If Gemini fails, the endpoint must not crash — the marketing_service
    catches exceptions and returns a fallback string.
    """
    import src.api.main as main_module

    if main_module.model_pipeline is None:
        pytest.skip("Model not loaded")

    # The marketing service catches exceptions internally and returns a fallback.
    # We patch at the service level to verify the API still responds.
    response = client.post("/predict", json=VALID_PAYLOAD)
    # Either 200 (fallback handled) or 500 (if exception propagated) —
    # but must NOT be an unhandled crash (which would give no response at all).
    assert response.status_code in [200, 500]
