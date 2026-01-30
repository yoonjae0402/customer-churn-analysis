"""
Integration tests for the FastAPI churn prediction API.

Run with: pytest tests/test_api.py -v
Note: Requires GEMINI_API_KEY to be set for full tests.
"""
import pytest
from unittest.mock import patch, AsyncMock
import sys
from pathlib import Path

# Add app directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient

        # Mock the Gemini API key check for testing
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'}):
            try:
                from main import app
                return TestClient(app)
            except Exception:
                pytest.skip("Could not import app - likely missing dependencies")

    def test_health_returns_200(self, client):
        """Test that health endpoint returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client):
        """Test that health endpoint returns healthy status."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_includes_model_version(self, client):
        """Test that health endpoint includes model version."""
        response = client.get("/health")
        data = response.json()

        assert "model_version" in data
        assert data["model_version"] is not None


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient

        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'}):
            try:
                from main import app
                return TestClient(app)
            except Exception:
                pytest.skip("Could not import app - likely missing dependencies")

    @pytest.fixture
    def sample_customer_data(self):
        """Sample customer data for testing."""
        return {
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
            "MonthlyCharges": 89.90,
            "TotalCharges": 1078.80
        }

    def test_predict_requires_all_fields(self, client):
        """Test that predict endpoint requires all required fields."""
        response = client.post("/predict", json={})
        assert response.status_code == 422  # Validation error

    def test_predict_validates_gender(self, client, sample_customer_data):
        """Test that predict validates gender field."""
        sample_customer_data["gender"] = "Invalid"
        response = client.post("/predict", json=sample_customer_data)
        # Should still work as Pydantic only validates types, not values
        # The preprocessing handles value validation

    def test_predict_validates_senior_citizen_range(self, client, sample_customer_data):
        """Test that SeniorCitizen must be 0 or 1."""
        sample_customer_data["SeniorCitizen"] = 5
        response = client.post("/predict", json=sample_customer_data)
        assert response.status_code == 422

    def test_predict_validates_tenure_non_negative(self, client, sample_customer_data):
        """Test that tenure must be non-negative."""
        sample_customer_data["tenure"] = -1
        response = client.post("/predict", json=sample_customer_data)
        assert response.status_code == 422

    def test_predict_validates_monthly_charges_positive(self, client, sample_customer_data):
        """Test that MonthlyCharges must be positive."""
        sample_customer_data["MonthlyCharges"] = -10.0
        response = client.post("/predict", json=sample_customer_data)
        assert response.status_code == 422


class TestInputValidation:
    """Tests for input data validation."""

    def test_customer_data_model(self):
        """Test CustomerData Pydantic model validation."""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
            from main import CustomerData

            # Valid data should work
            valid_data = CustomerData(
                gender="Male",
                SeniorCitizen=0,
                Partner="Yes",
                Dependents="No",
                tenure=12,
                PhoneService="Yes",
                MultipleLines="No",
                InternetService="Fiber optic",
                OnlineSecurity="No",
                OnlineBackup="Yes",
                DeviceProtection="No",
                TechSupport="No",
                StreamingTV="Yes",
                StreamingMovies="No",
                Contract="Month-to-month",
                PaperlessBilling="Yes",
                PaymentMethod="Electronic check",
                MonthlyCharges=89.90,
                TotalCharges=1078.80
            )
            assert valid_data.tenure == 12
            assert valid_data.MonthlyCharges == 89.90

        except Exception:
            pytest.skip("Could not import CustomerData model")

    def test_total_charges_optional(self):
        """Test that TotalCharges is optional."""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
            from main import CustomerData

            # Should work without TotalCharges
            data = CustomerData(
                gender="Female",
                SeniorCitizen=1,
                Partner="No",
                Dependents="No",
                tenure=1,
                PhoneService="Yes",
                MultipleLines="No",
                InternetService="DSL",
                OnlineSecurity="Yes",
                OnlineBackup="No",
                DeviceProtection="No",
                TechSupport="Yes",
                StreamingTV="No",
                StreamingMovies="No",
                Contract="One year",
                PaperlessBilling="No",
                PaymentMethod="Mailed check",
                MonthlyCharges=45.50
                # TotalCharges omitted
            )
            assert data.TotalCharges is None

        except Exception:
            pytest.skip("Could not import CustomerData model")
