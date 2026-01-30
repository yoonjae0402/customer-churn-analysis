"""
Unit tests for the churn prediction model.

Run with: pytest tests/ -v
"""
import pytest
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Test paths
MODELS_DIR = Path(__file__).parent.parent / "app" / "models"
DATA_DIR = Path(__file__).parent.parent / "app" / "data" / "processed"


class TestModelLoading:
    """Tests for model loading functionality."""

    def test_churn_model_exists(self):
        """Test that the production model file exists."""
        model_path = MODELS_DIR / "churn_model.pkl"
        assert model_path.exists(), f"Model file not found at {model_path}"

    def test_churn_model_loads_correctly(self):
        """Test that the model loads and has expected components."""
        model_path = MODELS_DIR / "churn_model.pkl"
        pipeline = joblib.load(model_path)

        assert "scaler" in pipeline, "Pipeline missing 'scaler' component"
        assert "model" in pipeline, "Pipeline missing 'model' component"

    def test_scaler_exists(self):
        """Test that the scaler file exists."""
        scaler_path = MODELS_DIR / "scaler.pkl"
        assert scaler_path.exists(), f"Scaler not found at {scaler_path}"

    def test_best_model_exists(self):
        """Test that the best model file exists."""
        model_path = MODELS_DIR / "best_model.pkl"
        assert model_path.exists(), f"Best model not found at {model_path}"


class TestFeatureNames:
    """Tests for feature name loading."""

    def test_feature_names_file_exists(self):
        """Test that feature names file exists."""
        feature_path = DATA_DIR / "feature_names.txt"
        assert feature_path.exists(), f"Feature names file not found at {feature_path}"

    def test_feature_names_count(self):
        """Test that we have the expected number of features (62)."""
        feature_path = DATA_DIR / "feature_names.txt"
        with open(feature_path, 'r') as f:
            lines = f.readlines()

        # Count non-header, non-empty lines
        feature_count = sum(
            1 for line in lines
            if line.strip()
            and not line.strip().startswith('==')
            and not line.strip().startswith('Feature')
        )
        assert feature_count == 62, f"Expected 62 features, got {feature_count}"


class TestModelPrediction:
    """Tests for model prediction functionality."""

    @pytest.fixture
    def loaded_model(self):
        """Fixture to load the model for tests."""
        pipeline = joblib.load(MODELS_DIR / "churn_model.pkl")
        return pipeline["model"], pipeline["scaler"]

    @pytest.fixture
    def sample_features(self):
        """Fixture providing sample feature data."""
        # Create a sample with 62 features (all zeros as baseline)
        return np.zeros((1, 62))

    def test_prediction_returns_array(self, loaded_model, sample_features):
        """Test that prediction returns an array."""
        model, scaler = loaded_model
        prediction = model.predict(sample_features)

        assert isinstance(prediction, np.ndarray), "Prediction should be numpy array"
        assert len(prediction) == 1, "Should return one prediction"

    def test_prediction_binary_output(self, loaded_model, sample_features):
        """Test that prediction is binary (0 or 1)."""
        model, scaler = loaded_model
        prediction = model.predict(sample_features)

        assert prediction[0] in [0, 1], f"Prediction should be 0 or 1, got {prediction[0]}"

    def test_probability_in_valid_range(self, loaded_model, sample_features):
        """Test that probability is between 0 and 1."""
        model, scaler = loaded_model
        proba = model.predict_proba(sample_features)

        assert proba.shape == (1, 2), "Should return probabilities for 2 classes"
        assert 0 <= proba[0, 0] <= 1, "Probability should be between 0 and 1"
        assert 0 <= proba[0, 1] <= 1, "Probability should be between 0 and 1"
        assert abs(proba[0, 0] + proba[0, 1] - 1.0) < 1e-6, "Probabilities should sum to 1"


class TestDataFiles:
    """Tests for data file integrity."""

    def test_raw_data_exists(self):
        """Test that raw data file exists."""
        raw_data_path = Path(__file__).parent.parent / "app" / "data" / "raw" / "telco_churn.csv"
        assert raw_data_path.exists(), f"Raw data not found at {raw_data_path}"

    def test_raw_data_columns(self):
        """Test that raw data has expected columns."""
        raw_data_path = Path(__file__).parent.parent / "app" / "data" / "raw" / "telco_churn.csv"
        df = pd.read_csv(raw_data_path)

        expected_columns = [
            'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
        ]

        for col in expected_columns:
            assert col in df.columns, f"Missing expected column: {col}"

    def test_raw_data_row_count(self):
        """Test that raw data has expected number of rows."""
        raw_data_path = Path(__file__).parent.parent / "app" / "data" / "raw" / "telco_churn.csv"
        df = pd.read_csv(raw_data_path)

        assert len(df) == 7043, f"Expected 7043 rows, got {len(df)}"

    def test_processed_data_exists(self):
        """Test that processed data file exists."""
        processed_path = DATA_DIR / "data_processed_final.csv"
        assert processed_path.exists(), f"Processed data not found at {processed_path}"


class TestModelComparison:
    """Tests for model comparison results."""

    def test_model_comparison_exists(self):
        """Test that model comparison CSV exists."""
        comparison_path = MODELS_DIR / "model_comparison.csv"
        assert comparison_path.exists(), f"Model comparison not found at {comparison_path}"

    def test_model_comparison_columns(self):
        """Test that model comparison has expected columns."""
        comparison_path = MODELS_DIR / "model_comparison.csv"
        df = pd.read_csv(comparison_path)

        expected_columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        for col in expected_columns:
            assert col in df.columns, f"Missing column in model comparison: {col}"
