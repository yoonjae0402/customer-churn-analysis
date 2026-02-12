
import pytest
import pandas as pd
import numpy as np
from src.model import create_model_pipeline, save_model, load_model, predict_churn

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "gender": ["Male", "Female"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "tenure": [12, 48],
        "PhoneService": ["Yes", "No"],
        "MultipleLines": ["No", "No phone service"],
        "InternetService": ["Fiber optic", "DSL"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["Yes", "No"],
        "StreamingMovies": ["No", "Yes"],
        "Contract": ["Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Bank transfer (automatic)"],
        "MonthlyCharges": [89.9, 45.0],
        "TotalCharges": [1078.8, 2160.0],
        # Target for training
        "Churn": [1, 0] 
    })

def test_pipeline_creation():
    pipeline = create_model_pipeline(model_type="random_forest")
    assert pipeline is not None
    assert len(pipeline.steps) == 4 # cleaner, engineer, preprocessor, classifier

def test_training_flow(sample_data, tmp_path):
    # 1. Pipeline
    pipeline = create_model_pipeline(model_type="logistic_regression")
    
    # 2. Fit
    X = sample_data.drop("Churn", axis=1)
    y = sample_data["Churn"]
    pipeline.fit(X, y)
    
    # 3. Predict
    preds = predict_churn(pipeline, X)
    assert len(preds) == 2
    assert isinstance(preds[0], float) # Probabilities
    
    # 4. Save/Load
    model_path = tmp_path / "test_model.pkl"
    save_model(pipeline, model_path)
    assert model_path.exists()
    
    loaded = load_model(model_path)
    assert loaded is not None
