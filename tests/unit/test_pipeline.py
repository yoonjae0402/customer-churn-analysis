import numpy as np
import pandas as pd
import pytest

from src.models.evaluation import find_optimal_threshold
from src.models.pipeline import (
    create_model_pipeline,
    load_model,
    load_threshold,
    predict_churn,
    predict_churn_with_threshold,
    save_model,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_data():
    """Minimal two-row dataset covering all required columns."""
    return pd.DataFrame(
        {
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
            "Churn": [1, 0],
        }
    )


@pytest.fixture
def imbalanced_data():
    """
    50-row dataset with ~26% churn rate, mimicking real class imbalance.
    Tests that class-weighted models train without errors.
    """
    rng = np.random.default_rng(42)
    n = 50
    churn = np.array([1] * 13 + [0] * 37)  # ~26% churn

    return pd.DataFrame(
        {
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "tenure": rng.integers(0, 72, n),
            "PhoneService": rng.choice(["Yes", "No"], n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": rng.uniform(20, 120, n).round(2),
            "TotalCharges": rng.uniform(0, 8000, n).round(2),
            "Churn": churn,
        }
    )


# ---------------------------------------------------------------------------
# Pipeline structure
# ---------------------------------------------------------------------------


def test_pipeline_creation():
    pipeline = create_model_pipeline(model_type="random_forest")
    assert pipeline is not None
    assert len(pipeline.steps) == 4  # cleaner, engineer, preprocessor, classifier


@pytest.mark.parametrize("model_type", ["logistic_regression", "random_forest"])
def test_pipeline_creation_all_types(model_type):
    pipeline = create_model_pipeline(model_type=model_type)
    assert pipeline is not None
    step_names = [name for name, _ in pipeline.steps]
    assert "cleaner" in step_names
    assert "engineer" in step_names
    assert "preprocessor" in step_names
    assert "classifier" in step_names


def test_invalid_model_type_raises():
    with pytest.raises(ValueError, match="Unsupported model type"):
        create_model_pipeline(model_type="unknown_model")


# ---------------------------------------------------------------------------
# Training flow
# ---------------------------------------------------------------------------


def test_training_flow(sample_data, tmp_path):
    pipeline = create_model_pipeline(model_type="logistic_regression")

    X = sample_data.drop("Churn", axis=1)
    y = sample_data["Churn"]
    pipeline.fit(X, y)

    preds = predict_churn(pipeline, X)
    assert len(preds) == 2
    assert isinstance(float(preds[0]), float)
    assert 0.0 <= preds[0] <= 1.0

    model_path = tmp_path / "test_model.pkl"
    save_model(pipeline, model_path)
    assert model_path.exists()

    loaded = load_model(model_path)
    assert loaded is not None

    loaded_preds = predict_churn(loaded, X)
    np.testing.assert_array_almost_equal(preds, loaded_preds)


def test_class_imbalance_training(imbalanced_data):
    """Weighted models must train and produce valid probabilities on imbalanced data."""
    X = imbalanced_data.drop("Churn", axis=1)
    y = imbalanced_data["Churn"]

    model_cases = [
        (
            "logistic_regression",
            {"max_iter": 200, "C": 1.0, "class_weight": "balanced"},
        ),  # noqa: E501
        ("random_forest", {"n_estimators": 10, "class_weight": "balanced"}),
    ]
    for model_type, params in model_cases:
        pipeline = create_model_pipeline(model_type, params)
        pipeline.fit(X, y)
        probs = predict_churn(pipeline, X)

        assert probs.shape == (len(X),), f"{model_type}: wrong output shape"
        assert np.all(
            (probs >= 0) & (probs <= 1)
        ), f"{model_type}: probabilities out of range"


# ---------------------------------------------------------------------------
# Threshold-aware prediction
# ---------------------------------------------------------------------------


def test_predict_with_threshold(sample_data):
    pipeline = create_model_pipeline(model_type="logistic_regression")
    X = sample_data.drop("Churn", axis=1)
    y = sample_data["Churn"]
    pipeline.fit(X, y)

    probs, preds = predict_churn_with_threshold(pipeline, X, threshold=0.5)
    assert probs.shape == preds.shape
    assert set(preds).issubset({0, 1})

    # Lowering threshold should predict more positives (or equal)
    _, preds_low = predict_churn_with_threshold(pipeline, X, threshold=0.1)
    assert preds_low.sum() >= preds.sum()

    # Raising threshold should predict fewer positives (or equal)
    _, preds_high = predict_churn_with_threshold(pipeline, X, threshold=0.9)
    assert preds_high.sum() <= preds.sum()


def test_load_threshold_fallback(tmp_path):
    """load_threshold returns 0.5 when no threshold file exists."""
    threshold = load_threshold("random_forest", models_dir=str(tmp_path))
    assert threshold == 0.5


def test_load_threshold_from_file(tmp_path):
    """load_threshold correctly reads saved threshold."""
    import json

    threshold_file = tmp_path / "xgboost_threshold.json"
    threshold_file.write_text(json.dumps({"threshold": 0.38, "strategy": "f1"}))

    threshold = load_threshold("xgboost", models_dir=str(tmp_path))
    assert abs(threshold - 0.38) < 1e-6


# ---------------------------------------------------------------------------
# Optimal threshold finding
# ---------------------------------------------------------------------------


def test_find_optimal_threshold_f1():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 100)
    y_proba = rng.uniform(0, 1, 100)

    threshold, metrics = find_optimal_threshold(y_true, y_proba, strategy="f1")
    assert 0.0 <= threshold <= 1.0
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert metrics["f1"] >= 0.0


def test_find_optimal_threshold_cost():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 100)
    y_proba = rng.uniform(0, 1, 100)

    threshold, metrics = find_optimal_threshold(
        y_true, y_proba, strategy="cost", cost_fn=10.0, cost_fp=1.0
    )
    assert 0.0 <= threshold <= 1.0


def test_find_optimal_threshold_invalid_strategy():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.4, 0.7])
    with pytest.raises(ValueError, match="Unknown strategy"):
        find_optimal_threshold(y_true, y_proba, strategy="bad_strategy")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_predict_single_row(sample_data):
    """Pipeline must handle a single-row DataFrame (inference scenario)."""
    pipeline = create_model_pipeline(model_type="logistic_regression")
    X = sample_data.drop("Churn", axis=1)
    y = sample_data["Churn"]
    pipeline.fit(X, y)

    single_row = X.iloc[[0]]
    probs = predict_churn(pipeline, single_row)
    assert len(probs) == 1
    assert 0.0 <= float(probs[0]) <= 1.0


def test_missing_total_charges_handled(sample_data):
    """Pipeline must handle NaN TotalCharges gracefully (new customers)."""
    pipeline = create_model_pipeline(model_type="logistic_regression")
    X = sample_data.drop("Churn", axis=1).copy()
    y = sample_data["Churn"]

    X.loc[0, "TotalCharges"] = float("nan")
    pipeline.fit(X, y)

    probs = predict_churn(pipeline, X)
    assert not np.isnan(
        probs
    ).any(), "NaN TotalCharges should not produce NaN probabilities"
