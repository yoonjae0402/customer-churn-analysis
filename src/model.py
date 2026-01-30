"""
Model training and prediction utilities.

This module provides functions for training, saving, loading,
and using machine learning models for churn prediction.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load_model(model_path: str | Path) -> Dict[str, Any]:
    """
    Load a trained model and scaler from disk.

    Args:
        model_path: Path to the saved model file (.pkl)

    Returns:
        Dictionary containing 'model' and 'scaler' objects

    Raises:
        FileNotFoundError: If model file doesn't exist
        KeyError: If required keys are missing from saved object
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    pipeline = joblib.load(path)

    if "model" not in pipeline:
        raise KeyError("Model key not found in saved pipeline")
    if "scaler" not in pipeline:
        raise KeyError("Scaler key not found in saved pipeline")

    return pipeline


def save_model(
    model: Any,
    scaler: StandardScaler,
    model_path: str | Path,
) -> None:
    """
    Save a trained model and scaler to disk.

    Args:
        model: Trained model object
        scaler: Fitted StandardScaler object
        model_path: Path to save the model file
    """
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = {"model": model, "scaler": scaler}
    joblib.dump(pipeline, path)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "logistic_regression",
    scale_features: bool = True,
    **kwargs,
) -> Tuple[Any, Optional[StandardScaler]]:
    """
    Train a churn prediction model.

    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model ('logistic_regression', 'random_forest', etc.)
        scale_features: Whether to scale numerical features
        **kwargs: Additional arguments for the model

    Returns:
        Tuple of (trained_model, scaler or None)
    """
    scaler = None

    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    if model_type == "logistic_regression":
        model = LogisticRegression(
            max_iter=kwargs.get("max_iter", 1000),
            random_state=kwargs.get("random_state", 42),
        )
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", None),
            random_state=kwargs.get("random_state", 42),
        )
    elif model_type == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            learning_rate=kwargs.get("learning_rate", 0.1),
            max_depth=kwargs.get("max_depth", 3),
            random_state=kwargs.get("random_state", 42),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)

    return model, scaler


def predict_churn(
    model: Any,
    X: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    return_proba: bool = True,
) -> np.ndarray:
    """
    Predict churn for customer data.

    Args:
        model: Trained model
        X: Customer features
        scaler: Optional scaler to transform features
        return_proba: If True, return probabilities; else return class labels

    Returns:
        Array of predictions (probabilities or class labels)
    """
    if scaler is not None:
        X = scaler.transform(X)

    if return_proba:
        return model.predict_proba(X)[:, 1]
    else:
        return model.predict(X)


def get_risk_level(churn_probability: float) -> str:
    """
    Convert churn probability to a risk level category.

    Args:
        churn_probability: Predicted churn probability (0-1)

    Returns:
        Risk level string: 'LOW', 'MEDIUM', or 'HIGH'
    """
    if churn_probability < 0.3:
        return "LOW"
    elif churn_probability < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"
