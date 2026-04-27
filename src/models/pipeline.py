import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import xgboost as xgb
except ImportError:
    xgb = None
except Exception:  # Handle XGBoostError (libomp missing)
    xgb = None

from src.config import config
from src.features.engineering import CategoricalCleaner, FeatureEngineer


def get_feature_lists() -> Dict[str, List[str]]:
    """Returns numerical and categorical feature lists expected after engineering."""
    base_num = config.feature_config["numerical"]

    derived_num = [
        "is_new_customer",
        "is_long_term",
        "tenure_bins",
        "avg_monthly_spend",
        "charge_per_tenure",
        "price_increase",
        "high_monthly_charge",
        "total_services",
        "has_internet",
        "has_phone",
        "has_tech_support",
        "has_online_security",
        "is_month_to_month",
        "has_auto_payment",
        "is_paperless",
        "is_senior",
        "has_family",
        "is_single_no_deps",
        "gender",  # Mapped to 0/1
    ]

    categorical_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]

    numerical_cols = base_num + derived_num

    return {
        "numerical": list(set(numerical_cols)),
        "categorical": categorical_cols,
    }


def create_preprocessor() -> ColumnTransformer:
    """Creates the sklearn ColumnTransformer for scaling and encoding."""
    feature_lists = get_feature_lists()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_lists["numerical"]),
            ("cat", categorical_transformer, feature_lists["categorical"]),
        ],
        remainder="drop",
    )

    return preprocessor


def create_model_pipeline(
    model_type: str = "random_forest", params: Optional[Dict] = None
) -> Pipeline:
    """Create the full pipeline: FeatureEngineer -> Preprocessor -> Classifier."""
    if params is None:
        params = {}

    if model_type == "logistic_regression":
        clf = LogisticRegression(**params)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(**params)
    elif model_type == "xgboost":
        if xgb is None:
            raise ImportError(
                "XGBoost is not installed or dependencies (libomp) are missing."
            )
        clf = xgb.XGBClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    pipeline = Pipeline(
        steps=[
            ("cleaner", CategoricalCleaner()),
            ("engineer", FeatureEngineer()),
            ("preprocessor", create_preprocessor()),
            ("classifier", clf),
        ]
    )

    return pipeline


def save_model(pipeline: Any, model_path: str) -> None:
    """Save the full pipeline to disk."""
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(model_path: str) -> Any:
    """Load the full pipeline from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def predict_churn(pipeline: Any, X: pd.DataFrame) -> np.ndarray:
    """Predict churn probabilities using the pipeline."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    else:
        return pipeline.predict(X)


def predict_churn_with_threshold(
    pipeline: Any, X: pd.DataFrame, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict churn probabilities and binary labels using a custom threshold.

    Using a threshold tuned to maximise F1 (rather than the default 0.5)
    significantly improves recall on the minority churn class.

    Args:
        pipeline: Fitted model pipeline
        X: Raw feature DataFrame
        threshold: Classification threshold (from find_optimal_threshold)

    Returns:
        Tuple of (probabilities, binary_predictions)
    """
    probabilities = predict_churn(pipeline, X)
    binary_predictions = (probabilities >= threshold).astype(int)
    return probabilities, binary_predictions


def load_threshold(model_type: str, models_dir: str = "models") -> float:
    """
    Load the saved optimal threshold for a model type.

    Falls back to 0.5 if the threshold file does not exist.

    Args:
        model_type: Model type string (e.g. "random_forest")
        models_dir: Directory where model artifacts are stored

    Returns:
        Optimal threshold float
    """
    threshold_path = Path(models_dir) / f"{model_type}_threshold.json"
    if threshold_path.exists():
        data = json.loads(threshold_path.read_text())
        return float(data["threshold"])
    return 0.5
