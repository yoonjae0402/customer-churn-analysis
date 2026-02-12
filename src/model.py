
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
try:
    import xgboost as xgb
except ImportError:
    xgb = None
except Exception: # Handle XGBoostError (libomp missing)
    xgb = None

from typing import Any, Dict, List, Optional

from .features import FeatureEngineer, CategoricalCleaner
from .config import config

def get_feature_lists() -> Dict[str, List[str]]:
    """
    Returns the lists of numerical and categorical features 
    expected AFTER feature engineering.
    """
    # Base features from config
    base_num = config.feature_config["numerical"]
    base_cat = config.feature_config["categorical"]
    
    # Derived numerical/boolean features (treated as numeric for scaling or passthrough)
    derived_num = [
        "is_new_customer", "is_long_term", "tenure_bins",
        "avg_monthly_spend", "charge_per_tenure", "price_increase", "high_monthly_charge",
        "total_services", "has_internet", "has_phone", "has_tech_support", "has_online_security",
        "is_month_to_month", "has_auto_payment", "is_paperless",
        "is_senior", "has_family", "is_single_no_deps",
        "gender" # Mapped to 0/1
    ]
    
    # Derived categorical features (none currently, tenure_group is handled via binning or dropped)
    # The feature engineer adds binary flags which are numeric.
    
    # The CategoricalCleaner maps some binary binaries (Partner, Dependents) to 0/1, so they become numeric
    # Features that remain truly categorical for OneHotEncoding:
    # MultipleLines, InternetService, OnlineSecurity... PaymentMethod, Contract
    
    # Let's align with what FeatureEngineer produces and CategoricalCleaner cleans.
    # CategoricalCleaner maps: Partner, Dependents, PhoneService, PaperlessBilling, gender -> numeric 0/1
    
    # True Categoricals for OneHot:
    categorical_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]
    
    # All other columns from FeatureEngineer will be numeric/boolean
    # We need to capture them.
    # Base numeric: tenure, MonthlyCharges, TotalCharges
    numerical_cols = base_num + derived_num 
    
    return {
        "numerical": list(set(numerical_cols)),
        "categorical": categorical_cols
    }

def create_preprocessor() -> ColumnTransformer:
    """
    Creates the sklearn ColumnTransformer for scaling and encoding.
    """
    feature_lists = get_feature_lists()
    
    # Numeric Pipeline: Impute -> Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical Pipeline: Impute -> OneHot
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, feature_lists["numerical"]),
            ('cat', categorical_transformer, feature_lists["categorical"])
        ],
        remainder='drop' # Drop columns not specified (e.g. original ID if present)
    )
    
    return preprocessor

def create_model_pipeline(model_type: str = "random_forest", params: Optional[Dict] = None) -> Pipeline:
    """
    Creates the full end-to-end pipeline:
    Feature Engineering -> Preprocessing -> Classifier
    """
    if params is None:
        params = {}
        
    # 1. Select Classifier
    if model_type == "logistic_regression":
        clf = LogisticRegression(**params)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(**params)
    elif model_type == "xgboost":
        if xgb is None:
            raise ImportError("XGBoost is not installed or dependencies (libomp) are missing.")
        clf = xgb.XGBClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # 2. Build Pipeline
    # Note: FeatureEngineer returns a DataFrame.
    # ColumnTransformer takes that DataFrame and splits it by column name.
    
    pipeline = Pipeline(steps=[
        ('cleaner', CategoricalCleaner()),         # Fix types, map binary strings
        ('engineer', FeatureEngineer()),           # Add derived features
        ('preprocessor', create_preprocessor()),   # Scale & Encode
        ('classifier', clf)
    ])
    
    return pipeline

def save_model(pipeline: Any, model_path: str) -> None:
    """
    Save the full pipeline to disk.
    """
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)

def load_model(model_path: str) -> Any:
    """
    Load the full pipeline from disk.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

def predict_churn(pipeline: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Predict churn probabilities using the pipeline.
    """
    # The pipeline handles preprocessing, so we pass raw X
    # ensure probability of positive class
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    else:
        return pipeline.predict(X)

