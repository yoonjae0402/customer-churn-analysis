"""
Customer Churn Analysis - Utility Modules

This package provides reusable utilities for:
- Data loading and processing
- Feature engineering and preprocessing
- Model training and evaluation
"""

from .data_loader import load_csv_data, load_processed_data, validate_data
from .preprocessing import (
    create_tenure_features,
    create_charge_features,
    create_service_features,
    create_contract_features,
    create_demographic_features,
    encode_categorical_features,
    preprocess_customer_data,
)
from .model import load_model, save_model, train_model, predict_churn
from .evaluation import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    generate_classification_report,
)

__version__ = "1.0.0"
__all__ = [
    "load_csv_data",
    "load_processed_data",
    "validate_data",
    "create_tenure_features",
    "create_charge_features",
    "create_service_features",
    "create_contract_features",
    "create_demographic_features",
    "encode_categorical_features",
    "preprocess_customer_data",
    "load_model",
    "save_model",
    "train_model",
    "predict_churn",
    "calculate_metrics",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_roc_curve",
    "generate_classification_report",
]
