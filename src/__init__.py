"""
Customer Churn Analysis - Utility Modules
"""

from src.config import config
from src.data.loader import load_csv_data, split_data, validate_data
from src.features.engineering import CategoricalCleaner, FeatureEngineer
from src.logger import logger
from src.models.pipeline import (
    create_model_pipeline,
    load_model,
    predict_churn,
    save_model,
)

__version__ = "2.0.0"

__all__ = [
    "config",
    "logger",
    "load_csv_data",
    "split_data",
    "validate_data",
    "FeatureEngineer",
    "CategoricalCleaner",
    "create_model_pipeline",
    "save_model",
    "load_model",
    "predict_churn",
]
