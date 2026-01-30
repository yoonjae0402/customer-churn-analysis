"""
Feature engineering and preprocessing utilities.

This module contains functions to transform raw customer data
into features suitable for machine learning models.
"""

from typing import List, Optional

import numpy as np
import pandas as pd


def create_tenure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create tenure-based features.

    Args:
        df: DataFrame with 'tenure' column

    Returns:
        DataFrame with additional tenure features
    """
    df = df.copy()

    # Tenure groups
    bins = [0, 12, 24, 48, 72]
    labels = ["0-12", "12-24", "24-48", "48+"]
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=bins, labels=labels, right=False
    ).astype(str)

    # Binary flags
    df["is_new_customer"] = (df["tenure"] <= 12).astype(int)
    df["is_long_term"] = (df["tenure"] > 24).astype(int)

    # Binned tenure (8 bins)
    df["tenure_bins"] = pd.cut(df["tenure"], bins=8, labels=False, right=False)
    df["tenure_bins"] = df["tenure_bins"].fillna(0).astype(int)

    return df


def create_charge_features(
    df: pd.DataFrame, monthly_median: Optional[float] = None
) -> pd.DataFrame:
    """
    Create charge-based features.

    Args:
        df: DataFrame with 'MonthlyCharges', 'TotalCharges', 'tenure'
        monthly_median: Reference median for high charge flag (default: calculated)

    Returns:
        DataFrame with additional charge features
    """
    df = df.copy()

    # Handle missing TotalCharges
    if df["TotalCharges"].isna().any():
        mask = df["TotalCharges"].isna()
        df.loc[mask, "TotalCharges"] = df.loc[mask, "MonthlyCharges"] * df.loc[
            mask, "tenure"
        ]

    # Average monthly spend
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"].replace(0, np.nan) + 1)
    df["avg_monthly_spend"].fillna(0, inplace=True)

    # Charge per tenure
    df["charge_per_tenure"] = df["MonthlyCharges"] / (
        df["tenure"].replace(0, np.nan) + 1
    )
    df["charge_per_tenure"].fillna(0, inplace=True)

    # Price increase indicator
    df["price_increase"] = (df["MonthlyCharges"] > df["avg_monthly_spend"]).astype(int)

    # High monthly charge flag
    if monthly_median is None:
        monthly_median = df["MonthlyCharges"].median()
    df["high_monthly_charge"] = (df["MonthlyCharges"] > monthly_median).astype(int)

    return df


def create_service_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create service-based features.

    Args:
        df: DataFrame with service columns

    Returns:
        DataFrame with additional service features
    """
    df = df.copy()

    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    # Total services count
    df["total_services"] = 0
    for col in service_cols:
        if col in df.columns:
            df["total_services"] += (df[col] == "Yes").astype(int)

    # Service flags
    df["has_internet"] = (df["InternetService"] != "No").astype(int)
    df["has_phone"] = (df["PhoneService"] == "Yes").astype(int)
    df["has_tech_support"] = (df["TechSupport"] == "Yes").astype(int)
    df["has_online_security"] = (df["OnlineSecurity"] == "Yes").astype(int)

    return df


def create_contract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create contract and payment features.

    Args:
        df: DataFrame with contract/payment columns

    Returns:
        DataFrame with additional contract features
    """
    df = df.copy()

    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    auto_payment_methods = ["Bank transfer (automatic)", "Credit card (automatic)"]
    df["has_auto_payment"] = df["PaymentMethod"].isin(auto_payment_methods).astype(int)

    df["is_paperless"] = (df["PaperlessBilling"] == "Yes").astype(int)

    return df


def create_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create demographic features.

    Args:
        df: DataFrame with demographic columns

    Returns:
        DataFrame with additional demographic features
    """
    df = df.copy()

    df["is_senior"] = df["SeniorCitizen"]

    has_partner = (df["Partner"] == "Yes").astype(int)
    has_dependents = (df["Dependents"] == "Yes").astype(int)
    df["has_family"] = ((has_partner + has_dependents) > 0).astype(int)

    df["is_single_no_deps"] = (
        (df["Partner"] == "No") & (df["Dependents"] == "No")
    ).astype(int)

    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables.

    Args:
        df: DataFrame with categorical columns

    Returns:
        DataFrame with encoded features
    """
    df = df.copy()

    # Binary encoding
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        if col in df.columns:
            if col == "gender":
                df[col] = df[col].map({"Male": 1, "Female": 0}).fillna(0).astype(int)
            else:
                df[col] = (df[col] == "Yes").astype(int)

    # One-hot encoding
    cols_for_onehot = [
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
        "tenure_group",
    ]

    cols_present = [col for col in cols_for_onehot if col in df.columns]
    df = pd.get_dummies(df, columns=cols_present, drop_first=False)

    return df


def preprocess_customer_data(
    df: pd.DataFrame,
    feature_names: List[str],
    scaler=None,
    monthly_median: Optional[float] = None,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for customer data.

    Args:
        df: Raw customer DataFrame
        feature_names: List of expected feature names (in order)
        scaler: Optional fitted scaler for numerical features
        monthly_median: Reference median for charge features

    Returns:
        Preprocessed DataFrame ready for model inference
    """
    # Apply all feature engineering steps
    df = create_tenure_features(df)
    df = create_charge_features(df, monthly_median)
    df = create_service_features(df)
    df = create_contract_features(df)
    df = create_demographic_features(df)
    df = encode_categorical_features(df)

    # Align columns with expected features
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Select and order columns
    df = df[feature_names]

    return df
