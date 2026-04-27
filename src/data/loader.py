"""
Data loading utilities for customer churn analysis.

This module provides functions to load and validate customer data
from various sources (CSV files, processed data directories).
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd


def load_csv_data(
    file_path: str | Path,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Load customer data from a CSV file.

    Args:
        file_path: Path to the CSV file
        encoding: File encoding (default: utf-8)

    Returns:
        DataFrame containing the loaded data

    Raises:
        FileNotFoundError: If the file does not exist
        pd.errors.EmptyDataError: If the file is empty
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, encoding=encoding)

    # Convert TotalCharges to numeric (may contain spaces)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


def load_processed_data(
    data_dir: str | Path,
    filename: str = "data_processed_final.csv",
) -> pd.DataFrame:
    """
    Load preprocessed data from the processed data directory.

    Args:
        data_dir: Path to the processed data directory
        filename: Name of the processed data file

    Returns:
        DataFrame containing the processed data
    """
    path = Path(data_dir) / filename
    return load_csv_data(path)


def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that the DataFrame contains all required columns.

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, list of missing columns)
    """
    required_columns = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    is_valid = len(missing) == 0

    return is_valid, missing


def get_feature_names(file_path: str | Path) -> List[str]:
    """
    Load feature names from a text file.

    Args:
        file_path: Path to the feature names file

    Returns:
        List of feature names in order
    """
    path = Path(file_path)
    feature_list = []

    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        stripped_line = line.strip()
        if (
            stripped_line
            and not stripped_line.startswith("==")
            and not stripped_line.startswith("Feature")
        ):
            parts = stripped_line.split(".", 1)
            if len(parts) > 1:
                feature_name = parts[1].strip()
                if feature_name:
                    feature_list.append(feature_name)

    return feature_list


def split_data(
    df: pd.DataFrame,
    target_column: str = "Churn",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.

    Args:
        df: DataFrame containing features and target
        target_column: Name of the target column
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def temporal_split(
    df: pd.DataFrame,
    target_column: str = "Churn",
    tenure_col: str = "tenure",
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time-ordered split using tenure as a monotone proxy for acquisition date.

    Customers with higher tenure joined the company earlier and represent the
    historical cohort used for training.  The most recently acquired customers
    (lowest tenure) form the test set — the harder, more realistic scenario for
    a deployed churn model.

    The dataset has no explicit observation timestamp; tenure (0–72 months) is
    the closest available proxy.  Ties within a tenure value are kept in their
    original CSV order (stable sort) so the split is deterministic.

    Splits at the (1 - test_size) row boundary after descending-tenure sort:
      train → top 80 % rows  (tenure range ~6–72 months)
      test  → bottom 20 % rows (tenure range 0–6 months)

    Args:
        df: Raw DataFrame including target column
        target_column: Name of the target column
        tenure_col: Column used as the time proxy
        test_size: Fraction of rows reserved for the test set

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    df_sorted = df.sort_values(tenure_col, ascending=False, kind="stable").reset_index(
        drop=True
    )
    n_train = int(len(df_sorted) * (1 - test_size))

    train = df_sorted.iloc[:n_train].copy()
    test = df_sorted.iloc[n_train:].copy()

    X_train = train.drop(columns=[target_column])
    y_train = train[target_column]
    X_test = test.drop(columns=[target_column])
    y_test = test[target_column]

    return X_train, X_test, y_train, y_test
