"""
Model evaluation utilities.

This module provides functions for evaluating model performance,
generating metrics, and creating visualizations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for ROC-AUC)

    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> str:
    """
    Generate a formatted classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional names for classes

    Returns:
        Formatted classification report string
    """
    if target_names is None:
        target_names = ["No Churn", "Churn"]

    return classification_report(y_true, y_pred, target_names=target_names)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str | Path] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    Plot and optionally save a confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")

    plt.close()


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 10,
    save_path: Optional[str | Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> pd.DataFrame:
    """
    Plot feature importance from a trained model.

    Args:
        model: Trained model with feature_importances_ or coef_ attribute
        feature_names: List of feature names
        top_n: Number of top features to display
        save_path: Optional path to save the figure
        figsize: Figure size

    Returns:
        DataFrame of feature importances
    """
    import matplotlib.pyplot as plt

    # Get importance values
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model doesn't have feature_importances_ or coef_ attribute")

    # Create DataFrame
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    # Plot
    top_features = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        range(len(top_features)), top_features["importance"].values, color="steelblue"
    )
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importance")

    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")

    plt.close()

    return importance_df


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[str | Path] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> float:
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Optional path to save the figure
        figsize: Figure size

    Returns:
        ROC-AUC score
    """
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC curve (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")

    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")

    plt.close()

    return auc_score


def compare_models(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare multiple model results.

    Args:
        results: List of dicts with 'model_name' and metric values

    Returns:
        DataFrame comparing model performance
    """
    return pd.DataFrame(results).set_index("model_name").sort_values(
        "roc_auc", ascending=False
    )
