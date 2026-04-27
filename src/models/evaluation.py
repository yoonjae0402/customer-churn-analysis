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
    precision_recall_curve,
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
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
        # PR-AUC via trapezoidal integration (more meaningful for imbalanced data)
        metrics["pr_auc"] = float(np.trapz(precision_vals, recall_vals) * -1)

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


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    strategy: str = "f1",
    cost_fn: float = 1.0,
    cost_fp: float = 0.1,
) -> Tuple[float, Dict[str, float]]:
    """
    Find the optimal classification threshold.

    Three strategies:
    - "f1": Maximizes F1 score on the churn class (default).
    - "recall_at_precision": Maximizes recall subject to precision >= 0.5.
    - "cost": Minimizes business cost (FN = lost customer, FP = wasted offer).

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for the positive class
        strategy: One of "f1", "recall_at_precision", "cost"
        cost_fn: Cost of a false negative (missed churner) — used only for "cost"
        cost_fp: Cost of a false positive (wasted retention offer) — used only for
            "cost"

    Returns:
        Tuple of (best_threshold, metrics_at_threshold)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # thresholds has len = len(precisions) - 1; pair them up
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    if strategy == "f1":
        denom = precisions + recalls
        safe_denom = np.where(denom > 0, denom, 1)
        f1_scores = np.where(denom > 0, 2 * precisions * recalls / safe_denom, 0)
        best_idx = int(np.argmax(f1_scores))

    elif strategy == "recall_at_precision":
        # Maximise recall where precision >= 0.5
        valid = precisions >= 0.5
        if not valid.any():
            best_idx = int(np.argmax(precisions))
        else:
            best_idx = int(np.argmax(np.where(valid, recalls, -1)))

    elif strategy == "cost":
        total_costs = []
        for p, r, t in zip(precisions, recalls, thresholds):
            tp = r * y_true.sum()
            fp = tp / p - tp if p > 0 else 0
            fn = y_true.sum() - tp
            total_costs.append(cost_fn * fn + cost_fp * fp)
        best_idx = int(np.argmin(total_costs))

    else:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. "
            "Use 'f1', 'recall_at_precision', or 'cost'."
        )

    best_threshold = float(thresholds[best_idx])
    y_pred_at_threshold = (y_proba >= best_threshold).astype(int)

    metrics_at_threshold = {
        "threshold": best_threshold,
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
        "f1": float(
            2
            * precisions[best_idx]
            * recalls[best_idx]
            / (precisions[best_idx] + recalls[best_idx])
            if (precisions[best_idx] + recalls[best_idx]) > 0
            else 0
        ),
        "accuracy": float(accuracy_score(y_true, y_pred_at_threshold)),
    }

    return best_threshold, metrics_at_threshold


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
    ax.barh(
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
    ax.plot(
        fpr, tpr, color="steelblue", lw=2, label=f"ROC curve (AUC = {auc_score:.3f})"
    )
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


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    optimal_threshold: Optional[float] = None,
    save_path: Optional[str | Path] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> float:
    """
    Plot Precision-Recall curve with optional optimal threshold marker.

    PR-AUC is more informative than ROC-AUC for imbalanced datasets because
    it focuses on the minority (churn) class performance.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        optimal_threshold: If provided, marks the operating point on the curve
        save_path: Optional path to save the figure
        figsize: Figure size

    Returns:
        PR-AUC score
    """
    import matplotlib.pyplot as plt

    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = float(np.trapz(precision_vals, recall_vals) * -1)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        recall_vals,
        precision_vals,
        color="steelblue",
        lw=2,
        label=f"PR curve (AUC = {pr_auc:.3f})",
    )
    ax.axhline(
        y=baseline,
        color="gray",
        lw=1,
        linestyle="--",
        label=f"Baseline (random): {baseline:.3f}",
    )

    if optimal_threshold is not None:
        # Find the point on the curve closest to this threshold
        idx = np.searchsorted(thresholds, optimal_threshold, side="left")
        idx = min(idx, len(precision_vals) - 2)
        ax.scatter(
            recall_vals[idx],
            precision_vals[idx],
            color="red",
            zorder=5,
            s=100,
            label=f"Optimal threshold ({optimal_threshold:.2f})",
        )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Churn Class)")
    ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")

    plt.close()

    return pr_auc


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str | Path] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    Plot reliability diagram (calibration curve).

    A well-calibrated model has predicted probabilities that match
    actual churn rates. Tree-based models are often overconfident —
    this plot reveals whether the raw probabilities are trustworthy.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of calibration bins
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform"
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        color="steelblue",
        label="Model",
    )
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Actual Churn Rate)")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend(loc="upper left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")

    plt.close()


def plot_learning_curves(
    pipeline: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "roc_auc",
    train_sizes: Optional[np.ndarray] = None,
    save_path: Optional[str | Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot learning curves to diagnose bias/variance and confirm no overfitting.

    If train and validation scores converge at a high value — the model generalises.
    A large gap means overfitting; both curves low means underfitting.

    Args:
        pipeline: Fitted or unfitted sklearn pipeline
        X: Feature DataFrame
        y: Target array
        cv: Number of cross-validation folds
        scoring: Scoring metric
        train_sizes: Array of training set sizes (fractions)
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 8)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        train_sizes=train_sizes,
        n_jobs=-1,
        shuffle=True,
        random_state=42,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        train_sizes_abs, train_mean, "o-", color="steelblue", label="Training score"
    )
    ax.fill_between(
        train_sizes_abs,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color="steelblue",
    )
    ax.plot(
        train_sizes_abs,
        val_mean,
        "o-",
        color="darkorange",
        label="Cross-validation score",
    )
    ax.fill_between(
        train_sizes_abs,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15,
        color="darkorange",
    )

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel(scoring.upper().replace("_", "-"))
    ax.set_title("Learning Curves")
    ax.legend(loc="lower right")
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")

    plt.close()


def plot_shap_summary(
    pipeline: Any,
    X: pd.DataFrame,
    max_display: int = 15,
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Generate SHAP summary plot showing feature impact on churn predictions.

    SHAP values explain *why* the model made each prediction, and which
    features push the probability up or down for individual customers.
    This is more actionable than raw feature importance.

    Requires: pip install shap

    Args:
        pipeline: Fitted sklearn pipeline (must include 'preprocessor' and
            'classifier' steps)
        X: Raw feature DataFrame (before preprocessing)
        max_display: Max number of features to show
        save_path: Optional path to save the figure
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP is required for this plot. Run: pip install shap")

    import matplotlib.pyplot as plt

    # Transform X through all steps except the final classifier
    preprocessing_steps = pipeline.steps[:-1]
    from sklearn.pipeline import Pipeline as SKPipeline

    pre_pipeline = SKPipeline(preprocessing_steps)
    X_transformed = pre_pipeline.transform(X)

    classifier = pipeline.named_steps["classifier"]

    # Get feature names after preprocessing
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out().tolist()
        # Clean up sklearn-generated names like "num__tenure" -> "tenure"
        feature_names = [n.split("__", 1)[-1] for n in feature_names]
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    # Choose explainer based on model type
    try:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed)
        # For binary classifiers, shap_values may be a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception:
        explainer = shap.LinearExplainer(classifier, X_transformed)
        shap_values = explainer.shap_values(X_transformed)

    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )

    plt.title("SHAP Feature Impact on Churn Probability")
    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")

    plt.close()


def compare_models(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare multiple model results.

    Args:
        results: List of dicts with 'model_name' and metric values

    Returns:
        DataFrame comparing model performance
    """
    return (
        pd.DataFrame(results)
        .set_index("model_name")
        .sort_values("roc_auc", ascending=False)
    )
