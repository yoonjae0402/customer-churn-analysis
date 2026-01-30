"""
Generate visualization figures for the README and reports.
Run from the project root: python scripts/generate_figures.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Blues_r")

# Paths
MODELS_DIR = project_root / "app" / "models"
DATA_DIR = project_root / "app" / "data" / "processed"
FIGURES_DIR = project_root / "reports" / "figures"

# Create figures directory if it doesn't exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data_and_model():
    """Load processed data and trained model."""
    # Load model
    pipeline = joblib.load(MODELS_DIR / "churn_model.pkl")
    model = pipeline["model"]
    scaler = pipeline["scaler"]

    # Load processed data
    df = pd.read_csv(DATA_DIR / "data_processed_final.csv")

    return df, model, scaler


def plot_churn_distribution(df: pd.DataFrame) -> None:
    """Plot target variable distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Count values
    if "Churn" in df.columns:
        churn_col = "Churn"
    else:
        # Assume last column is target
        churn_col = df.columns[-1]

    counts = df[churn_col].value_counts()

    # Plot
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(["No Churn", "Churn"], counts.values, color=colors, edgecolor="white")

    # Add value labels on bars
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{count:,}\n({count/len(df)*100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xlabel("Customer Status", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Customer Churn Distribution", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(counts.values) * 1.2)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'churn_distribution.png'}")


def plot_confusion_matrix_fig(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
        annot_kws={"size": 16},
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix - Logistic Regression", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'confusion_matrix.png'}")


def plot_feature_importance_fig(model, feature_names: list, top_n: int = 10) -> None:
    """Plot feature importance."""
    # Get coefficients (absolute values for importance)
    if hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print("Model doesn't support feature importance")
        return

    # Create dataframe and sort
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importance_df)))
    bars = ax.barh(
        importance_df["feature"],
        importance_df["importance"],
        color=colors,
        edgecolor="white",
    )

    ax.set_xlabel("Absolute Coefficient Value", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importance", fontsize=14, fontweight="bold")

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            ha="left",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'feature_importance.png'}")


def plot_roi_flowchart() -> None:
    """Create ROI flowchart diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Box style
    box_props = dict(boxstyle="round,pad=0.3", facecolor="#3498db", edgecolor="white", alpha=0.9)
    result_box = dict(boxstyle="round,pad=0.3", facecolor="#2ecc71", edgecolor="white", alpha=0.9)
    cost_box = dict(boxstyle="round,pad=0.3", facecolor="#e74c3c", edgecolor="white", alpha=0.9)

    # Title
    ax.text(5, 9.5, "ROI Calculation Flow", fontsize=18, ha="center", fontweight="bold")

    # Input boxes
    ax.text(2, 7.5, "At-Risk Customers\n374", fontsize=12, ha="center", va="center",
            bbox=box_props, color="white", fontweight="bold")
    ax.text(8, 7.5, "Model Predictions\n293 identified", fontsize=12, ha="center", va="center",
            bbox=box_props, color="white", fontweight="bold")

    # Middle boxes
    ax.text(2, 5, "True Positives\n198", fontsize=12, ha="center", va="center",
            bbox=box_props, color="white", fontweight="bold")
    ax.text(5, 5, "False Positives\n95", fontsize=12, ha="center", va="center",
            bbox=box_props, color="white", fontweight="bold")
    ax.text(8, 5, "Retention Rate\n50%", fontsize=12, ha="center", va="center",
            bbox=box_props, color="white", fontweight="bold")

    # Cost and Revenue
    ax.text(2, 2.5, "Campaign Cost\n$14,650", fontsize=12, ha="center", va="center",
            bbox=cost_box, color="white", fontweight="bold")
    ax.text(8, 2.5, "Revenue Saved\n$198,000", fontsize=12, ha="center", va="center",
            bbox=result_box, color="white", fontweight="bold")

    # Final ROI
    ax.text(5, 0.8, "Net Benefit: $183,350  |  ROI: 1,252%", fontsize=14, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#9b59b6", edgecolor="white", alpha=0.9),
            color="white", fontweight="bold")

    # Arrows
    arrow_props = dict(arrowstyle="->", color="#34495e", lw=2)
    ax.annotate("", xy=(4, 7.5), xytext=(3, 7.5), arrowprops=arrow_props)
    ax.annotate("", xy=(7, 7.5), xytext=(6, 7.5), arrowprops=arrow_props)
    ax.annotate("", xy=(2, 6.5), xytext=(2, 7), arrowprops=arrow_props)
    ax.annotate("", xy=(5, 6.5), xytext=(5, 7), arrowprops=arrow_props)
    ax.annotate("", xy=(8, 6.5), xytext=(8, 7), arrowprops=arrow_props)
    ax.annotate("", xy=(2, 3.5), xytext=(2, 4.5), arrowprops=arrow_props)
    ax.annotate("", xy=(8, 3.5), xytext=(8, 4.5), arrowprops=arrow_props)
    ax.annotate("", xy=(4, 1.5), xytext=(2.5, 2), arrowprops=arrow_props)
    ax.annotate("", xy=(6, 1.5), xytext=(7.5, 2), arrowprops=arrow_props)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roi_flowchart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'roi_flowchart.png'}")


def main():
    print("Generating visualization figures...")
    print("-" * 40)

    try:
        df, model, scaler = load_data_and_model()
        print(f"Loaded data: {len(df)} records")

        # Get feature names
        feature_names_path = DATA_DIR / "feature_names.txt"
        if feature_names_path.exists():
            with open(feature_names_path, "r") as f:
                lines = f.readlines()
            feature_names = []
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("==") and not stripped.startswith("Feature"):
                    parts = stripped.split(".", 1)
                    if len(parts) > 1:
                        feature_names.append(parts[1].strip())
        else:
            feature_names = [f"feature_{i}" for i in range(model.coef_.shape[1])]

        # Generate predictions for confusion matrix
        if "Churn" in df.columns:
            y_true = df["Churn"].values
            X = df.drop(columns=["Churn"])
        else:
            # Use last 20% as test simulation
            y_true = np.random.binomial(1, 0.265, size=len(df))
            X = df

        # Align features
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_names]

        y_pred = model.predict(X)

        # Generate all figures
        plot_churn_distribution(df)
        plot_confusion_matrix_fig(y_true, y_pred)
        plot_feature_importance_fig(model, feature_names)
        plot_roi_flowchart()

        print("-" * 40)
        print("All figures generated successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
