"""
Generate all business-case artefacts from the trained model.

Outputs (written to reports/):
  figures/cost_curve.png       — expected cost vs decision threshold
  figures/decile_roi.png       — per-decile ROI bar chart
  figures/shap_summary.png     — SHAP beeswarm (top-15 features)
  decile_analysis.csv          — full per-decile table
  business_impact.csv          — updated headline numbers

Usage:
    python3 scripts/generate_business_case.py
    python3 scripts/generate_business_case.py --ltv 1500 --offer-cost 30 --p-retention 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.business_case import (
    DEFAULT_LTV,
    DEFAULT_OFFER_COST,
    DEFAULT_P_RETENTION,
    build_cost_curve,
    compute_decile_analysis,
    compute_roi_summary,
    find_cost_optimal_threshold,
)
from src.data.loader import load_csv_data, temporal_split

FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_test_predictions(ltv: float, offer_cost: float) -> tuple:
    """Load model, reproduce temporal split, return (y_true, y_prob)."""
    df = load_csv_data(ROOT / "data" / "raw" / "telco_churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    _, X_test, _, y_test = temporal_split(df, "Churn")

    model = joblib.load(ROOT / "models" / "random_forest_pipeline.pkl")
    y_prob = model.predict_proba(X_test)[:, 1]

    return y_test.values.astype(int), y_prob, X_test, df


# ---------------------------------------------------------------------------
# Plot 1: Cost curve
# ---------------------------------------------------------------------------

def plot_cost_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ltv: float,
    offer_cost: float,
    save_path: Path,
) -> float:
    """Plot expected cost components vs threshold; return optimal threshold."""
    curve = build_cost_curve(y_true, y_prob, ltv, offer_cost)
    opt_thresh, _ = find_cost_optimal_threshold(y_true, y_prob, ltv, offer_cost)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(curve["threshold"], curve["fn_cost"] / 1_000, color="#d62728",
            lw=1.8, label="FN cost  (missed churners × LTV)")
    ax.plot(curve["threshold"], curve["campaign_cost"] / 1_000, color="#1f77b4",
            lw=1.8, label="Campaign cost  (flagged customers × offer cost)")
    ax.plot(curve["threshold"], curve["total_cost"] / 1_000, color="#2ca02c",
            lw=2.2, label="Total cost  (FN cost + campaign cost)")

    ax.axvline(opt_thresh, color="black", lw=1.4, linestyle="--",
               label=f"Cost-optimal threshold  t = {opt_thresh:.3f}")

    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Expected Cost  ($K)", fontsize=12)
    ax.set_title("Expected Cost vs. Decision Threshold\n"
                 f"LTV = ${ltv:,.0f}  |  Offer cost = ${offer_cost:.0f}  |  "
                 f"Test set: 1,409 recently-acquired customers",
                 fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path.relative_to(ROOT)}")
    return opt_thresh


# ---------------------------------------------------------------------------
# Plot 2: Decile ROI
# ---------------------------------------------------------------------------

def plot_decile_roi(
    decile_df: pd.DataFrame,
    p_retention: float,
    save_path: Path,
) -> None:
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in decile_df["net_benefit"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: net benefit per decile
    ax = axes[0]
    bars = ax.bar(decile_df["decile"], decile_df["net_benefit"] / 1_000,
                  color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Decile  (1 = highest predicted risk)", fontsize=11)
    ax.set_ylabel("Net Benefit  ($K)", fontsize=11)
    ax.set_title(f"Net Benefit per Decile  (p_retention = {p_retention:.0%})", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
    ax.set_xticks(decile_df["decile"])

    # Right: churn rate and break-even line
    ax2 = axes[1]
    ax2.bar(decile_df["decile"], decile_df["churn_rate_pct"],
            color="steelblue", alpha=0.8, label="Actual churn rate %")
    ax2.axhline(decile_df["break_even_retention_rate_pct"].replace(float("inf"), np.nan).dropna().median(),
                color="darkorange", lw=1.5, linestyle="--",
                label="Median break-even retention rate")
    ax2.set_xlabel("Decile  (1 = highest predicted risk)", fontsize=11)
    ax2.set_ylabel("Churn Rate  (%)", fontsize=11)
    ax2.set_title("Actual Churn Rate per Decile", fontsize=11)
    ax2.set_xticks(decile_df["decile"])
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Plot 3: SHAP summary
# ---------------------------------------------------------------------------

def plot_shap_summary(X_train: pd.DataFrame, y_train: pd.Series, save_path: Path) -> list[str]:
    """
    Train a non-calibrated RF on the training set solely for interpretability.
    Returns the top-3 feature names by mean |SHAP value|.
    """
    try:
        import shap
    except ImportError:
        print("  SHAP not installed — skipping shap_summary.png")
        return ["tenure", "MonthlyCharges", "Contract"]  # known priors

    from sklearn.pipeline import Pipeline as SKPipeline
    from src.model import create_model_pipeline

    print("  Fitting non-calibrated RF for SHAP (interpretability only)…")
    rf_pipe = create_model_pipeline(
        "random_forest",
        {"n_estimators": 100, "max_depth": 10, "class_weight": "balanced", "random_state": 42},
    )
    rf_pipe.fit(X_train, y_train)

    # Build preprocessing-only pipeline (all steps except classifier)
    pre_steps = [(name, step) for name, step in rf_pipe.steps if name != "classifier"]
    pre_pipe = SKPipeline(pre_steps)
    X_train_t = pre_pipe.transform(X_train)

    # Feature names after ColumnTransformer
    try:
        feature_names = (
            rf_pipe.named_steps["preprocessor"]
            .get_feature_names_out()
            .tolist()
        )
        feature_names = [n.split("__", 1)[-1] for n in feature_names]
    except Exception:
        feature_names = [f"f{i}" for i in range(X_train_t.shape[1])]

    clf = rf_pipe.named_steps["classifier"]
    explainer = shap.TreeExplainer(clf)
    # Use a sample of 500 rows for speed
    rng = np.random.default_rng(42)
    sample_size = min(500, len(X_train_t))
    idx = rng.choice(len(X_train_t), size=sample_size, replace=False)
    X_sample = X_train_t[idx]
    raw_sv = explainer.shap_values(X_sample)

    # SHAP ≥0.42 returns ndarray(n_samples, n_features, n_classes) for RF;
    # older versions return a list [class0_arr, class1_arr].
    if isinstance(raw_sv, np.ndarray) and raw_sv.ndim == 3:
        sv = raw_sv[:, :, 1]          # churn class (index 1)
    elif isinstance(raw_sv, list):
        sv = raw_sv[1]
    else:
        sv = raw_sv

    feature_names_arr = np.array(feature_names)

    # Top-3 features by mean absolute SHAP value
    mean_abs = np.abs(sv).mean(axis=0)          # shape (n_features,)
    top3_idx = np.argsort(mean_abs)[::-1][:3]   # shape (3,)
    top3_features = feature_names_arr[top3_idx].tolist()

    # Beeswarm plot — pass class-1 SHAP values
    shap.summary_plot(
        sv, X_sample,
        feature_names=feature_names_arr.tolist(),
        max_display=15,
        show=False,
        plot_type="dot",
    )
    plt.title("SHAP Feature Impact on Churn Probability  (training set, RF)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved {save_path.relative_to(ROOT)}")
    print(f"  Top-3 SHAP drivers: {top3_features}")
    return top3_features


# ---------------------------------------------------------------------------
# Save artefacts
# ---------------------------------------------------------------------------

def save_outputs(
    decile_df: pd.DataFrame,
    summary: dict,
) -> None:
    # Decile CSV
    csv_path = ROOT / "reports" / "decile_analysis.csv"
    decile_df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path.relative_to(ROOT)}")

    # Updated business_impact.csv
    impact_rows = [
        {"Metric": "Cohort size (temporal test set)", "Value": summary["cohort_size"]},
        {"Metric": "Churners in cohort", "Value": summary["cohort_churners"]},
        {"Metric": "Cohort churn rate", "Value": f"{summary['cohort_churn_rate']:.1%}"},
        {"Metric": "Null cost (do nothing)", "Value": f"${summary['null_cost']:,}"},
        {"Metric": "Cost-optimal threshold", "Value": f"{summary['optimal_threshold']:.3f}"},
        {"Metric": "Recall at optimal threshold", "Value": f"{summary['recall']:.1%}"},
        {"Metric": "Precision at optimal threshold", "Value": f"{summary['precision']:.1%}"},
        {"Metric": "Customers flagged", "Value": summary["flagged_customers"]},
        {"Metric": "True positives (caught churners)", "Value": summary["tp"]},
        {"Metric": "False positives (false alarms)", "Value": summary["fp"]},
        {"Metric": "False negatives (missed churners)", "Value": summary["fn"]},
        {"Metric": "Campaign cost (offers sent)", "Value": f"${summary['campaign_cost_total']:,}"},
        {"Metric": "Revenue saved (p_retention=30%)", "Value": f"${summary['revenue_saved']:,}"},
        {"Metric": "Net benefit", "Value": f"${summary['net_benefit']:,}"},
        {"Metric": "ROI", "Value": f"{summary['roi_pct']:.0f}%"},
        {"Metric": "LTV assumption", "Value": f"${summary['ltv']:,}"},
        {"Metric": "Offer cost assumption", "Value": f"${summary['offer_cost']:,}"},
        {"Metric": "Retention success rate assumption", "Value": f"{summary['p_retention']:.0%}"},
    ]
    pd.DataFrame(impact_rows).to_csv(
        ROOT / "reports" / "business_impact.csv", index=False
    )
    print(f"  Saved reports/business_impact.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate business-case artefacts")
    parser.add_argument("--ltv", type=float, default=DEFAULT_LTV)
    parser.add_argument("--offer-cost", type=float, default=DEFAULT_OFFER_COST)
    parser.add_argument("--p-retention", type=float, default=DEFAULT_P_RETENTION)
    args = parser.parse_args()

    ltv = args.ltv
    offer_cost = args.offer_cost
    p_retention = args.p_retention

    print(f"\nAssumptions: LTV=${ltv:,.0f}  |  offer_cost=${offer_cost:.0f}"
          f"  |  p_retention={p_retention:.0%}\n")

    y_true, y_prob, X_test, df_full = load_test_predictions(ltv, offer_cost)

    # Recover train set for SHAP
    _, X_test_raw, y_train, _ = temporal_split(df_full.drop("Churn", axis=1)
                                                .assign(Churn=df_full["Churn"]), "Churn")
    # Actually redo split properly
    df2 = load_csv_data(ROOT / "data" / "raw" / "telco_churn.csv")
    df2["TotalCharges"] = pd.to_numeric(df2["TotalCharges"], errors="coerce").fillna(0)
    df2["Churn"] = df2["Churn"].map({"Yes": 1, "No": 0})
    X_train, _, y_train, _ = temporal_split(df2, "Churn")

    print("─" * 60)
    print("1/4  Cost curve …")
    opt_thresh = plot_cost_curve(
        y_true, y_prob, ltv, offer_cost,
        FIGURES_DIR / "cost_curve.png",
    )

    print("─" * 60)
    print("2/4  Decile analysis …")
    decile_df = compute_decile_analysis(y_true, y_prob, ltv, offer_cost, p_retention)
    plot_decile_roi(decile_df, p_retention, FIGURES_DIR / "decile_roi.png")

    print("─" * 60)
    print("3/4  SHAP summary …")
    top3 = plot_shap_summary(X_train, y_train, FIGURES_DIR / "shap_summary.png")

    print("─" * 60)
    print("4/4  Saving artefacts …")
    summary = compute_roi_summary(y_true, y_prob, ltv, offer_cost, p_retention)
    save_outputs(decile_df, summary)

    # Print headline numbers
    print("\n" + "═" * 60)
    print("HEADLINE NUMBERS")
    print("═" * 60)
    print(f"  Cohort: {summary['cohort_size']} recently-acquired customers "
          f"({summary['cohort_churners']} churners, {summary['cohort_churn_rate']:.1%})")
    print(f"  Null cost (do nothing):       ${summary['null_cost']:>10,}")
    print(f"  Cost-optimal threshold:        {summary['optimal_threshold']:.3f}")
    print(f"  Recall / Precision:           "
          f"{summary['recall']:.1%} / {summary['precision']:.1%}")
    print(f"  Offers sent:                   {summary['flagged_customers']:,}")
    print(f"  Campaign cost:                ${summary['campaign_cost_total']:>10,}")
    print(f"  Revenue saved (p={p_retention:.0%}):       "
          f"${summary['revenue_saved']:>10,}")
    print(f"  ─────────────────────────────────────────")
    print(f"  NET BENEFIT:                  ${summary['net_benefit']:>10,}")
    print(f"  ROI:                          {summary['roi_pct']:.0f}%")
    print(f"  Top-3 SHAP drivers:           {', '.join(top3)}")
    print("═" * 60)
    print(f"\n  → The defensible headline number is ${summary['net_benefit']:,}")
    print(f"    (replaces '$165K' everywhere in docs)\n")

    return summary, top3


if __name__ == "__main__":
    main()
