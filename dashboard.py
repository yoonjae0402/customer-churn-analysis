"""
Churn Retention ROI Simulator — Streamlit dashboard.

Usage:
    streamlit run dashboard.py

Sliders let you adjust LTV, offer cost, and retention success rate.
All business-case numbers update instantly.

Requires: streamlit, matplotlib, joblib (in requirements.txt)
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
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

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Churn Retention ROI Simulator",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Customer Churn — Retention ROI Simulator")
st.caption(
    "All numbers are computed from the temporal-split test set "
    "(1,409 recently-acquired customers, tenure 0–6 months, 53.7 % actual churn rate).  "
    "Adjust assumptions in the sidebar; the analysis updates instantly."
)

# ---------------------------------------------------------------------------
# Sidebar — cost assumptions
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Cost Assumptions")

    ltv = st.slider(
        "Customer LTV ($)",
        min_value=500, max_value=5_000, value=int(DEFAULT_LTV), step=100,
        help="Expected revenue lost per unretained churner.  "
             "Derived from dataset ARPU ($64.76) × ~29 remaining months ≈ $1,878; "
             "rounded to $2,000 (conservative).  "
             "Source: Bain & Company telecom benchmarks (2022).",
    )

    offer_cost = st.slider(
        "Retention Offer Cost ($)",
        min_value=5, max_value=100, value=int(DEFAULT_OFFER_COST), step=5,
        help="Cost of one proactive retention contact (e.g. one-month bill credit).  "
             "Source: McKinsey 'Reducing Churn in Telecom' (2021).",
    )

    p_retention = st.slider(
        "Retention Success Rate (%)",
        min_value=5, max_value=70, value=int(DEFAULT_P_RETENTION * 100), step=5,
        help="Fraction of contacted churners who accept the offer and stay.  "
             "30% is a conservative industry estimate.  "
             "Source: Bain & Company (2022).",
    ) / 100.0

    st.markdown("---")
    st.caption(
        "**Break-even threshold:**  "
        f"Flag a customer if P(churn) > offer\\_cost / (LTV × p_retention) "
        f"= **{offer_cost / (ltv * p_retention):.1%}**"
    )

# ---------------------------------------------------------------------------
# Load data (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_predictions() -> tuple[np.ndarray, np.ndarray]:
    df = load_csv_data(ROOT / "data" / "raw" / "telco_churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    _, X_test, _, y_test = temporal_split(df, "Churn")
    model = joblib.load(ROOT / "models" / "random_forest_pipeline.pkl")
    y_prob = model.predict_proba(X_test)[:, 1]
    return y_test.values.astype(int), y_prob


y_true, y_prob = load_predictions()

# ---------------------------------------------------------------------------
# Compute business metrics
# ---------------------------------------------------------------------------

summary = compute_roi_summary(y_true, y_prob, ltv, offer_cost, p_retention)
decile_df = compute_decile_analysis(y_true, y_prob, ltv, offer_cost, p_retention)
curve_df = build_cost_curve(y_true, y_prob, ltv, offer_cost)

opt_thresh = summary["optimal_threshold"]
break_even_p = offer_cost / (ltv * p_retention)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

st.subheader("Campaign Summary at Cost-Optimal Threshold")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Null cost (do nothing)", f"${summary['null_cost']:,}")
col2.metric("Optimal threshold", f"{opt_thresh:.3f}")
col3.metric("Recall / Precision", f"{summary['recall']:.0%} / {summary['precision']:.0%}")
col4.metric("Net benefit", f"${summary['net_benefit']:,}")
col5.metric("ROI", f"{summary['roi_pct']:.0f}%")

if break_even_p >= y_prob.min():
    st.info(
        f"ℹ️  At these cost parameters the break-even threshold ({break_even_p:.1%}) "
        f"is {'above' if break_even_p > y_prob.min() else 'at'} the model's minimum predicted "
        f"probability ({y_prob.min():.1%}).  "
        "Contacting every customer in this high-churn cohort is cost-optimal.  "
        "**The model's value is prioritisation**: which customers to contact first "
        "when the retention budget is limited."
    )

# ---------------------------------------------------------------------------
# Cost curve
# ---------------------------------------------------------------------------

st.subheader("Expected Cost vs. Decision Threshold")
st.caption(
    "Total cost = FN × LTV + flagged customers × offer cost.  "
    "The optimal threshold minimises this sum."
)

fig_cost, ax = plt.subplots(figsize=(10, 4))
ax.plot(curve_df["threshold"], curve_df["fn_cost"] / 1_000,
        color="#d62728", lw=1.8, label="FN cost  (missed churners × LTV)")
ax.plot(curve_df["threshold"], curve_df["campaign_cost"] / 1_000,
        color="#1f77b4", lw=1.8, label="Campaign cost  (flagged × offer cost)")
ax.plot(curve_df["threshold"], curve_df["total_cost"] / 1_000,
        color="#2ca02c", lw=2.2, label="Total cost")
ax.axvline(opt_thresh, color="black", lw=1.4, linestyle="--",
           label=f"Optimal threshold  t = {opt_thresh:.3f}")
ax.set_xlabel("Decision Threshold", fontsize=11)
ax.set_ylabel("Expected Cost  ($K)", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
fig_cost.tight_layout()
st.pyplot(fig_cost)
plt.close(fig_cost)

# ---------------------------------------------------------------------------
# Decile analysis
# ---------------------------------------------------------------------------

st.subheader("Decile Analysis — Prioritise Your Retention Budget")
st.caption(
    f"Customers ranked by predicted churn probability (decile 1 = highest risk).  "
    f"p_retention = {p_retention:.0%}.  "
    f"Break-even retention rate = offer\\_cost / (churn\\_rate × LTV)."
)

display_df = decile_df.copy()
display_df["churn_rate_pct"] = display_df["churn_rate_pct"].map("{:.1f}%".format)
display_df["avg_predicted_prob"] = display_df["avg_predicted_prob"].map("{:.3f}".format)
display_df["cumulative_recall_pct"] = display_df["cumulative_recall_pct"].map("{:.1f}%".format)
display_df["expected_loss_no_action"] = display_df["expected_loss_no_action"].map("${:,}".format)
display_df["campaign_cost"] = display_df["campaign_cost"].map("${:,}".format)
display_df["revenue_saved"] = display_df["revenue_saved"].map("${:,}".format)
display_df["net_benefit"] = display_df["net_benefit"].map("${:,}".format)
display_df["roi_pct"] = display_df["roi_pct"].map("{:.0f}%".format)
display_df["break_even_retention_rate_pct"] = display_df["break_even_retention_rate_pct"].map("{:.1f}%".format)
st.dataframe(display_df, use_container_width=True, hide_index=True)

# Decile ROI bar chart
fig_bar, ax2 = plt.subplots(figsize=(10, 4))
colors = ["#2ca02c" if v >= 0 else "#d62728" for v in decile_df["net_benefit"]]
ax2.bar(decile_df["decile"], decile_df["net_benefit"] / 1_000, color=colors,
        edgecolor="white", linewidth=0.5)
ax2.axhline(0, color="black", lw=0.8)
ax2.set_xlabel("Decile  (1 = highest predicted risk)", fontsize=11)
ax2.set_ylabel("Net Benefit  ($K)", fontsize=11)
ax2.set_title(f"Net Benefit per Decile  (p_retention = {p_retention:.0%})", fontsize=11)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
ax2.set_xticks(decile_df["decile"])
fig_bar.tight_layout()
st.pyplot(fig_bar)
plt.close(fig_bar)

# ---------------------------------------------------------------------------
# Cumulative ROI for budget scenarios
# ---------------------------------------------------------------------------

st.subheader("Cumulative ROI by Budget")
st.caption("How much net benefit do you capture as you expand the campaign to more deciles?")

decile_df["cum_net_benefit"] = decile_df["net_benefit"].cumsum()
decile_df["cum_campaign_cost"] = decile_df["campaign_cost"].cumsum()
decile_df["cum_roi_pct"] = (
    decile_df["cum_net_benefit"] / decile_df["cum_campaign_cost"] * 100
).round(1)

cum_display = decile_df[
    ["decile", "cumulative_recall_pct", "cum_campaign_cost",
     "cum_net_benefit", "cum_roi_pct"]
].copy()
cum_display.columns = [
    "Top N deciles", "Churners captured (%)",
    "Cumulative budget ($)", "Cumulative net benefit ($)", "Cumulative ROI (%)"
]
cum_display["Cumulative budget ($)"] = cum_display["Cumulative budget ($)"].map("${:,}".format)
cum_display["Cumulative net benefit ($)"] = cum_display["Cumulative net benefit ($)"].map("${:,}".format)
cum_display["Churners captured (%)"] = cum_display["Churners captured (%)"].map("{:.1f}%".format)
cum_display["Cumulative ROI (%)"] = cum_display["Cumulative ROI (%)"].map("{:.0f}%".format)
st.dataframe(cum_display, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Key assumptions box
# ---------------------------------------------------------------------------

with st.expander("Assumptions & Limitations"):
    st.markdown(f"""
**Monetary assumptions:**
- **LTV = ${ltv:,}**: derived from dataset mean ARPU ($64.76) × 29 expected remaining months.
  Consistent with Bain & Company telecom benchmarks ($1,500–$3,000 per subscriber, 2022).
- **Offer cost = ${offer_cost}**: one-month bill credit or equivalent service upgrade.
  McKinsey "Reducing Churn in Telecom" (2021).
- **Retention success rate = {p_retention:.0%}**: conservative industry estimate.
  Bain & Company (2022).

**Data / model limitations:**
- Numbers are for the **temporal test cohort**: 1,409 customers with tenure 0–6 months.
  This cohort has a 53.7 % churn rate — higher than the overall 26.5 % population rate.
- The model was trained on customers with tenure 6–72 months (lower churn) and generalises
  to newer customers.  ROC-AUC on this cohort = 0.729 (vs. 0.839 on a shuffled split).
- Calibration drift: predicted probabilities underestimate true churn rate (all < 0.50 vs.
  53.7 % actual).  Use the model for **ranking**, not for reading the probability as literal.
- Retention success rate is assumed constant across deciles; in practice, highly-at-risk
  customers may be harder to retain.
- Revenue saved counts only future monthly revenue; it excludes cost of service, CAC, etc.
""")
