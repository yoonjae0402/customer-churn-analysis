"""
Business-case computation for the churn retention model.

All monetary defaults are documented with sources so every number can be
defended in an interview.  Pass explicit ltv / offer_cost arguments to
override them in the dashboard or scenario analysis.

LTV derivation (dataset-grounded):
    Dataset mean MonthlyCharges (ARPU) = $64.76.
    Test cohort covers tenure 0–6 months (recently-acquired customers).
    Expected remaining tenure ≈ population mean (32 months) − cohort midpoint
    (3 months) = 29 months.
    Lost revenue per churner = ARPU × remaining_tenure = $64.76 × 29 ≈ $1,878.
    Rounded to $2,000 (conservative; consistent with Bain & Company telecom
    industry benchmarks of $1,500–$3,000 per subscriber).
    Reference: Bain & Company, "Winning in Wireless" (2022);
               CTIA Wireless Industry Survey (2022) — avg subscriber revenue ~$65/mo.

Offer-cost derivation:
    One-month bill credit ($25) or equivalent free service upgrade.
    Standard telecom retention incentive; represents the marginal cost of one
    proactive contact, not the full CAC.
    Reference: McKinsey & Company, "Reducing Churn in Telecom" (2021).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_LTV: float = 2_000.0   # USD per churner — see module docstring
DEFAULT_OFFER_COST: float = 25.0  # USD per flagged customer — see module docstring
DEFAULT_P_RETENTION: float = 0.30  # conservative success rate for retention offers


# ---------------------------------------------------------------------------
# Core cost helpers
# ---------------------------------------------------------------------------

def cost_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    ltv: float = DEFAULT_LTV,
    offer_cost: float = DEFAULT_OFFER_COST,
) -> Dict[str, float]:
    """
    Compute expected campaign cost at a fixed decision threshold.

    Cost model:
      FN (missed churner)   → ltv lost (customer leaves, no offer sent)
      TP (caught churner)   → offer_cost spent (offer sent; revenue saved
                              depends on p_retention and is handled in
                              compute_decile_analysis / compute_roi_summary)
      FP (false alarm)      → offer_cost spent (offer wasted on a stayer)
      TN (correctly skipped)→ $0

    Total cost = FN × ltv + (TP + FP) × offer_cost
    Null cost  = all churners × ltv  (do-nothing baseline)
    Cost saved = null_cost − total_cost
    """
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    fn_cost = fn * ltv
    campaign_cost = (tp + fp) * offer_cost
    total_cost = fn_cost + campaign_cost
    null_cost = float(y_true.sum()) * ltv
    cost_saved = null_cost - total_cost

    n_pos = tp + fn
    n_pred_pos = tp + fp
    recall = tp / n_pos if n_pos > 0 else 0.0
    precision = tp / n_pred_pos if n_pred_pos > 0 else 0.0

    return {
        "threshold": float(threshold),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "flagged": tp + fp,
        "fn_cost": fn_cost,
        "campaign_cost": campaign_cost,
        "total_cost": total_cost,
        "null_cost": null_cost,
        "cost_saved": cost_saved,
        "recall": recall,
        "precision": precision,
    }


def build_cost_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ltv: float = DEFAULT_LTV,
    offer_cost: float = DEFAULT_OFFER_COST,
    n_points: int = 400,
) -> pd.DataFrame:
    """
    Compute cost metrics across a fine grid of thresholds [0.01, 0.99].
    Returns a DataFrame suitable for plotting or tabular inspection.
    """
    thresholds = np.linspace(0.01, 0.99, n_points)
    rows = [cost_at_threshold(y_true, y_prob, t, ltv, offer_cost)
            for t in thresholds]
    return pd.DataFrame(rows)


def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ltv: float = DEFAULT_LTV,
    offer_cost: float = DEFAULT_OFFER_COST,
) -> Tuple[float, Dict]:
    """
    Return the threshold that minimises total expected cost (FN×LTV + offers×cost).

    Scans 400 evenly-spaced values in [0.01, 0.99].  The cost curve is smooth
    so this grid is more than fine enough for practical use.
    """
    curve = build_cost_curve(y_true, y_prob, ltv, offer_cost)
    best_idx = int(curve["total_cost"].idxmin())
    row = curve.iloc[best_idx].to_dict()
    return float(row["threshold"]), row


# ---------------------------------------------------------------------------
# Decile analysis
# ---------------------------------------------------------------------------

def compute_decile_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ltv: float = DEFAULT_LTV,
    offer_cost: float = DEFAULT_OFFER_COST,
    p_retention: float = DEFAULT_P_RETENTION,
) -> pd.DataFrame:
    """
    Rank customers by predicted churn probability; compute per-decile ROI.

    ROI model per decile:
        revenue_saved  = churners × p_retention × ltv
        campaign_cost  = customers × offer_cost
        net_benefit    = revenue_saved − campaign_cost
        roi_pct        = net_benefit / campaign_cost × 100 %

    break_even_retention_rate:
        Minimum p_retention for this decile to be net-positive.
        break_even = offer_cost / (churn_rate × ltv)
        At break-even, revenue_saved = campaign_cost exactly.

    Args:
        y_true:      Binary labels (1 = churner)
        y_prob:      Predicted churn probabilities from the model
        ltv:         USD lost per unretained churner
        offer_cost:  USD cost of one retention offer / contact
        p_retention: Fraction of contacted churners who accept and stay
    """
    df = pd.DataFrame({"y_true": y_true.astype(int), "y_prob": y_prob})
    df = df.sort_values("y_prob", ascending=False).reset_index(drop=True)

    # Assign decile labels 1 (highest risk) … 10 (lowest risk)
    n = len(df)
    decile_size = n // 10
    labels = np.repeat(np.arange(1, 11), decile_size)
    remainder = n - len(labels)
    if remainder:
        labels = np.concatenate([labels, np.full(remainder, 10)])
    df["decile"] = labels

    rows = []
    cumulative_churners = 0
    total_churners = int(y_true.sum())

    for decile, grp in df.groupby("decile"):
        n_cust = len(grp)
        n_churn = int(grp["y_true"].sum())
        churn_rate = n_churn / n_cust if n_cust > 0 else 0.0

        cumulative_churners += n_churn
        cumulative_recall = cumulative_churners / total_churners if total_churners > 0 else 0.0

        expected_loss = n_churn * ltv
        camp_cost = n_cust * offer_cost
        rev_saved = n_churn * p_retention * ltv
        net = rev_saved - camp_cost
        roi = (net / camp_cost * 100) if camp_cost > 0 else 0.0
        break_even = (offer_cost / (churn_rate * ltv)) if (churn_rate * ltv) > 0 else float("inf")

        rows.append({
            "decile": int(decile),
            "customers": n_cust,
            "churners": n_churn,
            "churn_rate_pct": round(churn_rate * 100, 1),
            "avg_predicted_prob": round(float(grp["y_prob"].mean()), 3),
            "cumulative_recall_pct": round(cumulative_recall * 100, 1),
            "expected_loss_no_action": int(round(expected_loss)),
            "campaign_cost": int(round(camp_cost)),
            "revenue_saved": int(round(rev_saved)),
            "net_benefit": int(round(net)),
            "roi_pct": round(roi, 1),
            "break_even_retention_rate_pct": round(break_even * 100, 1),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def compute_roi_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ltv: float = DEFAULT_LTV,
    offer_cost: float = DEFAULT_OFFER_COST,
    p_retention: float = DEFAULT_P_RETENTION,
) -> Dict:
    """
    High-level ROI summary at the cost-optimal threshold.

    Returns a dict with every number needed for the one-page IMPACT summary.
    """
    opt_thresh, cost_row = find_cost_optimal_threshold(y_true, y_prob, ltv, offer_cost)

    tp = int(cost_row["tp"])
    fp = int(cost_row["fp"])
    fn = int(cost_row["fn"])
    flagged = tp + fp

    revenue_saved = tp * p_retention * ltv
    campaign_cost_total = flagged * offer_cost
    net_benefit = revenue_saved - campaign_cost_total
    null_cost = float(y_true.sum()) * ltv
    roi_pct = (net_benefit / campaign_cost_total * 100) if campaign_cost_total > 0 else 0.0

    # Per-customer cost reduction (normalised to cohort size)
    n = len(y_true)
    cost_saved_per_customer = cost_row["cost_saved"] / n

    return {
        "ltv": ltv,
        "offer_cost": offer_cost,
        "p_retention": p_retention,
        "optimal_threshold": opt_thresh,
        "recall": cost_row["recall"],
        "precision": cost_row["precision"],
        "tp": tp, "fp": fp, "fn": fn,
        "flagged_customers": flagged,
        "null_cost": int(null_cost),
        "fn_cost_at_threshold": int(cost_row["fn_cost"]),
        "campaign_cost_total": int(campaign_cost_total),
        "revenue_saved": int(round(revenue_saved)),
        "net_benefit": int(round(net_benefit)),
        "roi_pct": round(roi_pct, 1),
        "cost_saved_per_customer": round(cost_saved_per_customer, 2),
        "cohort_size": n,
        "cohort_churners": int(y_true.sum()),
        "cohort_churn_rate": round(float(y_true.mean()), 4),
    }
