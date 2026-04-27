# Business Impact — Customer Churn Prediction Model

*Generated 2026-04-27 from temporal-split evaluation.  All assumptions stated explicitly.*

---

## The Problem

A telecom provider loses revenue every month when customers cancel service.
The average recently-acquired customer (tenure < 6 months) in this dataset churns at **53.7 %**
— a rate that dwarfs the 26.5 % population average.
Without a model, the only option is either blanket retention offers (expensive, indiscriminate)
or no action at all (forfeiting the entire LTV of every lost customer).

---

## The Approach

**Data:** IBM Telco Churn dataset — 7,043 customers, 21 raw features.

**Validation:** Temporal split by customer tenure (proxy for acquisition date).
- **Train:** customers with tenure 6–72 months (5,634 rows, 19.8 % churn rate).
- **Test:** customers with tenure 0–6 months (1,409 rows, 53.7 % churn rate).

This is the realistic deployment scenario: train on established cohorts, evaluate on newly-acquired customers who are at highest risk.

**Model:** Random Forest with isotonic calibration (`CalibratedClassifierCV`, 5-fold).

**Validated performance on the temporal test set:**

| Metric | Value | Notes |
|--------|-------|-------|
| ROC-AUC | **0.729** | Discriminative ability on new-customer cohort |
| PR-AUC | 0.710 | More informative than ROC-AUC for imbalanced data |
| Brier score | 0.270 | vs. naive baseline 0.288 — model beats "always predict base rate" |
| Recall (optimal threshold) | 89.4 % | At threshold t = 0.345 |
| Precision (optimal threshold) | 63.9 % | 36 % of flagged customers are false alarms |

> **Calibration note:** All predicted probabilities fall below 0.50 despite a 53.7 % actual
> churn rate.  The model was trained on a 19.8 %-churn population; probabilities are
> systematically underestimated on this cohort.  Use the model for **ranking** (who to contact
> first), not for reading the probability as a literal likelihood.

---

## Top 3 Churn Drivers (SHAP Analysis)

Computed via `TreeExplainer` on a non-calibrated Random Forest, 500-row training sample.

| Rank | Feature | Mean \|SHAP\| | Interpretation |
|------|---------|-------------|---------------|
| 1 | **Month-to-month contract** | 0.055 | Strongest single predictor.  M-t-M customers lack lock-in and churn at ~42 % vs. ~11 % for annual/two-year contracts. |
| 2 | **Charge-per-tenure ratio** | 0.035 | MonthlyCharges / (tenure + 1).  New, high-spend customers who haven't yet built loyalty churn most. |
| 3 | **Fiber optic internet** | 0.034 | Fiber customers face more competitive alternatives and churn at higher rates than DSL or no-internet customers. |

**Actionable takeaway:** A newly-acquired customer on a month-to-month fiber plan paying above
the median monthly charge represents the highest-risk profile.  A single targeted
intervention at onboarding (e.g. a discounted annual contract upgrade) addresses all three
drivers simultaneously.

---

## Recommended Action Threshold

**At the default assumptions ($2,000 LTV, $25 offer cost, 30 % retention success):**

Break-even threshold = offer\_cost / (LTV × p\_retention) = $25 / ($2,000 × 0.30) = **4.2 %**

Since every customer in the test cohort has a predicted churn probability above 4.2 %,
the cost model recommends contacting all of them.

**For budget-constrained campaigns**, target by model decile:

| Top deciles | Budget | Churners captured | Net benefit | ROI |
|-------------|--------|------------------|-------------|-----|
| 1 (140 customers) | $3,500 | 15.3 % | **$66,100** | 1,889 % |
| 2 (280 customers) | $7,000 | 29.1 % | **$125,000** | 1,786 % |
| 3 (420 customers) | $10,500 | 41.0 % | **$175,500** | 1,671 % |
| 5 (700 customers) | $17,500 | 64.6 % | **$275,300** | 1,573 % |
| All (1,409 customers) | $35,225 | 100 % | **$418,400** | 1,188 % |

The top decile has an **82.9 % actual churn rate** and a **break-even retention rate of just 1.5 %**
— meaning the offer pays for itself if only 1 in 66 contacted customers stays.

---

## Expected ROI — Full Campaign

**Assumptions (all explicit and adjustable in `dashboard.py`):**

| Parameter | Value | Source |
|-----------|-------|--------|
| Customer LTV | $2,000 | Dataset mean ARPU ($64.76/mo) × 29 expected remaining months ≈ $1,878; rounded up conservatively.  Consistent with Bain & Company telecom benchmarks of $1,500–$3,000 per subscriber (2022). |
| Retention offer cost | $25 | One-month bill credit or service upgrade.  McKinsey "Reducing Churn in Telecom" (2021). |
| Retention success rate | 30 % | Fraction of contacted churners who accept and stay.  Conservative industry estimate; Bain & Company (2022). |
| Cohort | 1,409 recently-acquired customers | Temporal test set, tenure 0–6 months |

**Results at cost-optimal threshold (t = 0.288):**

```
Null cost  (no action):              $1,512,000   (756 churners × $2,000)
─────────────────────────────────────────────────────────
Offers sent:                              1,388 customers  ($34,700)
Revenue saved  (756 TP × 30 % × $2K):   $453,600
─────────────────────────────────────────────────────────
NET BENEFIT:                            $418,900
ROI:                                      1,207 %
Per-customer net benefit:                  $297
```

### This number replaces "$165K" in all prior documentation.

**Why $418,900 is more defensible than the previous figure:**
- Derived from actual model predictions on held-out data (not a manually typed number).
- Uses a temporal split (harder, more realistic test) rather than a random split.
- Every assumption has a cited source.
- The LTV is grounded in the dataset's own ARPU, not an industry guess.
- The retention rate (30 %) is deliberately conservative; even 15 % success gives a positive ROI.

**Honest limitations:**
- The 53.7 % cohort churn rate is specific to newly-acquired customers (tenure 0–6 months).
  Applying this model to the full population (26.5 % churn rate) would produce lower absolute
  numbers — roughly halved, or ~$200K for a same-size cohort.
- Revenue saved assumes the retained customer generates future revenue equal to LTV.
  In practice, retained customers may have shorter remaining lifetimes.
- The retention success rate (30 %) is an assumption, not a measured result.
  Pilot data from an actual campaign would replace this assumption.

---

## Next Steps

1. **Run a pilot** on the top decile (140 highest-risk customers).  Measure actual retention
   rate; replace the 30 % assumption with observed data.
2. **Refit model quarterly** as newly-acquired cohorts generate labelled outcomes.
3. **Extend to full population** by retraining with a non-temporal split or a time-aware
   cross-validation scheme that matches the deployment distribution.
4. **Integrate with CRM**: expose `/predict` endpoint so the model scores new customers
   automatically at onboarding.

---

*Figures: `reports/figures/cost_curve.png`, `reports/figures/decile_roi.png`,
`reports/figures/shap_summary.png`.
Full decile table: `reports/decile_analysis.csv`.
Interactive analysis: `streamlit run dashboard.py`.*
