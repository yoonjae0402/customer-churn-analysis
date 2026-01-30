# Methodology

## Overview

This document describes the technical approach used for predicting customer churn, including data preprocessing, feature engineering, model selection, and evaluation.

---

## 1. Data Exploration

**Notebook:** `notebooks/01_exploration.ipynb`

### Key Findings

- **Dataset Size:** 7,043 customers with 21 features
- **Churn Rate:** 26.5% (imbalanced classes)
- **Missing Data:** 11 records with missing `TotalCharges` (new customers)

### Initial Observations

- Month-to-month contracts have significantly higher churn (~42%)
- Fiber optic customers show higher churn than DSL
- Customers without tech support or online security churn more
- Short tenure (<12 months) correlates strongly with churn

---

## 2. Feature Engineering

**Notebook:** `notebooks/02_feature_engineering.ipynb`

### Preprocessing Steps

1. **Handle Missing Values**
   - Fill missing `TotalCharges` with `MonthlyCharges * tenure`
   - Convert `TotalCharges` from string to float

2. **Create Derived Features**
   - Tenure groupings (0-12, 12-24, 24-48, 48+ months)
   - Average monthly spend calculation
   - Service aggregation counts
   - Binary flags for contract type, payment method, demographics

3. **Encoding**
   - One-hot encoding for categorical variables
   - Binary encoding for Yes/No features

4. **Scaling**
   - StandardScaler for numerical features (tenure, charges)
   - Scaler saved to `app/models/scaler.pkl`

### Final Feature Set

62 features total after engineering and encoding (see `data_dictionary.md`)

---

## 3. Model Selection

**Notebook:** `notebooks/03_modeling.ipynb`

### Models Evaluated

| Model | Description | Rationale |
|-------|-------------|-----------|
| Logistic Regression | Linear classifier | Baseline, interpretable |
| Decision Tree | Tree-based classifier | Captures non-linear relationships |
| Random Forest | Ensemble of trees | Reduces overfitting |
| Gradient Boosting | Sequential ensemble | High accuracy |

### Train/Test Split

- **Training Set:** 80% (5,634 samples)
- **Test Set:** 20% (1,409 samples)
- **Stratification:** Maintained class balance in splits
- **Random State:** 42 (for reproducibility)

### Hyperparameter Tuning

Grid search with 5-fold cross-validation for:
- **Random Forest:** `n_estimators`, `max_depth`, `min_samples_split`
- **Gradient Boosting:** `n_estimators`, `learning_rate`, `max_depth`

---

## 4. Model Evaluation

**Notebook:** `notebooks/04_evaluation.ipynb`

### Metrics Used

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| Accuracy | Overall correctness | General performance |
| Precision | True positives / Predicted positives | Cost of false alarms |
| Recall | True positives / Actual positives | Catching churners |
| F1-Score | Harmonic mean of P & R | Balance of precision/recall |
| ROC-AUC | Area under ROC curve | Model discrimination ability |

### Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.808 | 0.676 | 0.529 | 0.594 | **0.847** |
| Gradient Boosting (Tuned) | 0.800 | 0.658 | 0.513 | 0.577 | 0.843 |
| Random Forest (Tuned) | 0.795 | 0.638 | 0.524 | 0.576 | 0.839 |
| Gradient Boosting | 0.796 | 0.648 | 0.503 | 0.566 | 0.843 |
| Random Forest | 0.784 | 0.616 | 0.497 | 0.550 | 0.819 |
| Decision Tree | 0.737 | 0.504 | 0.519 | 0.511 | 0.667 |

### Model Selection Rationale

**Logistic Regression** was selected as the production model because:

1. **Best ROC-AUC (0.847):** Superior discrimination between churners and non-churners
2. **Interpretability:** Coefficients provide clear feature importance
3. **Efficiency:** Fast inference time for real-time predictions
4. **Simplicity:** Fewer hyperparameters, easier to maintain
5. **Probability Calibration:** Well-calibrated probability outputs

---

## 5. Production Integration

### Gemini AI Enhancement

The ML model provides churn probability, while Gemini AI generates personalized retention offers based on:
- Customer profile (services, contract, demographics)
- Churn risk level (low/medium/high)
- Specific risk factors identified

### API Design

- **FastAPI** for async request handling
- **Pydantic** for input validation
- **Singleton pattern** for model loading (efficiency)

---

## Reproducibility

To reproduce results:

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Run notebooks in order
jupyter notebook notebooks/01_exploration.ipynb
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_modeling.ipynb
jupyter notebook notebooks/04_evaluation.ipynb
```

**Note:** Results may vary slightly (2-3%) due to random state variations in cross-validation and train/test splits.
