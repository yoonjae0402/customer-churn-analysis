# Data Dictionary

## Dataset Overview

**Source:** [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
**File:** `telco_churn.csv`
**Records:** 7,043 customers
**Target Variable:** `Churn` (Yes/No)

---

## Raw Features

### Customer Demographics

| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| `customerID` | String | Unique customer identifier | Alphanumeric ID |
| `gender` | Categorical | Customer gender | Male, Female |
| `SeniorCitizen` | Binary | Whether customer is 65+ | 0 (No), 1 (Yes) |
| `Partner` | Categorical | Has a partner | Yes, No |
| `Dependents` | Categorical | Has dependents | Yes, No |

### Account Information

| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| `tenure` | Numeric | Months with company | 0-72 |
| `Contract` | Categorical | Contract type | Month-to-month, One year, Two year |
| `PaperlessBilling` | Categorical | Uses paperless billing | Yes, No |
| `PaymentMethod` | Categorical | Payment method | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) |
| `MonthlyCharges` | Numeric | Monthly charges ($) | 18.25 - 118.75 |
| `TotalCharges` | Numeric | Total amount charged ($) | 18.80 - 8684.80 |

### Services

| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| `PhoneService` | Categorical | Has phone service | Yes, No |
| `MultipleLines` | Categorical | Has multiple lines | Yes, No, No phone service |
| `InternetService` | Categorical | Internet service type | DSL, Fiber optic, No |
| `OnlineSecurity` | Categorical | Has online security add-on | Yes, No, No internet service |
| `OnlineBackup` | Categorical | Has online backup add-on | Yes, No, No internet service |
| `DeviceProtection` | Categorical | Has device protection | Yes, No, No internet service |
| `TechSupport` | Categorical | Has tech support | Yes, No, No internet service |
| `StreamingTV` | Categorical | Has streaming TV | Yes, No, No internet service |
| `StreamingMovies` | Categorical | Has streaming movies | Yes, No, No internet service |

### Target Variable

| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| `Churn` | Binary | Customer churned | Yes, No |

---

## Engineered Features

Features created during preprocessing (`notebooks/02_feature_engineering.ipynb`):

### Tenure-Based Features

| Feature | Description | Logic |
|---------|-------------|-------|
| `is_new_customer` | Customer tenure < 6 months | `tenure < 6` |
| `is_long_term` | Customer tenure > 24 months | `tenure > 24` |
| `tenure_bins` | Binned tenure categories | Discretized into ranges |
| `tenure_group_0-12` | Tenure 0-12 months | One-hot encoded |
| `tenure_group_12-24` | Tenure 12-24 months | One-hot encoded |
| `tenure_group_24-48` | Tenure 24-48 months | One-hot encoded |
| `tenure_group_48+` | Tenure 48+ months | One-hot encoded |

### Charge-Based Features

| Feature | Description | Logic |
|---------|-------------|-------|
| `avg_monthly_spend` | Average spend per month | `TotalCharges / tenure` |
| `charge_per_tenure` | Charge efficiency metric | `MonthlyCharges / (tenure + 1)` |
| `price_increase` | Potential price increase flag | `MonthlyCharges > avg_monthly_spend` |
| `high_monthly_charge` | High spender flag | `MonthlyCharges > median` |

### Service Aggregation Features

| Feature | Description | Logic |
|---------|-------------|-------|
| `total_services` | Count of subscribed services | Sum of all service flags |
| `has_internet` | Has any internet service | `InternetService != 'No'` |
| `has_phone` | Has phone service | `PhoneService == 'Yes'` |
| `has_tech_support` | Has tech support | `TechSupport == 'Yes'` |
| `has_online_security` | Has online security | `OnlineSecurity == 'Yes'` |

### Contract/Payment Features

| Feature | Description | Logic |
|---------|-------------|-------|
| `is_month_to_month` | Month-to-month contract | `Contract == 'Month-to-month'` |
| `has_auto_payment` | Uses automatic payment | Bank transfer or Credit card |
| `is_paperless` | Uses paperless billing | `PaperlessBilling == 'Yes'` |

### Demographic Flags

| Feature | Description | Logic |
|---------|-------------|-------|
| `is_senior` | Senior citizen | `SeniorCitizen == 1` |
| `has_family` | Has partner or dependents | `Partner == 'Yes' OR Dependents == 'Yes'` |
| `is_single_no_deps` | Single without dependents | No partner and no dependents |

---

## Data Quality Notes

- **Missing Values:** `TotalCharges` has 11 missing values (new customers with 0 tenure)
- **Data Types:** `TotalCharges` stored as string in raw data, converted to float
- **Class Imbalance:** ~26.5% churn rate (1,869 Yes / 5,174 No)

---

## Feature Importance

Top 10 features by model importance:

1. `tenure` - Customer tenure length
2. `Contract_Month-to-month` - Month-to-month contract
3. `MonthlyCharges` - Monthly charge amount
4. `TotalCharges` - Total charges
5. `InternetService_Fiber optic` - Fiber optic internet
6. `OnlineSecurity_No` - No online security
7. `TechSupport_No` - No tech support
8. `is_new_customer` - New customer flag
9. `PaymentMethod_Electronic check` - Electronic check payment
10. `has_auto_payment` - Automatic payment method
