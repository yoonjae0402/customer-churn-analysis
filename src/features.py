
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to generate derived features for customer churn prediction.
    """
    def __init__(self, monthly_median: Optional[float] = None):
        self.monthly_median = monthly_median
        self.service_cols = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        self.auto_payment_methods = ["Bank transfer (automatic)", "Credit card (automatic)"]

    def fit(self, X, y=None):
        """
        Learn any parameters needed for feature engineering.
        """
        if self.monthly_median is None:
            self.monthly_median = X["MonthlyCharges"].median()
        return self

    def transform(self, X):
        """
        Apply feature engineering transformations.
        """
        df = X.copy()
        
        # 1. Tenure Features
        df["is_new_customer"] = (df["tenure"] <= 12).astype(int)
        df["is_long_term"] = (df["tenure"] > 24).astype(int)
        df["tenure_bins"] = pd.cut(df["tenure"], bins=8, labels=False, right=False).fillna(0).astype(int)
        
        # 2. Charge Features
        # Handle missing TotalCharges if necessary (though imputation usually happens before)
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce').fillna(0)
            
        df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"].replace(0, np.nan) + 1)
        df["avg_monthly_spend"] = df["avg_monthly_spend"].fillna(0)
        
        df["charge_per_tenure"] = df["MonthlyCharges"] / (df["tenure"].replace(0, np.nan) + 1)
        df["charge_per_tenure"] = df["charge_per_tenure"].fillna(0)
        
        df["price_increase"] = (df["MonthlyCharges"] > df["avg_monthly_spend"]).astype(int)
        df["high_monthly_charge"] = (df["MonthlyCharges"] > self.monthly_median).astype(int)
        
        # 3. Service Features
        df["total_services"] = 0
        for col in self.service_cols:
            if col in df.columns:
                df["total_services"] += (df[col] == "Yes").astype(int)
                
        df["has_internet"] = (df["InternetService"] != "No").astype(int)
        df["has_phone"] = (df["PhoneService"] == "Yes").astype(int)
        df["has_tech_support"] = (df["TechSupport"] == "Yes").astype(int)
        df["has_online_security"] = (df["OnlineSecurity"] == "Yes").astype(int)
        
        # 4. Contract Logic
        df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
        df["has_auto_payment"] = df["PaymentMethod"].isin(self.auto_payment_methods).astype(int)
        df["is_paperless"] = (df["PaperlessBilling"] == "Yes").astype(int)
        
        # 5. Demographics
        df["is_senior"] = df["SeniorCitizen"]
        has_partner = (df["Partner"] == "Yes").astype(int)
        has_dependents = (df["Dependents"] == "Yes").astype(int)
        df["has_family"] = ((has_partner + has_dependents) > 0).astype(int)
        df["is_single_no_deps"] = ((df["Partner"] == "No") & (df["Dependents"] == "No")).astype(int)
        
        return df

class CategoricalCleaner(BaseEstimator, TransformerMixin):
    """
    Standardizes categorical values before encoding.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Ensure consistent string types
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            df[col] = df[col].astype(str)
        
        # Map binary columns to 0/1 if strictly binary strings 'Yes'/'No'
        # Note: Scikit-learn's OneHotEncoder can handle strings, but mapping manually preserves semantics
        binary_map = {'Yes': 1, 'No': 0, 'True': 1, 'False': 0}
        
        # Specific binary columns that might need mapping
        for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
            if col in df.columns:
                 df[col] = df[col].map(binary_map).fillna(0) # Fallback to 0
        
        if "gender" in df.columns:
             df["gender"] = df["gender"].map({'Male': 1, 'Female': 0}).fillna(0)
             
        return df
