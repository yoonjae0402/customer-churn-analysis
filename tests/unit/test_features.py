"""
Unit tests for feature engineering transformers.

These tests exercise the logic inside FeatureEngineer and CategoricalCleaner
directly — without running the full sklearn pipeline — so a wrong formula or
boundary condition fails here rather than silently corrupting model inputs.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import CategoricalCleaner, FeatureEngineer

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def base_row() -> dict:
    """Minimal valid customer row with known values for assertion."""
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 6,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 420.0,
    }


@pytest.fixture
def single_df(base_row) -> pd.DataFrame:
    return pd.DataFrame([base_row])


# ---------------------------------------------------------------------------
# FeatureEngineer — fit behaviour
# ---------------------------------------------------------------------------


class TestFeatureEngineerFit:
    def test_fit_stores_monthly_median(self, single_df):
        fe = FeatureEngineer()
        fe.fit(single_df)
        assert fe.monthly_median == 70.0

    def test_fit_respects_provided_median(self, single_df):
        fe = FeatureEngineer(monthly_median=50.0)
        fe.fit(single_df)
        assert fe.monthly_median == 50.0  # not overwritten

    def test_fit_returns_self(self, single_df):
        fe = FeatureEngineer()
        result = fe.fit(single_df)
        assert result is fe


# ---------------------------------------------------------------------------
# FeatureEngineer — tenure features
# ---------------------------------------------------------------------------


class TestTenureFeatures:
    @pytest.mark.parametrize(
        "tenure,expected_new,expected_long",
        [
            (0, 1, 0),
            (12, 1, 0),  # boundary: ≤12 → new
            (13, 0, 0),  # just above new threshold
            (24, 0, 0),  # boundary: ≤24 → not long-term
            (25, 0, 1),  # just above long-term threshold
            (72, 0, 1),
        ],
    )
    def test_is_new_and_long_term_boundaries(
        self, base_row, tenure, expected_new, expected_long
    ):
        base_row["tenure"] = tenure
        df = pd.DataFrame([base_row])
        fe = FeatureEngineer(monthly_median=70.0)
        fe.fit(df)
        out = fe.transform(df)
        assert out["is_new_customer"].iloc[0] == expected_new
        assert out["is_long_term"].iloc[0] == expected_long

    def test_tenure_bins_produces_integer(self, single_df):
        fe = FeatureEngineer(monthly_median=70.0)
        fe.fit(single_df)
        out = fe.transform(single_df)
        assert out["tenure_bins"].dtype in (np.int32, np.int64, int)
        assert 0 <= out["tenure_bins"].iloc[0] <= 7


# ---------------------------------------------------------------------------
# FeatureEngineer — charge features
# ---------------------------------------------------------------------------


class TestChargeFeatures:
    def test_charge_per_tenure_formula(self, base_row):
        """charge_per_tenure = MonthlyCharges / (tenure + 1)"""
        base_row["tenure"] = 6
        base_row["MonthlyCharges"] = 70.0
        df = pd.DataFrame([base_row])
        fe = FeatureEngineer(monthly_median=70.0)
        fe.fit(df)
        out = fe.transform(df)
        expected = 70.0 / (6 + 1)
        assert abs(out["charge_per_tenure"].iloc[0] - expected) < 1e-9

    def test_charge_per_tenure_zero_tenure(self, base_row):
        """tenure=0 → denominator is 0+1=1, should not produce NaN or inf."""
        base_row["tenure"] = 0
        base_row["TotalCharges"] = 0.0
        df = pd.DataFrame([base_row])
        fe = FeatureEngineer(monthly_median=70.0)
        fe.fit(df)
        out = fe.transform(df)
        assert np.isfinite(out["charge_per_tenure"].iloc[0])
        assert np.isfinite(out["avg_monthly_spend"].iloc[0])

    def test_high_monthly_charge_flag(self, base_row):
        """high_monthly_charge = 1 when MonthlyCharges > median."""
        base_row["MonthlyCharges"] = 80.0
        df = pd.DataFrame([base_row])
        fe = FeatureEngineer(monthly_median=70.0)
        fe.fit(df)
        out = fe.transform(df)
        assert out["high_monthly_charge"].iloc[0] == 1

        base_row["MonthlyCharges"] = 60.0
        df2 = pd.DataFrame([base_row])
        out2 = fe.transform(df2)
        assert out2["high_monthly_charge"].iloc[0] == 0

    def test_nan_total_charges_filled(self, base_row):
        """NaN TotalCharges must not propagate into derived features."""
        base_row["TotalCharges"] = float("nan")
        df = pd.DataFrame([base_row])
        fe = FeatureEngineer(monthly_median=70.0)
        fe.fit(df)
        out = fe.transform(df)
        assert np.isfinite(out["avg_monthly_spend"].iloc[0])
        assert np.isfinite(out["charge_per_tenure"].iloc[0])


# ---------------------------------------------------------------------------
# FeatureEngineer — service features
# ---------------------------------------------------------------------------


class TestServiceFeatures:
    def test_total_services_count(self, base_row):
        """total_services counts 'Yes' across the 6 add-on service columns."""
        # base_row has OnlineBackup=Yes, StreamingTV=Yes → 2 services
        df = pd.DataFrame([base_row])
        fe = FeatureEngineer(monthly_median=70.0)
        fe.fit(df)
        out = fe.transform(df)
        assert out["total_services"].iloc[0] == 2

    def test_total_services_all_yes(self, base_row):
        for col in [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]:
            base_row[col] = "Yes"
        df = pd.DataFrame([base_row])
        fe = FeatureEngineer(monthly_median=70.0)
        fe.fit(df)
        out = fe.transform(df)
        assert out["total_services"].iloc[0] == 6

    def test_total_services_all_no(self, base_row):
        for col in [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]:
            base_row[col] = "No"
        df = pd.DataFrame([base_row])
        fe = FeatureEngineer(monthly_median=70.0)
        fe.fit(df)
        out = fe.transform(df)
        assert out["total_services"].iloc[0] == 0

    def test_has_internet_fiber_and_dsl(self, base_row):
        """has_internet = 1 for DSL and Fiber optic, 0 for No."""
        for service, expected in [("DSL", 1), ("Fiber optic", 1), ("No", 0)]:
            base_row["InternetService"] = service
            df = pd.DataFrame([base_row])
            fe = FeatureEngineer(monthly_median=70.0)
            fe.fit(df)
            out = fe.transform(df)
            assert out["has_internet"].iloc[0] == expected, f"InternetService={service}"


# ---------------------------------------------------------------------------
# FeatureEngineer — contract / payment features
# ---------------------------------------------------------------------------


class TestContractFeatures:
    @pytest.mark.parametrize(
        "contract,expected",
        [
            ("Month-to-month", 1),
            ("One year", 0),
            ("Two year", 0),
        ],
    )
    def test_is_month_to_month(self, base_row, contract, expected):
        base_row["Contract"] = contract
        df = pd.DataFrame([base_row])
        fe = FeatureEngineer(monthly_median=70.0)
        fe.fit(df)
        out = fe.transform(df)
        assert out["is_month_to_month"].iloc[0] == expected

    @pytest.mark.parametrize(
        "method,expected",
        [
            ("Bank transfer (automatic)", 1),
            ("Credit card (automatic)", 1),
            ("Electronic check", 0),
            ("Mailed check", 0),
        ],
    )
    def test_has_auto_payment(self, base_row, method, expected):
        base_row["PaymentMethod"] = method
        df = pd.DataFrame([base_row])
        fe = FeatureEngineer(monthly_median=70.0)
        fe.fit(df)
        out = fe.transform(df)
        assert out["has_auto_payment"].iloc[0] == expected


# ---------------------------------------------------------------------------
# FeatureEngineer — demographics
# ---------------------------------------------------------------------------


class TestDemographicFeatures:
    def test_has_family_partner_only(self, base_row):
        base_row["Partner"] = "Yes"
        base_row["Dependents"] = "No"
        df = pd.DataFrame([base_row])
        fe = FeatureEngineer(monthly_median=70.0)
        fe.fit(df)
        out = fe.transform(df)
        assert out["has_family"].iloc[0] == 1

    def test_has_family_neither(self, base_row):
        base_row["Partner"] = "No"
        base_row["Dependents"] = "No"
        df = pd.DataFrame([base_row])
        fe = FeatureEngineer(monthly_median=70.0)
        fe.fit(df)
        out = fe.transform(df)
        assert out["has_family"].iloc[0] == 0
        assert out["is_single_no_deps"].iloc[0] == 1

    def test_has_family_both(self, base_row):
        base_row["Partner"] = "Yes"
        base_row["Dependents"] = "Yes"
        df = pd.DataFrame([base_row])
        fe = FeatureEngineer(monthly_median=70.0)
        fe.fit(df)
        out = fe.transform(df)
        assert out["has_family"].iloc[0] == 1
        assert out["is_single_no_deps"].iloc[0] == 0


# ---------------------------------------------------------------------------
# FeatureEngineer — output completeness
# ---------------------------------------------------------------------------


class TestOutputCompleteness:
    EXPECTED_DERIVED = [
        "is_new_customer",
        "is_long_term",
        "tenure_bins",
        "avg_monthly_spend",
        "charge_per_tenure",
        "price_increase",
        "high_monthly_charge",
        "total_services",
        "has_internet",
        "has_phone",
        "has_tech_support",
        "has_online_security",
        "is_month_to_month",
        "has_auto_payment",
        "is_paperless",
        "is_senior",
        "has_family",
        "is_single_no_deps",
    ]

    def test_all_derived_columns_present(self, single_df):
        fe = FeatureEngineer(monthly_median=70.0)
        fe.fit(single_df)
        out = fe.transform(single_df)
        for col in self.EXPECTED_DERIVED:
            assert col in out.columns, f"Missing derived column: {col}"

    def test_no_nans_in_derived_columns(self, single_df):
        fe = FeatureEngineer(monthly_median=70.0)
        fe.fit(single_df)
        out = fe.transform(single_df)
        for col in self.EXPECTED_DERIVED:
            assert not out[col].isna().any(), f"NaN in derived column: {col}"

    def test_transform_does_not_mutate_input(self, single_df):
        original_cols = set(single_df.columns)
        fe = FeatureEngineer(monthly_median=70.0)
        fe.fit(single_df)
        fe.transform(single_df)
        assert set(single_df.columns) == original_cols


# ---------------------------------------------------------------------------
# CategoricalCleaner
# ---------------------------------------------------------------------------


class TestCategoricalCleaner:
    def test_partner_yes_maps_to_1(self, single_df):
        cc = CategoricalCleaner()
        out = cc.transform(single_df)
        assert out["Partner"].iloc[0] == 1

    def test_dependents_no_maps_to_0(self, single_df):
        cc = CategoricalCleaner()
        out = cc.transform(single_df)
        assert out["Dependents"].iloc[0] == 0

    def test_gender_male_maps_to_1(self, single_df):
        cc = CategoricalCleaner()
        out = cc.transform(single_df)
        assert out["gender"].iloc[0] == 1

    def test_gender_female_maps_to_0(self, base_row):
        base_row["gender"] = "Female"
        df = pd.DataFrame([base_row])
        cc = CategoricalCleaner()
        out = cc.transform(df)
        assert out["gender"].iloc[0] == 0

    def test_phone_service_maps_correctly(self, base_row):
        for val, expected in [("Yes", 1), ("No", 0)]:
            base_row["PhoneService"] = val
            df = pd.DataFrame([base_row])
            cc = CategoricalCleaner()
            out = cc.transform(df)
            assert out["PhoneService"].iloc[0] == expected

    def test_paperless_billing_maps_correctly(self, base_row):
        base_row["PaperlessBilling"] = "No"
        df = pd.DataFrame([base_row])
        cc = CategoricalCleaner()
        out = cc.transform(df)
        assert out["PaperlessBilling"].iloc[0] == 0

    def test_categorical_cols_stay_string(self, single_df):
        """Columns not in the binary-map list must remain string dtype."""
        cc = CategoricalCleaner()
        out = cc.transform(single_df)
        assert out["Contract"].dtype == object
        assert out["PaymentMethod"].dtype == object

    def test_transform_does_not_mutate_input(self, single_df):
        original = single_df["Partner"].iloc[0]
        cc = CategoricalCleaner()
        cc.transform(single_df)
        assert single_df["Partner"].iloc[0] == original
