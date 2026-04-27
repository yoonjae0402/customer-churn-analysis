"""
Unit tests for scripts/check_drift.py drift detection functions.

Tests the pure analysis functions directly — no filesystem, no model needed.
"""

import json
import sys
from pathlib import Path

import pytest

# Make the scripts package importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.check_drift import (  # noqa: E402
    compute_categorical_drift,
    compute_numerical_drift,
    load_feature_log,
    load_reference_stats,
    summarise_drift,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ref_stats():
    return {
        "numerical": {
            "tenure": {
                "mean": 32.0,
                "std": 24.0,
                "min": 0.0,
                "max": 72.0,
                "p25": 9.0,
                "p75": 55.0,
            },
            "MonthlyCharges": {
                "mean": 64.76,
                "std": 30.09,
                "min": 18.25,
                "max": 118.75,
                "p25": 35.5,
                "p75": 89.85,
            },
        },
        "categorical": {
            "Contract": {
                "Month-to-month": 0.55,
                "One year": 0.21,
                "Two year": 0.24,
            },
        },
    }


@pytest.fixture
def matching_rows():
    """Rows whose distribution closely matches the reference stats."""
    return [
        {"tenure": 30, "MonthlyCharges": 65.0, "Contract": "Month-to-month"},
        {"tenure": 34, "MonthlyCharges": 63.0, "Contract": "One year"},
        {"tenure": 28, "MonthlyCharges": 66.0, "Contract": "Month-to-month"},
        {"tenure": 36, "MonthlyCharges": 65.5, "Contract": "Two year"},
        {"tenure": 32, "MonthlyCharges": 64.5, "Contract": "Month-to-month"},
    ]


@pytest.fixture
def drifted_rows():
    """Rows with very long tenures — mean far from the 32-month training mean."""
    return [
        {"tenure": 70, "MonthlyCharges": 65.0, "Contract": "Month-to-month"},
        {"tenure": 68, "MonthlyCharges": 63.0, "Contract": "One year"},
        {"tenure": 72, "MonthlyCharges": 66.0, "Contract": "Month-to-month"},
        {"tenure": 69, "MonthlyCharges": 64.0, "Contract": "Two year"},
        {"tenure": 71, "MonthlyCharges": 65.0, "Contract": "Month-to-month"},
    ]


# ---------------------------------------------------------------------------
# compute_numerical_drift
# ---------------------------------------------------------------------------


class TestNumericalDrift:
    def test_no_drift_when_means_match(self, matching_rows, ref_stats):
        results = compute_numerical_drift(matching_rows, ref_stats, threshold=2.0)
        assert "tenure" in results
        assert results["tenure"]["drifted"] is False

    def test_drift_detected_for_shifted_mean(self, drifted_rows, ref_stats):
        """tenure ~70, train_mean=32, std=24 → z≈1.58. Drifts at threshold=1.0."""
        results = compute_numerical_drift(drifted_rows, ref_stats, threshold=1.0)
        assert results["tenure"]["drifted"] is True

    def test_drift_not_detected_at_loose_threshold(self, drifted_rows, ref_stats):
        """Same data at threshold=3.0 should not flag (z ≈ 1.58 < 3.0)."""
        results = compute_numerical_drift(drifted_rows, ref_stats, threshold=3.0)
        assert results["tenure"]["drifted"] is False

    def test_z_score_is_positive(self, matching_rows, ref_stats):
        results = compute_numerical_drift(matching_rows, ref_stats)
        assert results["tenure"]["z_score"] >= 0

    def test_missing_feature_in_rows_is_skipped(self, ref_stats):
        rows = [{"MonthlyCharges": 65.0}]  # no "tenure" key
        results = compute_numerical_drift(rows, ref_stats)
        assert "tenure" not in results
        assert "MonthlyCharges" in results

    def test_zero_std_does_not_raise(self, ref_stats):
        """If training std is 0 (constant feature), z-score defaults to 0."""
        ref_stats["numerical"]["tenure"]["std"] = 0.0
        rows = [{"tenure": 99.0}]
        results = compute_numerical_drift(rows, ref_stats)
        assert results["tenure"]["z_score"] == 0.0

    def test_returns_log_mean(self, ref_stats):
        rows = [{"tenure": 10.0}, {"tenure": 20.0}]
        results = compute_numerical_drift(rows, ref_stats)
        assert abs(results["tenure"]["log_mean"] - 15.0) < 1e-6


# ---------------------------------------------------------------------------
# compute_categorical_drift
# ---------------------------------------------------------------------------


class TestCategoricalDrift:
    def test_no_drift_when_frequencies_match(self, matching_rows, ref_stats):
        # 3/5 Month-to-month = 0.60, ref=0.55, dev=0.05 < 0.15
        results = compute_categorical_drift(matching_rows, ref_stats, threshold=0.15)
        assert "Contract" in results
        assert results["Contract"]["drifted"] is False

    def test_drift_detected_when_distribution_shifts(self, ref_stats):
        # All rows Month-to-month: freq=1.0, ref=0.55, dev=0.45 > 0.15
        rows = [{"Contract": "Month-to-month"}] * 10
        results = compute_categorical_drift(rows, ref_stats, threshold=0.15)
        assert results["Contract"]["drifted"] is True

    def test_new_category_triggers_drift(self, ref_stats):
        rows = [{"Contract": "Month-to-month"}, {"Contract": "Five year"}]
        results = compute_categorical_drift(rows, ref_stats)
        assert results["Contract"]["drifted"] is True
        assert "Five year" in results["Contract"]["new_categories"]

    def test_max_deviation_is_non_negative(self, matching_rows, ref_stats):
        results = compute_categorical_drift(matching_rows, ref_stats)
        assert results["Contract"]["max_deviation"] >= 0

    def test_missing_feature_in_rows_is_skipped(self, ref_stats):
        rows = [{"tenure": 30}]  # no "Contract" key
        results = compute_categorical_drift(rows, ref_stats)
        assert "Contract" not in results


# ---------------------------------------------------------------------------
# summarise_drift
# ---------------------------------------------------------------------------


class TestSummariseDrift:
    def test_no_drift(self):
        num = {"tenure": {"drifted": False}, "MonthlyCharges": {"drifted": False}}
        cat = {"Contract": {"drifted": False}}
        any_drift, flagged = summarise_drift(num, cat)
        assert any_drift is False
        assert flagged == []

    def test_numerical_drift_flagged(self):
        num = {"tenure": {"drifted": True}, "MonthlyCharges": {"drifted": False}}
        cat = {"Contract": {"drifted": False}}
        any_drift, flagged = summarise_drift(num, cat)
        assert any_drift is True
        assert "tenure" in flagged

    def test_categorical_drift_flagged(self):
        num = {"tenure": {"drifted": False}}
        cat = {"Contract": {"drifted": True}}
        any_drift, flagged = summarise_drift(num, cat)
        assert any_drift is True
        assert "Contract" in flagged

    def test_multiple_features_flagged(self):
        num = {"tenure": {"drifted": True}, "MonthlyCharges": {"drifted": True}}
        cat = {"Contract": {"drifted": True}}
        any_drift, flagged = summarise_drift(num, cat)
        assert len(flagged) == 3


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


class TestLoadFeatureLog:
    def test_loads_valid_jsonl(self, tmp_path):
        log = tmp_path / "feature_log.jsonl"
        row1 = json.dumps(
            {"timestamp": "2026-01-01T00:00:00Z", "features": {"tenure": 5}}
        )
        row2 = json.dumps(
            {"timestamp": "2026-01-01T00:01:00Z", "features": {"tenure": 10}}
        )
        log.write_text(row1 + "\n" + row2 + "\n")
        rows = load_feature_log(log)
        assert len(rows) == 2
        assert rows[0]["tenure"] == 5

    def test_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_feature_log(tmp_path / "nonexistent.jsonl")

    def test_skips_blank_lines(self, tmp_path):
        log = tmp_path / "feature_log.jsonl"
        log.write_text(
            json.dumps({"timestamp": "t", "features": {"tenure": 1}}) + "\n"
            "\n" + json.dumps({"timestamp": "t", "features": {"tenure": 2}}) + "\n"
        )
        rows = load_feature_log(log)
        assert len(rows) == 2


class TestLoadReferenceStats:
    def test_loads_valid_json(self, tmp_path, ref_stats):
        path = tmp_path / "train_stats.json"
        path.write_text(json.dumps(ref_stats))
        loaded = load_reference_stats(path)
        assert "numerical" in loaded
        assert "categorical" in loaded

    def test_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_reference_stats(tmp_path / "nonexistent.json")
