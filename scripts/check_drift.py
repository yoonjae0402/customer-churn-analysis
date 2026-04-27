"""
Feature drift detection script.

Loads the prediction feature log (logs/feature_log.jsonl) and compares each
feature's distribution against the training reference (models/train_stats.json).
Flags features that have shifted significantly.

Detection rules
---------------
Numerical  : flag if |log_mean − train_mean| / train_std > --num-threshold (default 2.0)
Categorical: flag if any category's observed frequency deviates from training
             frequency by more than --cat-threshold (default 0.15 = 15 pp)

Exit codes
----------
0  — no drift detected (or fewer than --min-samples logged rows)
1  — drift detected in one or more features
2  — missing input file (no log or no reference stats)

Usage
-----
    python3 scripts/check_drift.py
    python3 scripts/check_drift.py --log-path logs/feature_log.jsonl \\
                                   --stats-path models/train_stats.json \\
                                   --num-threshold 3.0 \\
                                   --cat-threshold 0.20 \\
                                   --min-samples 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_feature_log(log_path: Path) -> List[dict]:
    """Read JSONL feature log; return list of feature dicts (one per request)."""
    if not log_path.exists():
        raise FileNotFoundError(f"Feature log not found: {log_path}")
    rows = []
    with open(log_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line)["features"])
    return rows


def load_reference_stats(stats_path: Path) -> dict:
    """Load training reference statistics JSON."""
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Training reference stats not found: {stats_path}. "
            "Run `make train` to generate them."
        )
    return json.loads(stats_path.read_text())


# ---------------------------------------------------------------------------
# Drift computation (pure functions — importable for unit tests)
# ---------------------------------------------------------------------------

def compute_numerical_drift(
    rows: List[dict],
    ref_stats: dict,
    threshold: float = 2.0,
) -> Dict[str, dict]:
    """
    For each numerical feature, compare the log mean against the training mean.
    Returns a dict keyed by feature name with drift details.

    Flags the feature when:
        |log_mean - train_mean| / train_std > threshold
    """
    results: Dict[str, dict] = {}
    for feature, ref in ref_stats.get("numerical", {}).items():
        vals = []
        for row in rows:
            v = row.get(feature)
            if v is not None:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
        if not vals:
            continue
        log_mean = sum(vals) / len(vals)
        train_std = ref["std"]
        z = abs(log_mean - ref["mean"]) / train_std if train_std > 0 else 0.0
        results[feature] = {
            "log_mean": round(log_mean, 4),
            "train_mean": ref["mean"],
            "train_std": train_std,
            "z_score": round(z, 3),
            "drifted": z > threshold,
        }
    return results


def compute_categorical_drift(
    rows: List[dict],
    ref_stats: dict,
    threshold: float = 0.15,
) -> Dict[str, dict]:
    """
    For each categorical feature, compare observed frequency vs training frequency.
    Returns a dict keyed by feature name with drift details.

    Flags the feature when any category's |observed_freq - train_freq| > threshold.
    """
    results: Dict[str, dict] = {}
    for feature, ref_freq in ref_stats.get("categorical", {}).items():
        vals = [row[feature] for row in rows if feature in row and row[feature] is not None]
        if not vals:
            continue
        n = len(vals)
        observed: Dict[str, float] = {}
        for v in vals:
            observed[str(v)] = observed.get(str(v), 0) + 1
        observed = {k: round(v / n, 4) for k, v in observed.items()}

        # Compute max absolute frequency deviation across all known categories
        all_cats = set(ref_freq) | set(observed)
        deviations = {
            cat: round(abs(observed.get(cat, 0.0) - ref_freq.get(cat, 0.0)), 4)
            for cat in all_cats
        }
        max_dev = max(deviations.values(), default=0.0)
        new_cats = sorted(set(observed) - set(ref_freq))

        results[feature] = {
            "observed_freq": observed,
            "train_freq": ref_freq,
            "max_deviation": round(max_dev, 4),
            "new_categories": new_cats,
            "drifted": max_dev > threshold or bool(new_cats),
        }
    return results


def summarise_drift(
    num_results: Dict[str, dict],
    cat_results: Dict[str, dict],
) -> Tuple[bool, List[str]]:
    """Return (any_drift, list_of_drifted_feature_names)."""
    flagged = []
    for feat, r in num_results.items():
        if r["drifted"]:
            flagged.append(feat)
    for feat, r in cat_results.items():
        if r["drifted"]:
            flagged.append(feat)
    return bool(flagged), flagged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Feature drift detection")
    parser.add_argument(
        "--log-path", type=Path,
        default=ROOT / "logs" / "feature_log.jsonl",
        help="Path to the JSONL prediction log",
    )
    parser.add_argument(
        "--stats-path", type=Path,
        default=ROOT / "models" / "train_stats.json",
        help="Path to the training reference stats JSON",
    )
    parser.add_argument(
        "--num-threshold", type=float, default=2.0,
        help="Z-score threshold for numerical drift (default 2.0)",
    )
    parser.add_argument(
        "--cat-threshold", type=float, default=0.15,
        help="Max frequency deviation for categorical drift (default 0.15)",
    )
    parser.add_argument(
        "--min-samples", type=int, default=20,
        help="Minimum logged rows before drift is reported (default 20)",
    )
    args = parser.parse_args()

    # Load inputs
    try:
        rows = load_feature_log(args.log_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    try:
        ref_stats = load_reference_stats(args.stats_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    n = len(rows)
    print(f"Feature drift report — {n} logged requests")
    print(f"Reference: {args.stats_path.relative_to(ROOT)}")
    print(f"Log:       {args.log_path.relative_to(ROOT) if args.log_path.is_absolute() else args.log_path}")
    print()

    if n < args.min_samples:
        print(
            f"  Only {n} samples in log (min={args.min_samples}). "
            "Insufficient data — no drift reported."
        )
        return 0

    num_results = compute_numerical_drift(rows, ref_stats, args.num_threshold)
    cat_results = compute_categorical_drift(rows, ref_stats, args.cat_threshold)
    any_drift, flagged = summarise_drift(num_results, cat_results)

    # ── Numerical report ────────────────────────────────────────────────────
    print("Numerical features:")
    if num_results:
        for feat, r in num_results.items():
            flag = "  DRIFT" if r["drifted"] else "     ok"
            print(
                f"  {flag}  {feat:<20}  "
                f"log_mean={r['log_mean']:>8.3f}  "
                f"train_mean={r['train_mean']:>8.3f}  "
                f"train_std={r['train_std']:>7.3f}  "
                f"z={r['z_score']:.2f}"
            )
    else:
        print("  (no numerical features in log)")

    # ── Categorical report ───────────────────────────────────────────────────
    print("\nCategorical features:")
    if cat_results:
        for feat, r in cat_results.items():
            flag = "  DRIFT" if r["drifted"] else "     ok"
            extras = ""
            if r["new_categories"]:
                extras = f"  [new categories: {r['new_categories']}]"
            print(
                f"  {flag}  {feat:<20}  "
                f"max_deviation={r['max_deviation']:.3f}{extras}"
            )
            if r["drifted"]:
                for cat in sorted(set(r["observed_freq"]) | set(r["train_freq"])):
                    obs = r["observed_freq"].get(cat, 0.0)
                    ref = r["train_freq"].get(cat, 0.0)
                    print(f"           {cat:<30}  obs={obs:.3f}  ref={ref:.3f}  Δ={obs-ref:+.3f}")
    else:
        print("  (no categorical features in log)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    if any_drift:
        print(f"DRIFT DETECTED in: {', '.join(flagged)}")
        return 1
    else:
        print("No drift detected.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
