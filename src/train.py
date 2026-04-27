import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import config
from src.data.loader import load_csv_data, temporal_split, validate_data
from src.logger import logger
from src.models.evaluation import (
    find_optimal_threshold,
    plot_calibration_curve,
    plot_learning_curves,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.models.pipeline import create_model_pipeline


def main():  # noqa: C901
    parser = argparse.ArgumentParser(description="Train Customer Churn Model")
    parser.add_argument(
        "--model",
        type=str,
        help="Model type (logistic_regression, random_forest, xgboost)",
        default=None,
    )
    parser.add_argument(
        "--threshold-strategy",
        type=str,
        choices=["f1", "recall_at_precision", "cost"],
        default="f1",
        help="Strategy for finding the optimal classification threshold",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating evaluation plots (faster for CI/CD)",
    )
    args = parser.parse_args()

    # 1. Resolve model type from CLI or config
    model_type = args.model if args.model else config.model_config["type"]
    logger.info(f"Starting training pipeline with model: {model_type}")

    # 2. Load Data
    data_path = Path(config.paths["raw_data"])
    try:
        df = load_csv_data(data_path)
        logger.info(f"Loaded data from {data_path}, shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Data file not found at {data_path}. Please check config.yaml")
        return

    # 3. Validate Data
    is_valid, missing = validate_data(df)
    if not is_valid:
        logger.error(f"Data validation failed. Missing columns: {missing}")
        return

    # 4. Basic cleanup
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    target_col = config.feature_config["target"]
    if df[target_col].dtype == "object":
        df[target_col] = df[target_col].map({"Yes": 1, "No": 0})

    pos_rate = df[target_col].mean()
    logger.info(
        f"Class distribution — Churn: {pos_rate:.1%}, No Churn: {1 - pos_rate:.1%}"
    )

    # 5. Temporal split — train on oldest 80 % of customers (by tenure),
    #    test on the most recently acquired 20 %.  Simulates real deployment:
    #    model built on established cohorts, evaluated on new-joiner churn risk.
    random_state = config.config.get("project", {}).get("random_state", 42)
    X_train, X_test, y_train, y_test = temporal_split(
        df,
        target_column=target_col,
        tenure_col="tenure",
        test_size=config.training_config["test_size"],
    )
    logger.info(
        f"Temporal split — train: {X_train.shape[0]} rows "
        f"(tenure {X_train['tenure'].min()}–{X_train['tenure'].max()} months, "
        f"churn {y_train.mean():.1%})  |  "
        f"test: {X_test.shape[0]} rows "
        f"(tenure {X_test['tenure'].min()}–{X_test['tenure'].max()} months, "
        f"churn {y_test.mean():.1%})"
    )

    # 6. Build Pipeline
    model_params = config.model_config["params"].get(model_type, {})
    pipeline = create_model_pipeline(model_type, model_params)

    # 7. Cross-Validation (using roc_auc + f1 for imbalanced data awareness)
    cv_folds = config.training_config["cv_folds"]
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    logger.info(f"Running {cv_folds}-fold stratified cross-validation...")

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        scoring=scoring,
        n_jobs=-1,
    )

    logger.info("Cross-Validation Results:")
    for metric in scoring:
        mean_score = np.mean(cv_results[f"test_{metric}"])
        std_score = np.std(cv_results[f"test_{metric}"])
        logger.info(f"  {metric}: {mean_score:.4f} (+/- {std_score:.4f})")

    # 8. Train with probability calibration
    # Tree models (RF, XGBoost) are often overconfident. Wrapping in
    # CalibratedClassifierCV with isotonic regression and 5-fold CV produces
    # more reliable probabilities. The pipeline is fitted internally here.
    logger.info("Training final model with isotonic calibration (5-fold)...")
    calibrated_pipeline = CalibratedClassifierCV(pipeline, cv=5, method="isotonic")
    calibrated_pipeline.fit(X_train, y_train)

    # 10. Find optimal threshold on the test set
    y_prob = calibrated_pipeline.predict_proba(X_test)[:, 1]
    optimal_threshold, threshold_metrics = find_optimal_threshold(
        y_test.values, y_prob, strategy=args.threshold_strategy
    )
    logger.info(
        f"Optimal threshold ({args.threshold_strategy}): {optimal_threshold:.4f}"
    )
    logger.info(f"  Precision: {threshold_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {threshold_metrics['recall']:.4f}")
    logger.info(f"  F1:        {threshold_metrics['f1']:.4f}")
    logger.info(f"  Accuracy:  {threshold_metrics['accuracy']:.4f}")

    # 11. Full test-set evaluation at optimal threshold
    logger.info("Test set evaluation at optimal threshold:")
    y_pred = (y_prob >= optimal_threshold).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    logger.info(f"Test ROC-AUC: {auc:.4f}")
    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
    logger.info(f"Classification Report:\n{report}")

    # 12. Save model + threshold + metadata together
    model_dir = Path(config.paths["models"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_type}_pipeline.pkl"
    threshold_path = model_dir / f"{model_type}_threshold.json"
    metadata_path = model_dir / f"{model_type}_metadata.json"

    joblib.dump(calibrated_pipeline, model_path)
    threshold_path.write_text(
        json.dumps(
            {"threshold": optimal_threshold, "strategy": args.threshold_strategy}
        )
    )

    # Capture git SHA for reproducibility; graceful fallback if not in a repo
    import subprocess

    try:
        git_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        git_sha = "unknown"

    from datetime import datetime, timezone

    metadata = {
        "model_type": model_type,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "threshold": optimal_threshold,
        "threshold_strategy": args.threshold_strategy,
        "metrics": {
            "roc_auc": round(float(auc), 4),
            "precision": round(float(threshold_metrics["precision"]), 4),
            "recall": round(float(threshold_metrics["recall"]), 4),
            "f1": round(float(threshold_metrics["f1"]), 4),
        },
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "test_churn_rate": round(float(y_test.mean()), 4),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # Save training reference stats for drift detection
    # Covers the three raw numerical inputs and three key categorical features
    num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_features = ["Contract", "InternetService", "PaymentMethod"]
    train_stats: dict = {"numerical": {}, "categorical": {}}
    for col in num_features:
        if col in X_train.columns:
            vals = X_train[col].dropna()
            train_stats["numerical"][col] = {
                "mean": round(float(vals.mean()), 4),
                "std": round(float(vals.std()), 4),
                "min": round(float(vals.min()), 4),
                "max": round(float(vals.max()), 4),
                "p25": round(float(vals.quantile(0.25)), 4),
                "p75": round(float(vals.quantile(0.75)), 4),
            }
    for col in cat_features:
        if col in X_train.columns:
            freq = X_train[col].value_counts(normalize=True).round(4).to_dict()
            train_stats["categorical"][col] = {
                str(k): float(v) for k, v in freq.items()
            }
    train_stats_path = model_dir / "train_stats.json"
    train_stats_path.write_text(json.dumps(train_stats, indent=2))

    logger.info(f"Calibrated pipeline saved to {model_path}")
    logger.info(f"Optimal threshold saved to {threshold_path}")
    logger.info(f"Metadata saved to {metadata_path}")
    logger.info(f"Training reference stats saved to {train_stats_path}")

    # 13. Optional evaluation plots
    if not args.skip_plots:
        reports_dir = Path("reports/figures")
        reports_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating evaluation plots...")

        plot_roc_curve(
            y_test.values,
            y_prob,
            save_path=reports_dir / f"{model_type}_roc_curve.png",
        )
        plot_precision_recall_curve(
            y_test.values,
            y_prob,
            optimal_threshold=optimal_threshold,
            save_path=reports_dir / f"{model_type}_pr_curve.png",
        )
        plot_calibration_curve(
            y_test.values,
            y_prob,
            save_path=reports_dir / f"{model_type}_calibration.png",
        )
        plot_learning_curves(
            pipeline,
            X_train,
            y_train,
            cv=cv_folds,
            save_path=reports_dir / f"{model_type}_learning_curves.png",
        )

        logger.info(f"Plots saved to {reports_dir}/")


if __name__ == "__main__":
    main()
