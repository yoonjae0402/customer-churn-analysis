
import argparse
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, roc_auc_score

from src.config import config
from src.logger import logger
from src.data_loader import load_csv_data, split_data, validate_data
from src.model import create_model_pipeline, save_model
from src.evaluation import (
    find_optimal_threshold,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_calibration_curve,
    plot_learning_curves,
)


def main():
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

    # 5. Split Data
    random_state = config.config.get("project", {}).get("random_state", 42)
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column=target_col,
        test_size=config.training_config["test_size"],
        random_state=random_state,
    )
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

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
    calibrated_pipeline = CalibratedClassifierCV(
        pipeline, cv=5, method="isotonic"
    )
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
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

    # 12. Save model + threshold together
    model_dir = Path(config.paths["models"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_type}_pipeline.pkl"
    threshold_path = model_dir / f"{model_type}_threshold.json"

    joblib.dump(calibrated_pipeline, model_path)
    threshold_path.write_text(
        json.dumps({"threshold": optimal_threshold, "strategy": args.threshold_strategy})
    )
    logger.info(f"Calibrated pipeline saved to {model_path}")
    logger.info(f"Optimal threshold saved to {threshold_path}")

    # 13. Optional evaluation plots
    if not args.skip_plots:
        reports_dir = Path("reports/figures")
        reports_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating evaluation plots...")

        plot_roc_curve(
            y_test.values, y_prob,
            save_path=reports_dir / f"{model_type}_roc_curve.png",
        )
        plot_precision_recall_curve(
            y_test.values, y_prob,
            optimal_threshold=optimal_threshold,
            save_path=reports_dir / f"{model_type}_pr_curve.png",
        )
        plot_calibration_curve(
            y_test.values, y_prob,
            save_path=reports_dir / f"{model_type}_calibration.png",
        )
        plot_learning_curves(
            pipeline, X_train, y_train,
            cv=cv_folds,
            save_path=reports_dir / f"{model_type}_learning_curves.png",
        )

        logger.info(f"Plots saved to {reports_dir}/")


if __name__ == "__main__":
    main()
