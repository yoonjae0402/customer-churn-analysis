
import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, roc_auc_score

from src.config import config
from src.logger import logger
from src.data_loader import load_csv_data, split_data, validate_data
from src.model import create_model_pipeline, save_model

def main():
    parser = argparse.ArgumentParser(description="Train Customer Churn Model")
    parser.add_argument("--model", type=str, help="Model type (logistic_regression, random_forest, xgboost)", default=None)
    args = parser.parse_args()

    # 1. Update Config if CLI arg provided
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

    # 4. Preprocessing Clean-up (Basic)
    # Convert TotalCharges to numeric if not already done by loader
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    
    # Map Target to 0/1
    target_col = config.feature_config["target"]
    if df[target_col].dtype == 'object':
        df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})
    
    # 5. Split Data
    X_train, X_test, y_train, y_test = split_data(
        df, 
        target_column=target_col,
        test_size=config.training_config["test_size"],
        random_state=config.config.get("project", {}).get("random_state", 42)
    )
    logger.info(f"Data split: Train shape {X_train.shape}, Test shape {X_test.shape}")

    # 6. Create Pipeline
    model_params = config.model_config["params"].get(model_type, {})
    pipeline = create_model_pipeline(model_type, model_params)
    
    # 7. Cross-Validation
    cv_folds = config.training_config["cv_folds"]
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    logger.info(f"Running {cv_folds}-fold Cross-Validation...")
    
    cv_results = cross_validate(
        pipeline, X_train, y_train, 
        cv=StratifiedKFold(n_splits=cv_folds), 
        scoring=scoring,
        n_jobs=-1
    )
    
    logger.info("Cross-Validation Results:")
    for metric in scoring:
        mean_score = np.mean(cv_results[f"test_{metric}"])
        std_score = np.std(cv_results[f"test_{metric}"])
        logger.info(f"  {metric}: {mean_score:.4f} (+/- {std_score:.4f})")

    # 8. Train Final Model
    logger.info("Training final model on full training set...")
    pipeline.fit(X_train, y_train)
    
    # 9. Evaluate on Test Set
    logger.info("Evaluating on Test Set...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    logger.info(f"Test ROC AUC: {auc:.4f}")
    
    report = classification_report(y_test, y_pred)
    logger.info(f"Classification Report:\n{report}")

    # 10. Save Model
    model_dir = Path(config.paths["models"])
    model_path = model_dir / f"{model_type}_pipeline.pkl"
    
    joblib.dump(pipeline, model_path)
    logger.info(f"Model pipeline saved to {model_path}")

    # Save feature names logic?
    # The new pipeline handles features dynamically, but for explanation/shap, we might want names.
    # We can inspect the pipeline steps if needed.

if __name__ == "__main__":
    main()
