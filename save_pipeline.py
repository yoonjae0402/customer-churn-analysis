import joblib
from pathlib import Path

# Define paths
MODELS_DIR = Path('models/')
SCALER_PATH = MODELS_DIR / 'scaler.pkl'
BEST_MODEL_PATH = MODELS_DIR / 'best_model.pkl'
CHURN_MODEL_PATH = MODELS_DIR / 'churn_model.pkl'

print(f"Loading scaler from: {SCALER_PATH}")
# Load the scaler
scaler = joblib.load(SCALER_PATH)
print("Scaler loaded successfully.")

print(f"Loading best model from: {BEST_MODEL_PATH}")
# Load the best model
best_model = joblib.load(BEST_MODEL_PATH)
print("Best model loaded successfully.")

# Create a dictionary to hold both the scaler and the model
full_pipeline = {
    'scaler': scaler,
    'model': best_model
}

print(f"Saving combined pipeline to: {CHURN_MODEL_PATH}")
# Save the combined pipeline
joblib.dump(full_pipeline, CHURN_MODEL_PATH)
print("Combined pipeline (scaler and model) saved successfully as churn_model.pkl.")
