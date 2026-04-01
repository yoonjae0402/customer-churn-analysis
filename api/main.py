
import time
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import Dict, Any

from src.config import config
from src.model import load_model, predict_churn_with_threshold, load_threshold
from src.services.marketing import marketing_service
from .schemas import CustomerData, PredictionResponse

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn likelihood and generates personalised retention offers.",
    version=config.config["project"]["version"],
)

# CORS: restrict to known origins in production.
# Override ALLOWED_ORIGINS env var to add more origins (comma-separated).
import os

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Global state: model pipeline + optimal threshold
model_pipeline = None
optimal_threshold: float = config.training_config.get("default_threshold", 0.5)


@app.on_event("startup")
async def startup_event():
    """Load the trained model pipeline and threshold on startup."""
    global model_pipeline, optimal_threshold
    model_type = config.model_config["type"]
    model_path = f"{config.paths['models']}/{model_type}_pipeline.pkl"
    logger.info(f"Loading model pipeline from {model_path}...")
    try:
        model_pipeline = load_model(model_path)
        optimal_threshold = load_threshold(model_type, config.paths["models"])
        logger.info(
            f"Model loaded. Optimal threshold: {optimal_threshold:.4f}"
        )
    except FileNotFoundError:
        logger.error(
            f"Model file not found at {model_path}. Train the model first: make setup-model"
        )
        model_pipeline = None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_pipeline = None


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check logs for details."},
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    status = "healthy" if model_pipeline is not None else "degraded (model not loaded)"
    return {
        "status": status,
        "version": config.config["project"]["version"],
        "model_type": config.model_config["type"],
        "threshold": optimal_threshold,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn_endpoint(customer_data: CustomerData):
    """Predict churn probability and generate a personalised retention offer."""
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first: make setup-model",
        )

    start_time = time.time()

    try:
        input_data: Dict[str, Any] = customer_data.model_dump()
        df = pd.DataFrame([input_data])

        churn_prob, churn_pred = predict_churn_with_threshold(
            model_pipeline, df, threshold=optimal_threshold
        )
        churn_prob_scalar = float(churn_prob[0])
        churn_pred_scalar = int(churn_pred[0])

        marketing_offer = await marketing_service.generate_offer(
            churn_prob_scalar, input_data
        )

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "churn_probability": churn_prob_scalar,
            "churn_prediction": churn_pred_scalar,
            "marketing_offer": marketing_offer,
            "metadata": {
                "model_version": config.config["project"]["version"],
                "model_type": config.model_config["type"],
                "threshold": optimal_threshold,
                "latency_ms": latency_ms,
            },
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
