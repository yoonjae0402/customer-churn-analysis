import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import CustomerData, PredictionResponse
from src.api.services.marketing import marketing_service
from src.config import config
from src.models.pipeline import load_model, load_threshold, predict_churn_with_threshold

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predicts churn likelihood" " and generates personalised retention offers."
    ),
    version=config.config["project"]["version"],
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

_raw_origins = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080"
)
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    response = await call_next(request)
    latency_ms = round((time.time() - start) * 1000, 1)
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"→ {response.status_code} ({latency_ms} ms)"
    )
    response.headers["X-Request-Id"] = request_id
    return response


# ---------------------------------------------------------------------------
# Global state: model pipeline + metadata
# ---------------------------------------------------------------------------

model_pipeline = None
optimal_threshold: float = config.training_config.get("default_threshold", 0.5)
model_metadata: dict = {}


@app.on_event("startup")
async def startup_event():
    """Load the trained model pipeline, threshold, and metadata on startup."""
    global model_pipeline, optimal_threshold, model_metadata
    model_type = config.model_config["type"]
    model_path = f"{config.paths['models']}/{model_type}_pipeline.pkl"
    metadata_path = Path(config.paths["models"]) / f"{model_type}_metadata.json"

    logger.info(f"Loading model pipeline from {model_path}...")
    try:
        model_pipeline = load_model(model_path)
        optimal_threshold = load_threshold(model_type, config.paths["models"])
        logger.info(f"Model loaded. Optimal threshold: {optimal_threshold:.4f}")
    except FileNotFoundError:
        logger.error(
            f"Model file not found at {model_path}. Train the model first: make train"
        )
        model_pipeline = None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_pipeline = None

    if metadata_path.exists():
        try:
            model_metadata = json.loads(metadata_path.read_text())
        except Exception:
            model_metadata = {}


# ---------------------------------------------------------------------------
# Feature distribution logging
# ---------------------------------------------------------------------------

_FEATURE_LOG_PATH = Path(os.getenv("FEATURE_LOG_PATH", "logs/feature_log.jsonl"))


def _log_features(features: dict) -> None:
    """Append raw input features to the JSONL feature log.

    Errors are silenced so a logging failure never impacts predictions.
    Override the log path with the FEATURE_LOG_PATH env var (useful in tests).
    """
    try:
        _FEATURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "features": features,
        }
        with open(_FEATURE_LOG_PATH, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.warning(f"Feature log write failed (non-fatal): {exc}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check logs for details."},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/healthz")
async def healthz():
    """Liveness probe — used by load balancers and container orchestrators."""
    status = "healthy" if model_pipeline is not None else "degraded"
    return {
        "status": status,
        "version": config.config["project"]["version"],
        "model_type": config.model_config["type"],
        "threshold": optimal_threshold,
    }


@app.get("/health")
async def health_check():
    """Legacy alias for /healthz — kept for backwards compatibility."""
    return await healthz()


@app.get("/model/info")
async def model_info():
    """Return model provenance: trained_at, git SHA, and evaluation metrics."""
    if not model_metadata:
        return {
            "status": "no metadata available — retrain with `make train`",
            "model_type": config.model_config["type"],
            "loaded": model_pipeline is not None,
        }
    return {
        "loaded": model_pipeline is not None,
        **model_metadata,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn_endpoint(customer_data: CustomerData):
    """Predict churn probability and generate a personalised retention offer."""
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first: make train",
        )

    start_time = time.time()

    try:
        input_data: Dict[str, Any] = customer_data.model_dump()
        _log_features(input_data)
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

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
