
import time
import logging
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import Dict, Any

from src.config import config
from src.model import load_model, predict_churn
from src.services.marketing import marketing_service
from .schemas import CustomerData, PredictionResponse

# Setup logger
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn likelihood and generates personalized retention offers.",
    version=config.config["project"]["version"]
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model_pipeline = None

@app.on_event("startup")
async def startup_event():
    """
    Load the trained model pipeline on startup.
    """
    global model_pipeline
    model_type = config.model_config["type"]
    model_path = f"{config.paths['models']}/{model_type}_pipeline.pkl"
    logger.info(f"Loading model pipeline from {model_path}...")
    try:
        model_pipeline = load_model(model_path)
        logger.info("Model pipeline loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}. Please train the model first using src/train.py")
        # In a real production system, we might want to fail fast, but for dev we can return errors later
        model_pipeline = None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_pipeline = None

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error. Please check logs for details."}
    )

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    status = "healthy" if model_pipeline is not None else "degraded (model not loaded)"
    return {
        "status": status,
        "version": config.config["project"]["version"],
        "model_type": config.model_config["type"]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn_endpoint(customer_data: CustomerData):
    """
    Predict churn probability and generate marketing offer.
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    start_time = time.time()
    
    try:
        # Convert Pydantic model to DataFrame
        # Data conversion is minimal because the pipeline handles feature engineering
        input_data = customer_data.dict()
        df = pd.DataFrame([input_data])
        
        # Predict churn
        churn_prob = predict_churn(model_pipeline, df)[0]
        
        # Generate generic offer via Gemini
        marketing_offer = await marketing_service.generate_offer(churn_prob, input_data)
        
        latency_ms = round((time.time() - start_time) * 1000, 2)
        
        return {
            "churn_probability": churn_prob,
            "marketing_offer": marketing_offer,
            "metadata": {
                "model_version": config.config["project"]["version"],
                "latency_ms": latency_ms,
                "model_type": config.model_config["type"]
            }
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
