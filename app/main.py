import os
import re
import argparse
import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

import google.generativeai as genai

# Rich imports (for CLI mode)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Rich console instance
console = Console()

# --- Logging Configuration ---
LOG_FILE = "app.log"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File Handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# --- Custom Exception ---
class ChurnException(HTTPException):
    def __init__(self, status_code: int, detail: str, ex_type: str = "ChurnPredictionError"):
        """
        Initializes a new ChurnException instance.

        Args:
            status_code (int): The HTTP status code to return with this exception.
            detail (str): A detailed message describing the error.
            ex_type (str, optional): A specific type for the exception (e.g., "ModelLoadingError").
                                     Defaults to "ChurnPredictionError".
        """
        super().__init__(status_code=status_code, detail=detail)
        self.ex_type = ex_type
        logger.error(f"{ex_type}: {detail}")

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in .env file or environment variables.")
    raise ChurnException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                         detail="Gemini API key is not configured.", ex_type="ConfigurationError")

genai.configure(api_key=GEMINI_API_KEY)
logger.info("Gemini API configured successfully.")

MODELS_DIR: Path = Path('./models/')
FEATURE_NAMES_PATH: Path = Path('./data/processed/feature_names.txt')
DATA_DIR: Path = Path('./data/processed/') # Used for loading data_after_eda.csv for categories

# --- Singleton for Model Loading ---
class ChurnModelLoader:
    MODEL_VERSION: str = "1.0.0" # Hardcoded version for now
    _instance: Optional["ChurnModelLoader"] = None
    _scaler: Any = None
    _model: Any = None
    _feature_names: Optional[List[str]] = None

    def __new__(cls) -> "ChurnModelLoader":
        """
        Implements the Singleton pattern for ChurnModelLoader.
        Ensures that model and feature names are loaded only once.

        Returns:
            ChurnModelLoader: The single instance of the ChurnModelLoader.
        """
        if cls._instance is None:
            cls._instance = super(ChurnModelLoader, cls).__new__(cls)
            cls._instance._load_models_and_features()
        return cls._instance

    def _load_models_and_features(self) -> None:
        """
        Loads the ML model (scaler and predictor) and feature names from disk.
        This method is called only once during the instantiation of the singleton.

        Raises:
            ChurnException: If the model file or feature names file cannot be found
                            or if there's an error loading its components.
        """
        logger.info("Loading ML model and scaler...")
        try:
            combined_pipeline = joblib.load(MODELS_DIR / 'churn_model.pkl')
            self._scaler = combined_pipeline['scaler']
            self._model = combined_pipeline['model']
            logger.info("ML model and scaler loaded successfully.")
        except FileNotFoundError:
            detail = f"churn_model.pkl not found at {MODELS_DIR}. Please ensure 'save_pipeline.py' was run."
            logger.error(detail)
            raise ChurnException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                 detail=detail, ex_type="ModelLoadingError")
        except KeyError as e:
            detail = f"Error loading components from churn_model.pkl: {e}. Ensure 'scaler' and 'model' keys exist in the pickled dictionary."
            logger.error(detail)
            raise ChurnException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                 detail=detail, ex_type="ModelLoadingError")
        except Exception as e:
            detail = f"An unexpected error occurred while loading churn_model.pkl: {e}"
            logger.error(detail)
            raise ChurnException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                 detail=detail, ex_type="ModelLoadingError")

        logger.info("Loading feature names...")
        try:
            self._feature_names = self._load_feature_names_from_file(FEATURE_NAMES_PATH)
            if not self._feature_names:
                detail = f"No feature names loaded from {FEATURE_NAMES_PATH}. Check file content and format."
                logger.error(detail)
                raise ChurnException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                     detail=detail, ex_type="FeatureLoadingError")
            logger.info(f"Loaded {len(self._feature_names)} feature names.")
        except Exception as e:
            detail = f"Error loading feature names from {FEATURE_NAMES_PATH}: {e}"
            logger.error(detail)
            raise ChurnException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                 detail=detail, ex_type="FeatureLoadingError")

    def _load_feature_names_from_file(self, file_path: Path) -> List[str]:
        """
        Loads feature names from a text file, skipping header and formatting lines.

        Args:
            file_path (Path): The path to the file containing feature names.

        Returns:
            List[str]: A list of feature names in the order they should be used by the model.
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        feature_list: List[str] = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line and not stripped_line.startswith('==') and not stripped_line.startswith('Feature'):
                try:
                    parts = stripped_line.split('.', 1)
                    if len(parts) > 1:
                        feature_name = parts[1].strip()
                        if feature_name:
                            feature_list.append(feature_name)
                except ValueError:
                    continue
        return feature_list

    @property
    def scaler(self) -> Any:
        """
        Returns the loaded StandardScaler object.

        Returns:
            Any: The fitted StandardScaler object.
        """
        return self._scaler

    @property
    def model(self) -> Any:
        """
        Returns the loaded ML prediction model.

        Returns:
            Any: The trained ML model (e.g., LogisticRegression).
        """
        return self._model
    
    @property
    def feature_names(self) -> List[str]:
        """
        Returns the list of feature names in the order expected by the model.

        Returns:
            List[str]: The list of feature names.

        Raises:
            ChurnException: If feature names were not loaded correctly during initialization.
        """
        if self._feature_names is None:
            # This should ideally not happen if __new__ is correctly loading
            raise ChurnException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                                 detail="Feature names not loaded.", ex_type="InternalError")
        return self._feature_names

# Initialize model loader
try:
    model_loader = ChurnModelLoader()
    scaler = model_loader.scaler
    model = model_loader.model
    FEATURE_NAMES = model_loader.feature_names
except ChurnException as e:
    # ChurnException already logged, just re-raise or handle as fatal
    logger.critical(f"Fatal error during application startup: {e.detail}")
    sys.exit(1) # Exit if model loading fails at startup


# --- Define Input Pydantic Model ---
class CustomerData(BaseModel):
    gender: str = Field(..., description="Customer's gender (Male/Female)")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Is customer a senior citizen? (0 or 1)")
    Partner: str = Field(..., description="Does customer have a partner? (Yes/No)")
    Dependents: str = Field(..., description="Does customer have dependents? (Yes/No)")
    tenure: int = Field(..., ge=0, description="Number of months customer has stayed with the company")
    PhoneService: str = Field(..., description="Does customer have phone service? (Yes/No)")
    MultipleLines: str = Field(..., description="Does customer have multiple lines? (Yes/No/No phone service)")
    InternetService: str = Field(..., description="Customer's internet service type (DSL/Fiber optic/No)")
    OnlineSecurity: str = Field(..., description="Does customer have online security? (No/Yes/No internet service)")
    OnlineBackup: str = Field(..., description="Does customer have online backup? (No/Yes/No internet service)")
    DeviceProtection: str = Field(..., description="Does customer have device protection? (No/Yes/No internet service)")
    TechSupport: str = Field(..., description="Does customer have tech support? (No/Yes/No internet service)")
    StreamingTV: str = Field(..., description="Does customer have streaming TV? (No/Yes/No internet service)")
    StreamingMovies: str = Field(..., description="Does customer have streaming movies? (No/Yes/No internet service)")
    Contract: str = Field(..., description="Customer's contract type (Month-to-month/One year/Two year)")
    PaperlessBilling: str = Field(..., description="Does customer have paperless billing? (Yes/No)")
    PaymentMethod: str = Field(..., description="Customer's payment method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic))")
    MonthlyCharges: float = Field(..., gt=0, description="Customer's monthly charges")
    TotalCharges: Optional[float] = Field(None, ge=0, description="Customer's total charges (optional, will be calculated from MonthlyCharges * tenure if not provided)") 

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn and generates personalized marketing offers using Gemini AI.",
    version=ChurnModelLoader.MODEL_VERSION
)

# --- CORS Middleware Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(ChurnException)
async def churn_exception_handler(request: Request, exc: ChurnException) -> JSONResponse:
    """
    Handles ChurnException, logging the error and returning a standardized JSON response.

    Args:
        request (Request): The incoming FastAPI request.
        exc (ChurnException): The exception that was raised.

    Returns:
        JSONResponse: A FastAPI JSON response with the error details.
    """
    logger.error(f"Handled ChurnException: {exc.ex_type} - {exc.detail} (Status: {exc.status_code})")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


# --- Preprocessing Function (Replicates logic from 02_feature_engineering.ipynb) ---
async def preprocess_data(raw_data: CustomerData) -> pd.DataFrame:
    """
    Preprocesses raw customer data by applying feature engineering, encoding, and scaling
    steps as defined in the 02_feature_engineering.ipynb notebook.

    Args:
        raw_data (CustomerData): The raw customer data received from the API or CLI.

    Returns:
        pd.DataFrame: A DataFrame with features transformed and scaled, ready for ML model inference.

    Raises:
        ChurnException: If there's an issue loading reference data for preprocessing (e.g., data_after_eda.csv).
    """
    logger.info(f"Preprocessing data for customer: {raw_data.model_dump_json()}")
    df = pd.DataFrame([raw_data.model_dump()])

    # --- Data Cleaning ---
    if df['TotalCharges'].iloc[0] is None:
        df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']
        df.loc[df['tenure'] == 0, 'TotalCharges'] = 0 
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True) 

    # --- Feature Engineering ---
    bins = [0, 12, 24, 48, 72]
    labels = ['0-12', '12-24', '24-48', '48+']
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=bins,
        labels=labels,
        right=False
    ).astype(str)
    
    df['is_new_customer'] = (df['tenure'] <= 12).astype(int)
    df['is_long_term'] = (df['tenure'] > 24).astype(int)
    
    df['tenure_bins'] = pd.cut(df['tenure'], bins=8, labels=False, right=False).astype(int)
    df['tenure_bins'] = df['tenure_bins'].fillna(0).astype(int) 

    # 2. Charge-based features
    df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'].replace(0, np.nan) + 1)
    df['charge_per_tenure'] = df['MonthlyCharges'] / (df['tenure'].replace(0, np.nan) + 1)
    
    df['avg_monthly_spend'].fillna(0, inplace=True)
    df['charge_per_tenure'].fillna(0, inplace=True)

    df['price_increase'] = (df['MonthlyCharges'] > df['avg_monthly_spend']).astype(int)
    
    try:
        sample_df = pd.read_csv(DATA_DIR / 'data_after_eda.csv')
        sample_df['TotalCharges'] = pd.to_numeric(sample_df['TotalCharges'], errors='coerce')
        sample_df['TotalCharges'].fillna(0, inplace=True)
        sample_df['avg_monthly_spend'] = sample_df['TotalCharges'] / (sample_df['tenure'].replace(0, np.nan) + 1).fillna(1)
        monthly_median_ref: float = sample_df['MonthlyCharges'].median()
        logger.info("Successfully loaded data_after_eda.csv for reference median.")
    except Exception as e:
        logger.warning(f"Could not load data_after_eda.csv to get reference median: {e}. Using hardcoded median.")
        monthly_median_ref = 69.5

    df['high_monthly_charge'] = (df['MonthlyCharges'] > monthly_median_ref).astype(int)

    # 3. Service-based features
    service_cols: List[str] = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                               'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['total_services'] = 0
    for col in service_cols:
        if col in df.columns: 
            df['total_services'] += (df[col] == 'Yes').astype(int)

    df['has_internet'] = (df['InternetService'] != 'No').astype(int)
    df['has_phone'] = (df['PhoneService'] == 'Yes').astype(int)
    
    df['has_tech_support'] = (df['TechSupport'] == 'Yes').astype(int)
    df['has_online_security'] = (df['OnlineSecurity'] == 'Yes').astype(int)


    # 4. Contract & Payment Features
    df['is_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)
    auto_payment_methods: List[str] = ['Bank transfer (automatic)', 'Credit card (automatic)']
    df['has_auto_payment'] = df['PaymentMethod'].isin(auto_payment_methods).astype(int)
    df['is_paperless'] = (df['PaperlessBilling'] == 'Yes').astype(int)

    # 5. Demographic Features
    df['is_senior'] = df['SeniorCitizen']
    has_partner: pd.Series = (df['Partner'] == 'Yes').astype(int)
    has_dependents: pd.Series = (df['Dependents'] == 'Yes').astype(int)
    df['has_family'] = ((has_partner + has_dependents) > 0).astype(int)
    df['is_single_no_deps'] = (
        (df['Partner'] == 'No') & 
        (df['Dependents'] == 'No')
    ).astype(int)


    # --- Encode Categorical Variables ---
    notebook_binary_cols: List[str] = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in notebook_binary_cols:
        if col in df.columns:
            if col == 'gender':
                df[col] = df[col].map({'Male': 1, 'Female': 0}).fillna(df[col].mode()[0]).astype(int) 
            else: 
                df[col] = (df[col] == 'Yes').astype(int)
    
    cols_for_onehot_encoding: List[str] = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaymentMethod', 'tenure_group'
    ]

    cols_for_onehot_encoding = [col for col in cols_for_onehot_encoding if col in df.columns]
    
    df_processed: pd.DataFrame = pd.get_dummies(df, columns=cols_for_onehot_encoding, drop_first=False)

    # --- Align columns with FEATURE_NAMES from training ---
    for col in FEATURE_NAMES:
        if col not in df_processed.columns:
            df_processed[col] = 0
            
    processed_df: pd.DataFrame = df_processed[FEATURE_NAMES]

    logger.info("Data preprocessing completed.")
    return processed_df

# --- Prediction Function ---
async def predict_churn(processed_data: pd.DataFrame) -> float:
    """
    Predicts the churn probability for a given processed customer data.

    Args:
        processed_data (pd.DataFrame): A DataFrame containing the preprocessed
                                       and feature-engineered customer data.

    Returns:
        float: The predicted churn probability (a value between 0 and 1).

    Raises:
        ChurnException: If an error occurs during the model prediction phase.
    """
    logger.info("Predicting churn probability...")
    try:
        churn_probability: float = float(model.predict_proba(processed_data)[:, 1][0])
        logger.info(f"Churn probability predicted: {churn_probability:.2%}")
        return churn_probability
    except Exception as e:
        detail = f"Error during churn prediction: {e}"
        logger.error(detail)
        raise ChurnException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                             detail=detail, ex_type="PredictionError")


# --- Gemini Offer Generation Function ---
async def generate_marketing_offer(churn_probability: float, customer_data: CustomerData) -> str:
    """
    Generates a personalized marketing offer using Google's Gemini AI.
    The offer is based on the predicted churn probability and the customer's profile.

    Args:
        churn_probability (float): The predicted churn probability for the customer.
        customer_data (CustomerData): The raw customer data used for prediction.

    Returns:
        str: A personalized marketing offer generated by Gemini.

    Raises:
        ChurnException: If an error occurs during the Gemini API call (only in API mode).
    """
    logger.info(f"Generating marketing offer for churn probability: {churn_probability:.2%}")
    senior_citizen_status: str = "Yes" if customer_data.SeniorCitizen == 1 else "No"

    prompt: str = f"""
    The customer has the following characteristics:
    - Gender: {customer_data.gender}
    - Senior Citizen: {senior_citizen_status}
    - Partner: {customer_data.Partner}
    - Dependents: {customer_data.Dependents}
    - Tenure (months): {customer_data.tenure}
    - Phone Service: {customer_data.PhoneService}
    - Multiple Lines: {customer_data.MultipleLines}
    - Internet Service: {customer_data.InternetService}
    - Online Security: {customer_data.OnlineSecurity}
    - Online Backup: {customer_data.OnlineBackup}
    - Device Protection: {customer_data.DeviceProtection}
    - Tech Support: {customer_data.TechSupport}
    - Streaming TV: {customer_data.StreamingTV}
    - Streaming Movies: {customer_data.StreamingMovies}
    - Contract Type: {customer_data.Contract}
    - Paperless Billing: {customer_data.PaperlessBilling}
    - Payment Method: {customer_data.PaymentMethod}
    - Monthly Charges: ${customer_data.MonthlyCharges:.2f}
    - Total Charges: ${customer_data.TotalCharges if customer_data.TotalCharges is not None else 0.0:.2f}

    The predicted churn probability for this customer is {churn_probability:.2%}.

    Based on these details, please generate a personalized marketing offer aimed at retaining this customer.
    Consider their services, contract type, charges, and churn probability.
    The offer should be concise, persuasive, and suggest specific actions or benefits.
    If the churn probability is high (e.g., above 50%), focus on strong retention offers.
    If the churn probability is low, perhaps suggest value-added services or loyalty programs.
    Present the offer directly.
    """

    try:
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        response = await gemini_model.generate_content_async(prompt)
        marketing_offer: str = response.text
        logger.info("Marketing offer generated successfully by Gemini.")
        return marketing_offer
    except Exception as e:
        detail = f"Gemini API error during offer generation: {e}"
        logger.error(detail)
        if sys.argv[1:] and "--cli" in sys.argv[1:]:
            return "Error generating marketing offer."
        else:
            raise ChurnException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                 detail=detail, ex_type="GeminiAPIError")


# --- Health Check Endpoint ---
@app.get("/health", response_model=Dict[str, Any]) # Changed to Any because model_version is str
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint to verify server status and model version.

    Returns:
        Dict[str, Any]: A dictionary containing the health status, model version, and current timestamp.
    """
    logger.info("Health check endpoint accessed.")
    return {"status": "healthy", "model_version": ChurnModelLoader.MODEL_VERSION, "timestamp": time.time()}


# --- API Endpoint ---
@app.post("/predict")
async def predict_churn_endpoint(customer_data: CustomerData) -> Dict[str, Any]:
    """
    FastAPI endpoint to predict customer churn probability and generate a personalized
    marketing offer.

    Args:
        customer_data (CustomerData): The incoming customer data for prediction.

    Returns:
        Dict[str, Any]: A dictionary containing the churn probability, marketing offer,
                        and metadata including model version and inference latency.

    Raises:
        ChurnException: If any error occurs during preprocessing, prediction, or Gemini API call.
    """
    logger.info("Received API request for churn prediction.")
    start_time: float = time.time()
    try:
        processed_df: pd.DataFrame = await preprocess_data(customer_data)
        churn_prob: float = await predict_churn(processed_df)
        marketing_offer: str = await generate_marketing_offer(churn_prob, customer_data)

        end_time: float = time.time()
        latency_ms: float = round((end_time - start_time) * 1000, 2)

        logger.info("API request completed successfully.")
        return {
            "churn_probability": churn_prob,
            "marketing_offer": marketing_offer,
            "metadata": {
                "model_version": ChurnModelLoader.MODEL_VERSION,
                "latency_ms": latency_ms
            }
        }
    except ChurnException:
        raise
    except Exception as e:
        detail = f"An unexpected error occurred in API endpoint: {e}"
        logger.error(detail)
        raise ChurnException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                             detail=detail, ex_type="UnexpectedAPIError")

# --- New CLI Result Display Function ---
def display_results_cli(churn_prob: float, marketing_offer: str, customer_data: CustomerData) -> None:
    """
    Displays the churn prediction results and marketing offer in a formatted way
    to the console using Rich library.

    Args:
        churn_prob (float): The predicted churn probability.
        marketing_offer (str): The personalized marketing offer from Gemini.
        customer_data (CustomerData): The raw customer data provided.
    """
    console.print(Panel("[bold yellow]Customer Churn Prediction Result[/bold yellow]", expand=False))

    risk_color: str = "bold red" if churn_prob > 0.5 else "bold green"
    risk_text: str = "High Churn Risk" if churn_prob > 0.5 else "Low Churn Risk"
    churn_text: Text = Text(f"Churn Probability: {churn_prob:.2%} ({risk_text})", style=risk_color)
    console.print(churn_text)

    table = Table(title="Customer Profile", show_header=True, header_style="bold blue")
    table.add_column("Feature", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for field, value in customer_data.model_dump().items():
        if field == "SeniorCitizen":
            table.add_row(field, "Yes" if value == 1 else "No")
        elif field == "TotalCharges" and value is None: 
            table.add_row(field, "Calculated")
        else:
            table.add_row(field, str(value))
    
    console.print(table)

    console.print(Panel(marketing_offer, title="[bold green]Personalized Marketing Offer[/bold green]", border_style="green"))
    logger.info("CLI results displayed to console.")


# --- Argparse Setup for CLI ---
parser = argparse.ArgumentParser(description="Customer Churn Prediction API or CLI tool.")
parser.add_argument("--cli", action="store_true", help="Run in Command Line Interface mode.")

parser.add_argument("--gender", type=str, choices=["Male", "Female"], help="Customer's gender (Male/Female)", required=False)
parser.add_argument("--senior_citizen", type=int, choices=[0, 1], help="Is customer a senior citizen? (0 or 1)", required=False)
parser.add_argument("--partner", type=str, choices=["Yes", "No"], help="Does customer have a partner? (Yes/No)", required=False)
parser.add_argument("--dependents", type=str, choices=["Yes", "No"], help="Does customer have dependents? (Yes/No)", required=False)
parser.add_argument("--tenure", type=int, help="Number of months customer has stayed with the company", required=False)
parser.add_argument("--phone_service", type=str, choices=["Yes", "No"], help="Does customer have phone service? (Yes/No)", required=False)
parser.add_argument("--multiple_lines", type=str, choices=["No phone service", "No", "Yes"], help="Does customer have multiple lines? (Yes/No/No phone service)", required=False)
parser.add_argument("--internet_service", type=str, choices=["DSL", "Fiber optic", "No"], help="Customer's internet service type (DSL/Fiber optic/No)", required=False)
parser.add_argument("--online_security", type=str, choices=["No", "Yes", "No internet service"], help="Does customer have online security? (No/Yes/No internet service)", required=False)
parser.add_argument("--online_backup", type=str, choices=["No", "Yes", "No internet service"], help="Does customer have online backup? (No/Yes/No internet service)", required=False)
parser.add_argument("--device_protection", type=str, choices=["No", "Yes", "No internet service"], help="Does customer have device protection? (No/Yes/No internet service)", required=False)
parser.add_argument("--tech_support", type=str, choices=["No", "Yes", "No internet service"], help="Does customer have tech support? (No/Yes/No internet service)", required=False)
parser.add_argument("--streaming_tv", type=str, choices=["No", "Yes", "No internet service"], help="Does customer have streaming TV? (No/Yes/No internet service)", required=False)
parser.add_argument("--streaming_movies", type=str, choices=["No", "Yes", "No internet service"], help="Does customer have streaming movies? (No/Yes/No internet service)", required=False)
parser.add_argument("--contract", type=str, choices=["Month-to-month", "One year", "Two year"], help="Customer's contract type (Month-to-month/One year/Two year)", required=False)
parser.add_argument("--paperless_billing", type=str, choices=["Yes", "No"], help="Does customer have paperless billing? (Yes/No)", required=False)
parser.add_argument("--payment_method", type=str, choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], help="Customer's payment method", required=False)
parser.add_argument("--monthly_charges", type=float, help="Customer's monthly charges", required=False)
parser.add_argument("--total_charges", type=float, default=None, help="Customer's total charges (optional, will be calculated from MonthlyCharges * tenure if not provided)")


# --- Main execution block ---
if __name__ == "__main__":
    args = parser.parse_args()

    # Create a dummy CustomerData instance to extract required fields
    dummy_customer_data_schema = CustomerData.model_json_schema()
    cli_required_fields = [
        field for field, properties in dummy_customer_data_schema.get('properties', {}).items()
        if field in dummy_customer_data_schema.get('required', [])
    ]
    if 'TotalCharges' in cli_required_fields:
        cli_required_fields.remove('TotalCharges')


    if args.cli:
        logger.info("Running in CLI mode.")
        # Direct mapping from Pydantic field names to argparse argument names
        field_to_arg = {
            'gender': 'gender', 'SeniorCitizen': 'senior_citizen', 'Partner': 'partner',
            'Dependents': 'dependents', 'tenure': 'tenure', 'PhoneService': 'phone_service',
            'MultipleLines': 'multiple_lines', 'InternetService': 'internet_service',
            'OnlineSecurity': 'online_security', 'OnlineBackup': 'online_backup',
            'DeviceProtection': 'device_protection', 'TechSupport': 'tech_support',
            'StreamingTV': 'streaming_tv', 'StreamingMovies': 'streaming_movies',
            'Contract': 'contract', 'PaperlessBilling': 'paperless_billing',
            'PaymentMethod': 'payment_method', 'MonthlyCharges': 'monthly_charges'
        }
        missing_args: List[str] = []
        for field_name in cli_required_fields:
            arg_name = field_to_arg.get(field_name, field_name.lower())
            if getattr(args, arg_name, None) is None:
                missing_args.append(f"--{arg_name}")
        
        if missing_args:
            error_msg: str = f"Missing required CLI arguments: {', '.join(missing_args)}. All fields except --total_charges are required."
            console.print(Panel(f"[bold red]Error: {error_msg}[/bold red]",
                                title="[bold red]CLI Error[/bold red]", border_style="red"))
            parser.print_help()
            sys.exit(1)
        
        try:
            customer_data_cli: CustomerData = CustomerData(
                gender=args.gender,
                SeniorCitizen=args.senior_citizen,
                Partner=args.partner,
                Dependents=args.dependents,
                tenure=args.tenure,
                PhoneService=args.phone_service,
                MultipleLines=args.multiple_lines,
                InternetService=args.internet_service,
                OnlineSecurity=args.online_security,
                OnlineBackup=args.online_backup,
                DeviceProtection=args.device_protection,
                TechSupport=args.tech_support,
                StreamingTV=args.streaming_tv,
                StreamingMovies=args.streaming_movies,
                Contract=args.contract,
                PaperlessBilling=args.paperless_billing,
                PaymentMethod=args.payment_method,
                MonthlyCharges=args.monthly_charges,
                TotalCharges=args.total_charges
            )
        except Exception as e:
            error_msg = f"Error validating input data for CLI: {e}"
            console.print(Panel(f"[bold red]Error: {error_msg}[/bold red]",
                                title="[bold red]Input Validation Error[/bold red]", border_style="red"))
            logger.error(error_msg)
            sys.exit(1)


        try:
            import asyncio
            
            async def run_cli_tasks() -> None:
                # Inner async function to run the CLI prediction and display tasks.
                processed_df: pd.DataFrame = await preprocess_data(customer_data_cli)
                churn_prob: float = await predict_churn(processed_df)
                marketing_offer: str = await generate_marketing_offer(churn_prob, customer_data_cli)
                display_results_cli(churn_prob, marketing_offer, customer_data_cli)

            asyncio.run(run_cli_tasks())

        except ChurnException:
            logger.critical("CLI execution failed due to a ChurnException (details logged above).")
            sys.exit(1)
        except Exception as e:
            error_msg = f"An unexpected error occurred during CLI execution: {e}"
            console.print(Panel(f"[bold red]Error: {error_msg}[/bold red]",
                                title="[bold red]CLI Error[/bold red]", border_style="red"))
            logger.critical(error_msg)
            sys.exit(1)
    else:
        logger.info("Starting FastAPI server.")
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
