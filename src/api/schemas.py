from typing import Literal, Optional

from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    gender: Literal["Male", "Female"] = Field(..., description="Customer's gender")
    SeniorCitizen: int = Field(
        ..., ge=0, le=1, description="Is customer a senior citizen? (0 or 1)"
    )
    Partner: Literal["Yes", "No"] = Field(
        ..., description="Does customer have a partner?"
    )
    Dependents: Literal["Yes", "No"] = Field(
        ..., description="Does customer have dependents?"
    )
    tenure: int = Field(
        ..., ge=0, description="Months customer has stayed with the company"
    )
    PhoneService: Literal["Yes", "No"] = Field(
        ..., description="Does customer have phone service?"
    )
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        ..., description="Does customer have multiple lines?"
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Customer's internet service type"
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Does customer have online security?"
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Does customer have online backup?"
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Does customer have device protection?"
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Does customer have tech support?"
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Does customer have streaming TV?"
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Does customer have streaming movies?"
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Customer's contract type"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Does customer have paperless billing?"
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., description="Customer's payment method")
    MonthlyCharges: float = Field(
        ..., gt=0, description="Customer's monthly charges in USD"
    )
    TotalCharges: Optional[float] = Field(
        None,
        ge=0,
        description="Customer's total charges (computed from tenure if omitted)",
    )


class PredictionResponse(BaseModel):
    churn_probability: float = Field(
        ..., description="Predicted probability of churn (0-1)"
    )
    churn_prediction: int = Field(
        ...,
        description="Binary churn prediction at threshold (1=churn, 0=no churn)",
    )
    marketing_offer: str = Field(
        ..., description="AI-generated personalised retention offer"
    )
    metadata: dict = Field(
        ..., description="model_version, latency_ms, model_type, threshold"
    )
