
from pydantic import BaseModel, Field
from typing import Optional

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
    TotalCharges: Optional[float] = Field(None, ge=0, description="Customer's total charges (optional)") 

class PredictionResponse(BaseModel):
    churn_probability: float
    marketing_offer: str
    metadata: dict
