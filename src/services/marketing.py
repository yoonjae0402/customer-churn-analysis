
from typing import Dict, Any, Optional
from google import genai
import logging
from src.config import config

logger = logging.getLogger(__name__)

class MarketingService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.gemini_api_key
        if not self.api_key:
            logger.warning("Gemini API key not configured. Marketing offers will not be generated.")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)

    async def generate_offer(self, churn_probability: float, customer_data: Dict[str, Any]) -> str:
        """
        Generates a personalized marketing offer using Gemini.
        """
        if not self.client:
            return "Marketing offer generation unavailable (API key missing)."
            
        try:
            senior_citizen_status = "Yes" if customer_data.get("SeniorCitizen") == 1 else "No"
            
            prompt = f"""
            The customer has the following characteristics:
            - Gender: {customer_data.get("gender")}
            - Senior Citizen: {senior_citizen_status}
            - Partner: {customer_data.get("Partner")}
            - Dependents: {customer_data.get("Dependents")}
            - Tenure (months): {customer_data.get("tenure")}
            - Phone Service: {customer_data.get("PhoneService")}
            - Multiple Lines: {customer_data.get("MultipleLines")}
            - Internet Service: {customer_data.get("InternetService")}
            - Online Security: {customer_data.get("OnlineSecurity")}
            - Online Backup: {customer_data.get("OnlineBackup")}
            - Device Protection: {customer_data.get("DeviceProtection")}
            - Tech Support: {customer_data.get("TechSupport")}
            - Streaming TV: {customer_data.get("StreamingTV")}
            - Streaming Movies: {customer_data.get("StreamingMovies")}
            - Contract Type: {customer_data.get("Contract")}
            - Paperless Billing: {customer_data.get("PaperlessBilling")}
            - Payment Method: {customer_data.get("PaymentMethod")}
            - Monthly Charges: ${customer_data.get("MonthlyCharges", 0):.2f}
            
            The predicted churn probability for this customer is {churn_probability:.2%}.
            
            Based on these details, please generate a personalized marketing offer aimed at retaining this customer.
            Consider their services, contract type, charges, and churn probability.
            The offer should be concise, persuasive, and suggest specific actions or benefits.
            If the churn probability is high (e.g., above 50%), focus on strong retention offers.
            If the churn probability is low, perhaps suggest value-added services or loyalty programs.
            Present the offer directly.
            """

            response = await self.client.aio.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return "Error generating marketing offer."

marketing_service = MarketingService()
