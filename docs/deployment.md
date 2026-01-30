# Deployment Guide

## Overview

This guide covers deploying the Customer Churn Prediction API to various environments.

---

## Prerequisites

- Python 3.9+
- Docker (for containerized deployment)
- Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

---

## Local Development

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/yoonjae0402/customer-churn-analysis.git
cd customer-churn-analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r app/requirements.txt
```

### 2. Configure Environment Variables

```bash
# Create .env file in app/ directory
echo "GEMINI_API_KEY=your_api_key_here" > app/.env
```

### 3. Prepare Model Artifacts

```bash
cd app && python3 save_pipeline.py
```

### 4. Run Application

```bash
# API Mode
cd app && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# CLI Mode
cd app && python3 main.py --cli --gender Male --tenure 12 ...
```

---

## Docker Deployment

### Build Image

```bash
docker build -f app/Dockerfile -t customer-churn-app .
```

### Run Container

```bash
# API Mode
docker run -p 8000:8000 \
    -e GEMINI_API_KEY="your_api_key_here" \
    customer-churn-app

# CLI Mode
docker run \
    -e GEMINI_API_KEY="your_api_key_here" \
    customer-churn-app python3 main.py --cli \
    --gender Male --tenure 12 ...
```

---

## Cloud Deployment Options

### AWS

#### Option 1: ECS (Elastic Container Service)

1. Push image to ECR:
```bash
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag customer-churn-app:latest <account>.dkr.ecr.<region>.amazonaws.com/customer-churn-app:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/customer-churn-app:latest
```

2. Create ECS Task Definition with:
   - Container image from ECR
   - Environment variable: `GEMINI_API_KEY` from Secrets Manager
   - Port mapping: 8000

3. Create ECS Service with Application Load Balancer

#### Option 2: Lambda + API Gateway

For serverless deployment, adapt the FastAPI app using Mangum:

```python
from mangum import Mangum
handler = Mangum(app)
```

### Google Cloud

#### Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/customer-churn-app

# Deploy
gcloud run deploy customer-churn-api \
    --image gcr.io/PROJECT_ID/customer-churn-app \
    --platform managed \
    --set-env-vars GEMINI_API_KEY=your_key
```

### Azure

#### Container Instances

```bash
az container create \
    --resource-group myResourceGroup \
    --name customer-churn-api \
    --image customer-churn-app:latest \
    --ports 8000 \
    --environment-variables GEMINI_API_KEY=your_key
```

---

## Production Considerations

### Security

1. **API Key Management**
   - Use cloud secret managers (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault)
   - Never commit secrets to version control
   - Rotate keys regularly

2. **Network Security**
   - Use HTTPS/TLS termination
   - Configure CORS appropriately
   - Use API rate limiting

3. **Input Validation**
   - Pydantic models enforce type safety
   - Add additional business rule validation as needed

### Performance

1. **Scaling**
   - Horizontal scaling: multiple container instances
   - Load balancing across instances
   - Auto-scaling based on CPU/memory metrics

2. **Caching**
   - Cache model in memory (already implemented via singleton)
   - Consider Redis for API response caching

3. **Monitoring**
   - Add health check endpoint: `/health`
   - Implement logging (structured JSON logs)
   - Set up alerts for error rates and latency

### Model Updates

1. **Versioning**
   - Tag model files with version numbers
   - Keep previous versions for rollback

2. **Retraining Pipeline**
   - Schedule quarterly model retraining
   - A/B test new models before full deployment

3. **Monitoring Model Drift**
   - Track prediction distribution over time
   - Alert if accuracy degrades significantly

---

## Health Check

The API includes a health endpoint:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Model file not found | Run `python3 save_pipeline.py` to generate artifacts |
| API key error | Verify `.env` file exists and contains valid key |
| Port already in use | Change port: `--port 8001` |
| Memory issues | Increase container memory allocation |

### Logging

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
uvicorn main:app --log-level debug
```

---

## API Reference

### POST /predict

**Request:**
```json
{
  "gender": "Male",
  "senior_citizen": 0,
  "partner": "Yes",
  "dependents": "No",
  "tenure": 12,
  "phone_service": "Yes",
  "multiple_lines": "No",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "online_backup": "Yes",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "streaming_movies": "No",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "monthly_charges": 89.90,
  "total_charges": 1078.80
}
```

**Response:**
```json
{
  "churn_probability": 0.78,
  "risk_level": "HIGH",
  "marketing_offer": "Personalized retention offer..."
}
```
