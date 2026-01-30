# 📈 Customer Churn Prediction & Personalized Marketing Assistant

## Project Overview

This project develops a comprehensive solution for predicting customer churn and generating personalized retention marketing offers. It leverages traditional Machine Learning (ML) models for robust churn probability prediction and integrates with a Large Language Model (LLM) – Google's Gemini API – to craft dynamic, context-aware marketing messages. The solution is accessible via both a FastAPI web API and a powerful Command Line Interface (CLI).

## ✨ Features

- **Churn Prediction:** Utilizes a Logistic Regression model to predict customer churn probability based on various customer attributes.
- **Personalized Offers (powered by Gemini AI):** Generates tailored marketing offers to retain at-risk customers, considering their specific profile and churn likelihood.
- **Scalable API:** A FastAPI endpoint (`/predict`) for real-time churn prediction and marketing offer generation.
- **Interactive CLI:** A command-line interface for quick predictions and offer generation, ideal for testing or ad-hoc analysis.
- **Secure API Key Handling:** Integrates `python-dotenv` for securely loading API keys from `.env` files, preventing hardcoding.
- **Containerized Deployment:** Dockerfile provided for easy, consistent deployment across different environments.

## 🚀 Architecture

The system's architecture is designed to integrate predictive analytics with generative AI, providing a seamless flow from raw customer data to actionable insights.

```mermaid
graph TD
    A[User Request (API / CLI)] --> B(FastAPI Endpoint / CLI Input)
    B --> C{Pydantic Validation}
    C --> D{Preprocessing Module}
    D --> E[Feature Engineered Data]
    E --> F(ML Churn Prediction Model)
    F --> G[Churn Probability]
    G --> H(Gemini API: Offer Generation)
    H --> I[Personalized Marketing Offer]
    G & I --> J[Final Response / CLI Output]
```

**Flow Description:**
1.  **User Request** originates from either the FastAPI `/predict` endpoint or the CLI.
2.  The input data undergoes **Pydantic Validation** to ensure correctness and adherence to defined constraints.
3.  The **Preprocessing Module** (derived from `02_feature_engineering.ipynb`) transforms this raw data. This involves data cleaning, creating engineered features (e.g., tenure groups, charge-based features, service usage counts), and encoding categorical variables (one-hot encoding) and scaling numerical features using a pre-trained `StandardScaler`.
4.  The **ML Churn Prediction Model** (a pre-trained Logistic Regression model loaded from `churn_model.pkl`) takes the preprocessed data and calculates the **Churn Probability**.
5.  The **Churn Probability** along with the original **Raw Customer Data** is then sent to **Gemini AI** for **Marketing Offer Generation**.
6.  **Gemini AI** analyzes the customer profile and churn risk to produce a **Personalized Marketing Offer**.
7.  Finally, the **Prediction Result & Offer** (churn probability and marketing message) is returned to the user via the API response or displayed in the CLI.

## 💡 Technical Rationale: Why Combine LLM with ML?

The synergy between traditional Machine Learning (ML) models and Large Language Models (LLMs) like Gemini offers a powerful solution that transcends the capabilities of either technology alone.

-   **ML for Quantitative Precision:** Our Logistic Regression model is highly effective at identifying subtle patterns in structured customer data to accurately quantify churn risk. It provides a precise, data-driven probability score, which is a core analytical strength of ML. This numerical output is critical for understanding the *likelihood* of churn.

-   **LLM for Qualitative Nuance & Actionability:** While the ML model tells us *if* a customer is likely to churn and by *how much*, it doesn't inherently suggest *what to do* about it in a human-friendly way. This is where Gemini shines. LLMs are trained on vast amounts of text data, allowing them to understand context, synthesize information, and generate creative, coherent, and persuasive human-like language. By feeding Gemini the churn probability and the customer's detailed profile, we enable it to:
    *   **Contextualize:** Understand the specific services, contract terms, and demographic factors influencing the customer's situation.
    *   **Personalize:** Generate an offer that speaks directly to the customer's likely needs or pain points, making it far more impactful than a generic message.
    *   **Recommend Actionable Strategies:** Suggest specific discounts, service upgrades, contract modifications, or support options that are relevant to the customer's profile and designed to mitigate their churn risk.

**In essence, the ML model provides the "what" (churn risk), and the LLM translates this "what" into an intelligent, personalized, and actionable "how" (retention strategy).** This combination transforms a raw prediction into a directly usable business tool.

## 🛠️ Technical Challenges & Solutions

### Asynchronous Processing for Latency Optimization

A key consideration for building a responsive real-time prediction and generation service is managing latency, especially when integrating external APIs like Google's Gemini. Traditional synchronous processing would block the server while waiting for I/O operations (e.g., calling the Gemini API, reading/writing files). This significantly degrades performance and limits the number of concurrent requests the server can handle.

**Solution:** This project is built using Python's `asyncio` framework, FastAPI, and `await`able operations to enable highly concurrent and efficient request handling.

-   **FastAPI's `async def` Endpoints:** FastAPI is inherently designed for asynchronous operations. All API endpoints (`/predict`, `/health`) and core logic functions (`preprocess_data`, `predict_churn`, `generate_marketing_offer`) are defined as `async def`.
-   **Non-Blocking I/O:** When the application makes an external call (e.g., `await model_gemini.generate_content_async(prompt)` to the Gemini API) or performs an I/O-bound task, it `awaits` the result. During this `await` period, instead of blocking the entire server, FastAPI can switch to processing other incoming requests.
-   **Improved Concurrency:** This pattern allows the server to handle many simultaneous user requests efficiently, maximizing throughput and reducing perceived latency, even when individual requests involve time-consuming external API calls.
-   **Model Loading as Singleton:** Furthermore, the ML model and scaler are loaded only once at application startup using a Singleton pattern. This prevents redundant disk I/O for each request, contributing to lower latency for inference.

This asynchronous design ensures that the application remains fast and responsive under load, providing a smooth user experience.

## ⚙️ Installation

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yunjaejung/customer-churn-analysis.git
    cd customer-churn-analysis
    ```

2.  **Create and activate a virtual environment:**
    It's recommended to use a virtual environment to manage dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r app/requirements.txt
    ```

4.  **Set up Gemini API Key:**
    Obtain your Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).
    Create a `.env` file in the `app/` directory and add your API key:
    ```
    GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
    ```
    *Refer to `app/.env.example` for the correct format.*

5.  **Prepare ML Artifacts:**
    Ensure that the `app/models/churn_model.pkl` and `app/data/processed/feature_names.txt` files exist. These are generated by running the `save_pipeline.py` script.
    ```bash
    cd app && python3 save_pipeline.py
    ```

## 🚀 Usage

The project can be used in two modes: FastAPI (Web API) and CLI (Command Line Interface).

### 🌐 FastAPI (Web API)

Run the FastAPI application from the `app/` directory:

```bash
cd app && uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Once running, access the interactive API documentation (Swagger UI) at `http://localhost:8000/docs`. You can use this interface to test the `/predict` endpoint.

### 💻 CLI (Command Line Interface)

To use the CLI mode, run from the `app/` directory with the `--cli` argument along with customer data features.

**Example:**
```bash
cd app && python3 main.py --cli \
    --gender Female \
    --senior_citizen 0 \
    --partner Yes \
    --dependents No \
    --tenure 12 \
    --phone_service Yes \
    --multiple_lines Yes \
    --internet_service Fiber optic \
    --online_security No \
    --online_backup Yes \
    --device_protection No \
    --tech_support No \
    --streaming_tv Yes \
    --streaming_movies No \
    --contract Month-to-month \
    --paperless_billing Yes \
    --payment_method Electronic check \
    --monthly_charges 89.90 \
    --total_charges 1078.80
```
*(Note: `--total_charges` is optional. If omitted, it will be calculated from `monthly_charges` and `tenure`.)*

### 🐳 Docker Deployment

To build and run the application using Docker:

1.  **Build the Docker image:**
    ```bash
    docker build -f app/Dockerfile -t customer-churn-app .
    ```

2.  **Run the Docker container (API Mode - Secure API Key Handling):**
    When running your Docker container, it's crucial to pass sensitive information like your `GEMINI_API_KEY` securely as an environment variable. **Never hardcode API keys directly into your Dockerfile or application code.**

    ```bash
    docker run -p 8000:8000 \
        -e GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE" \
        customer-churn-app
    ```
    Replace `"YOUR_GEMINI_API_KEY_HERE"` with your actual Gemini API key. This ensures the key is provided at runtime and not baked into the image.

3.  **Run the Docker container (CLI Mode):**
    You can also run the CLI mode directly within the Docker container by overriding the default `CMD`:

    ```bash
    docker run \
        -e GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE" \
        customer-churn-app python3 main.py --cli \
        --gender Male \
        --senior_citizen 0 \
        --partner No \
        --dependents No \
        --tenure 1 \
        --phone_service Yes \
        --multiple_lines No \
        --internet_service DSL \
        --online_security No \
        --online_backup No \
        --device_protection No \
        --tech_support No \
        --streaming_tv No \
        --streaming_movies No \
        --contract Month-to-month \
        --paperless_billing Yes \
        --payment_method Electronic check \
        --monthly_charges 45.0
    ```
    *(Remember to replace `"YOUR_GEMINI_API_KEY_HERE"` with your actual API key.)*

### 🔒 Enhanced Security with Secret Management Services (e.g., AWS Secrets Manager)

For production deployments, especially in cloud environments, it's highly recommended to integrate with dedicated secret management services rather than passing secrets directly via environment variables in `docker run` commands.

**Why use Secret Management?**
-   **Centralized Control:** Store and manage all your secrets in one secure location.
-   **Improved Security:** Secrets are encrypted at rest and in transit.
-   **Automatic Rotation:** Services can automatically rotate secrets, reducing the risk of compromised long-lived credentials.
-   **Granular Access:** Control who (or what services) can access which secrets with fine-grained permissions.
-   **Auditability:** Track access to secrets for compliance and security auditing.

**Conceptual Integration with AWS Secrets Manager:**

1.  **Store Secret:** Place your `GEMINI_API_KEY` into AWS Secrets Manager.
2.  **IAM Permissions:** Configure an AWS Identity and Access Management (IAM) role (e.g., an ECS Task Role, EKS Pod Identity, or EC2 Instance Profile) with permissions to read the specific secret(s) from Secrets Manager.
3.  **Application Access:**
    *   **Option A (Direct Retrieval in Code):** Modify your application's startup logic (e.g., in `main.py` or a dedicated `config.py`) to use the AWS SDK (Boto3 in Python) to fetch the `GEMINI_API_KEY` from Secrets Manager at runtime. The application would assume the IAM role, which grants it access.
    *   **Option B (Container Orchestration Integration):** If using services like AWS ECS or EKS, these platforms often have native integrations to inject secrets from Secrets Manager directly as environment variables into your container at startup. This approach is generally preferred as it keeps your application code cleaner.
4.  **Deployment:** Deploy your Docker container to an environment (like AWS ECS, EKS) where it can assume the configured IAM role.

This approach significantly hardens the security posture of your application by decoupling sensitive credentials from your deployment commands and code.

--- 

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.