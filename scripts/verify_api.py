
import subprocess
import time
import requests
import sys
import os

def test_live_api():
    print("Starting API server...")
    # Start the API server as a subprocess
    process = subprocess.Popen(
        ["uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Wait for server to start
        print("Waiting for server to be ready...")
        max_retries = 10
        for i in range(max_retries):
            try:
                print(f"Checking health... (Attempt {i+1})")
                response = requests.get("http://127.0.0.1:8000/health")
                if response.status_code == 200:
                    print("Server is ready!")
                    print(f"Health response: {response.json()}")
                    break
            except requests.ConnectionError:
                time.sleep(2)
        else:
            print("Failed to connect to server.")
            stdout, stderr = process.communicate(timeout=1)
            print("Server Output:", stdout)
            print("Server Error:", stderr)
            sys.exit(1)

        # Make a prediction request
        payload = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": 29.85
        }
        
        print("\nSending prediction request...")
        resp = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if resp.status_code == 200:
            print("Prediction success!")
            data = resp.json()
            print(f"Churn Probability: {data['churn_probability']:.2%}")
            print(f"Offer: {data['marketing_offer'][:50]}...")
        else:
            print(f"Prediction failed with status {resp.status_code}")
            print(resp.text)
            sys.exit(1)
            
    finally:
        print("\nStopping server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("Server stopped.")

if __name__ == "__main__":
    test_live_api()
