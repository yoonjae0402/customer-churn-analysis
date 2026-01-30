# Use a slim Python image for a smaller footprint
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Prevent Python from writing .pyc files to disc and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies (if any are needed for Python packages, e.g., for pandas/numpy often no specific ones are needed in slim)
# In this specific case, python:3.9-slim usually has what's needed for the listed libraries.

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the application
# When running the container, ensure to pass the GEMINI_API_KEY as an environment variable.
# Example: docker run -p 8000:8000 -e GEMINI_API_KEY="your_key" customer-churn-app
# For CLI mode: docker run -e GEMINI_API_KEY="your_key" customer-churn-app python main.py --cli ...
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]