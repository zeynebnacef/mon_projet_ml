# Use a Python base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy necessary files into the container
COPY requirements.txt .
COPY src/ ./src/
COPY tests/ ./tests/
COPY data/ ./data/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for MLflow and PostgreSQL
RUN pip install mlflow psycopg2-binary

# Expose the port for the application
EXPOSE 5000

# Default command to run the application
CMD ["sh", "-c", "mlflow server --backend-store-uri postgresql://mlflow_user:zeyneb@postgres:5432/mlflow_db2 --default-artifact-root /mlflow/artifacts --host 0.0.0.0"]