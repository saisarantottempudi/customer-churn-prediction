FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p logs data/raw data/processed models reports/figures

# Expose ports
EXPOSE 8000 8501

# Default command starts API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
