FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for distributed mode
RUN pip install --no-cache-dir flask requests

# Copy experiment files
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Create output directories
RUN mkdir -p results logs deployments

# Default command
CMD ["python", "scripts/run_fedxchain.py", "--config", "configs/experiment_config.yaml", "--output", "results"]
