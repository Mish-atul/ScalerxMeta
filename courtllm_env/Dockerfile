FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY pyproject.toml ./
COPY openenv.yaml ./
COPY models.py ./
COPY client.py ./
COPY __init__.py ./
COPY inference.py ./
COPY server/ ./server/
COPY data/ ./data/
COPY outputs/ ./outputs/
COPY training/ ./training/
COPY README.md ./

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Run FastAPI server on port 7860 (HF Spaces requirement)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
