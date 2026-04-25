FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY openenv.yaml ./
COPY models.py ./
COPY client.py ./
COPY __init__.py ./
COPY server/ ./server/
COPY data/ ./data/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    openenv-core>=0.2.3 \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    pydantic>=2.0.0 \
    sentence-transformers>=2.2.0 \
    wikipedia-api>=0.6.0 \
    transformers>=4.35.0 \
    torch>=2.0.0 \
    numpy>=1.24.0 \
    httpx>=0.25.0

# Expose port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
