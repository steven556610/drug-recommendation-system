# ====================================================================
# BioRec System - Production Dockerfile
# ====================================================================

FROM python:3.10-slim

# Set environment variables for Python optimization and standard outputs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install basic system diagnostic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency descriptions and install packages without caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and assets into the container
COPY . .

# Expose server port
EXPOSE 8000

# Health check to ensure the API and web services are active
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/autocomplete || exit 1

# Launch the FastAPI web server by default on all interfaces
CMD ["python", "run.py", "--mode", "web", "--host", "0.0.0.0", "--port", "8000"]
