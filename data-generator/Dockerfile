# Multi-stage Dockerfile for data generator
FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create non-root user
RUN useradd --create-home --shell /bin/bash generator && \
    chown -R generator:generator /app
USER generator

# Default command
CMD ["python", "src/stream_producer.py", "--help"]

# Development stage
FROM base as development

# Install development dependencies
USER root
RUN pip install --no-cache-dir \
    jupyter \
    ipykernel \
    pytest \
    pytest-cov

USER generator

# Override default command for development
CMD ["python", "src/stream_producer.py", "--help"]

# Production stage
FROM base as production

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command for production
CMD ["python", "src/stream_producer.py", "--help"]
