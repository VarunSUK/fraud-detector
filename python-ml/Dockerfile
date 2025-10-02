# Multi-stage Dockerfile for Python ML training pipeline
FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY train.py .
COPY notebooks/ ./notebooks/

# Create directories for data and models
RUN mkdir -p data models output

# Create non-root user
RUN useradd --create-home --shell /bin/bash mluser && \
    chown -R mluser:mluser /app
USER mluser

# Expose port for Jupyter notebook (optional)
EXPOSE 8888

# Default command
CMD ["python", "train.py", "--help"]

# Development stage
FROM base as development

# Install development dependencies
USER root
RUN pip install --no-cache-dir \
    jupyter \
    ipykernel \
    pytest \
    pytest-cov \
    black \
    flake8

USER mluser

# Override default command for development
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production stage
FROM base as production

# Copy any additional production files
COPY docker/entrypoint.sh .
RUN chmod +x entrypoint.sh

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Use entrypoint script
ENTRYPOINT ["./entrypoint.sh"]



