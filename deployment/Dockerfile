# Multi-stage Dockerfile for Consumer Segmentation Analysis
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed logs cache config

# Set development environment
ENV ENVIRONMENT=development

# Expose ports
EXPOSE 8501 8888

# Default command for development
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Production stage
FROM base as production

# Copy only necessary files
COPY src/ ./src/
COPY config.py .
COPY pipeline_manager.py .
COPY dashboard.py .
COPY run_dashboard.py .

# Create necessary directories
RUN mkdir -p data/raw data/processed logs cache config

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app

USER app

# Set production environment
ENV ENVIRONMENT=production

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Production command
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

# Testing stage
FROM development as testing

# Copy test files
COPY test_suite.py .
COPY tests/ ./tests/

# Run tests
RUN python -m pytest test_suite.py -v --tb=short

# Default command for testing
CMD ["python", "-m", "pytest", "test_suite.py", "-v"]