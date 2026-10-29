FROM python:3.10-slim

# Prevent python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install essential system dependencies for OpenCV and video handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install dependencies (CPU optimized)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application files and model weights
COPY . .

# Ensure storage and capture directories exist with correct permissions
RUN mkdir -p captures/images captures/videos uploads \
    && chmod -R 777 captures uploads

# Expose default port
EXPOSE 8000

# Healthcheck to ensure container stays healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

# Start the PashuX FastAPI application (respects Render dynamic $PORT)
CMD ["sh", "-c", "uvicorn pashux_web_dashboard:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
