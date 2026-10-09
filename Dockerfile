FROM python:3.11-slim

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1

# Working directory inside Docker
WORKDIR /app

# Install Linux packages needed by OpenCV, InsightFace and Torch
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    build-essential \
    portaudio19-dev \
    python3-dev \
    gcc \
    espeak-ng \
    libespeak-ng1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (Docker cache optimization)
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install all Python modules
RUN pip install --no-cache-dir -r requirements.txt

# Copy the complete VisionMate project
COPY . .

# Flask backend port
EXPOSE 8080

# Default command (same project entry point)
CMD ["python", "test/test_integration.py"]