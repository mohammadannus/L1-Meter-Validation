# Use a slim Python 3.11 Linux image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for OpenCV and PaddleOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies (CPU-only versions for smaller size)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Explicitly copy the model weights to ensure they are in the image
COPY L1v1-anas.pt /app/L1v1-anas.pt
COPY seal_detection.pt /app/seal_detection.pt

# Expose the port FastAPI runs on
EXPOSE 8000

# Start the application
CMD ["python", "main.py"]