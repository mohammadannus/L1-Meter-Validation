# MeterSerial-VLM | Production Server Deployment Guide

This guide details the complete hardware sizing, system prerequisites, environment setup, and deployment configurations needed to run **MeterSerial-VLM** reliably in a production environment.

---

## 1. Hardware Sizing & System Specifications

MeterSerial-VLM performs two primary computer vision workloads on incoming electricity meter images:
1. **Text Extraction (OCR / VLM)**: Serial Number and IMEI identification using PaddleOCR, LightOnOCR-2-1B, or Qwen2-VL-2B.
2. **Display Region & Green LED Classification**: YOLOv11 object detection (`L1v1-anas.pt`) + HSV chromaticity analysis.

### Hardware Matrix

| Specification | CPU-Only Mode (Minimum) | GPU-Accelerated (Recommended) | High-Throughput Cluster |
| :--- | :--- | :--- | :--- |
| **Target Workload** | Low traffic, development, batch | Production API (10–50 req/min) | High-volume production (>100 req/min) |
| **GPU Model** | None | 1× NVIDIA RTX 4060 Ti / T4 / L4 / A10G | 1× or 2× NVIDIA A10G / A100 / RTX 4090 |
| **GPU VRAM** | N/A | **8 GB – 16 GB** | **24 GB – 80 GB** |
| **System RAM** | **16 GB** (32 GB recommended) | **16 GB – 32 GB** | **64 GB** |
| **CPU / vCPU** | 8 vCPUs (Intel Xeon / AMD EPYC) | 4 – 8 vCPUs | 16+ vCPUs |
| **Disk Storage** | 30 GB SSD | 50 GB NVMe SSD | 100 GB NVMe SSD |
| **Inference Time (PaddleOCR)** | ~8 – 15 seconds (warm) | **~0.6 – 1.5 seconds** | **~0.4 – 0.8 seconds** |
| **Inference Time (LightOnOCR)**| ~25 – 45 seconds | **~1.2 – 2.5 seconds** | **~0.8 – 1.5 seconds** |

### Memory & VRAM Breakdown

- **PaddleOCR Backend (Default)**:
  - System RAM footprint: ~1.5 GB.
  - GPU VRAM footprint: ~1.8 GB – 2.5 GB.
- **LightOnOCR-2-1B Backend**:
  - GPU VRAM footprint: ~3.5 GB – 4.5 GB (fp16 / bfloat16).
- **Qwen2-VL-2B Backend**:
  - GPU VRAM footprint: ~5.5 GB – 7.5 GB (fp16 / bfloat16).
- **YOLOv11 Green Light Detector (`L1v1-anas.pt`)**:
  - GPU VRAM footprint: ~0.5 GB.
- **Combined Peak VRAM (PaddleOCR + YOLO)**: **~2.5 GB – 3.2 GB**.  
  *Therefore, any 8 GB+ NVIDIA GPU handles the default production pipeline with substantial headroom.*

---

## 2. Operating System & Software Prerequisites

- **Recommended OS**: **Ubuntu 22.04 LTS** or **Ubuntu 24.04 LTS** (64-bit)
- **Supported Alternate**: Windows Server 2022 (x64)
- **Python Version**: **Python 3.10** or **Python 3.11** (recommended: 3.10.12 or 3.11.8)
- **NVIDIA GPU Drivers**: Version **535.xx** or newer
- **CUDA Toolkit**: **CUDA 12.1** or **CUDA 12.4**
- **cuDNN**: Version **8.9+** or **9.x**

---  

## 3. Server Setup & Installation (Ubuntu Linux)

### Step 1: System Packages & Dependencies
```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libgomp1 \
    curl \
    git \
    nginx
```

### Step 2: Clone Repository & Create Virtual Environment
```bash
cd /opt
sudo git clone https://github.com/your-org/MeterSerial-VLM.git
sudo chown -R $USER:$USER /opt/MeterSerial-VLM
cd /opt/MeterSerial-VLM

# Create dedicated virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### Step 3: Install PyTorch & PaddlePaddle with CUDA Support

#### For GPU Support (CUDA 12.1):
```bash
# 1. Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install PaddlePaddle GPU (for Linux)
pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu120/

# 3. Install the application requirements
pip install -r requirements.txt
```

#### For CPU-Only Deployment:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install paddlepaddle==3.0.0b1
pip install -r requirements.txt
```

### Step 4: Verify GPU and Model Initialization
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 4. Production Configuration (`.env`)

Create `/opt/MeterSerial-VLM/.env`:
```ini
# Server Binding
HOST=0.0.0.0
PORT=8000
WORKERS=1
LOG_LEVEL=info

# Security: Allowed Origins for CORS (replace with your frontend domain in production)
CORS_ORIGINS=https://meter.yourdomain.com,http://localhost:8000

# OCR Backend ('paddleocr' is default and fastest)
DEFAULT_MODEL=paddleocr

# Green LED detection threshold (% of green pixels in display)
GREEN_LIGHT_THRESHOLD=1.0

# OpenMP runtime conflict safety
KMP_DUPLICATE_LIB_OK=TRUE

# Disable online check for PaddleX models to enable fast/offline startup
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
```

> **Important on Worker Count**:  
> For deep learning APIs utilizing a single GPU, use **`WORKERS=1`** (or 2 if VRAM >= 16 GB). Multiple workers each load separate model copies into GPU memory, which will cause Out-Of-Memory (OOM) errors on smaller GPUs. High concurrency is handled asynchronously by FastAPI and threadpools.

---

## 5. Systemd Service Configuration

To ensure the service runs continuously in the background and restarts automatically on failure or system reboot, create a Systemd unit:

Create `/etc/systemd/system/meterserial-vlm.service`:
```ini
[Unit]
Description=MeterSerial-VLM Inference Service
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/MeterSerial-VLM
Environment="PATH=/opt/MeterSerial-VLM/.venv/bin"
EnvironmentFile=/opt/MeterSerial-VLM/.env
ExecStart=/opt/MeterSerial-VLM/.venv/bin/uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info

Restart=always
RestartSec=5s
LimitNOFILE=65535

# Security hardening
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable meterserial-vlm
sudo systemctl start meterserial-vlm
sudo systemctl status meterserial-vlm
```

View live logs:
```bash
journalctl -u meterserial-vlm -f
```

---

## 6. Nginx Reverse Proxy & SSL Configuration

Create `/etc/nginx/sites-available/meterserial.conf`:
```nginx
# Rate limiting zone (optional, protects against abuse)
limit_req_zone $binary_remote_addr zone=meter_api:10m rate=30r/m;

server {
    listen 80;
    server_name meter.yourdomain.com;

    # Redirect all HTTP traffic to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name meter.yourdomain.com;

    # SSL Certificates (managed via Certbot / Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/meter.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/meter.yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # CRITICAL: Allow up to 25 MB image uploads
    client_max_body_size 25M;

    # API and Frontend Proxy
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;

        # WebSocket and connection headers
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Standard proxy headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts: Allow up to 120s for processing large images on CPU
        proxy_connect_timeout 60s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;

        # Disable proxy buffering for streaming responses
        proxy_buffering off;
    }
}
```

Enable site and restart Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/meterserial.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## 7. Docker Production Deployment

If deploying with Docker and NVIDIA Container Toolkit:

### `Dockerfile`
```dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Avoid prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu120/ \
    && pip3 install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### `docker-compose.yml`
```yaml
version: '3.8'

services:
  meter-serial-vlm:
    build: .
    container_name: meterserial-vlm
    restart: always
    ports:
      - "8000:8000"
    environment:
      - HOST=0.0.0.0
      - PORT=8000
      - WORKERS=1
      - DEFAULT_MODEL=paddleocr
      - GREEN_LIGHT_THRESHOLD=1.0
      - CORS_ORIGINS=*
      - KMP_DUPLICATE_LIB_OK=TRUE
      - PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
    volumes:
      - ./model_store:/app/model_store
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Run with:
```bash
docker compose up -d
```

---

## 8. Health Check & Monitoring

### 1. Automated Health Check Endpoint
- **URL**: `GET /api/v1/health`
- **Sample Output**:
  ```json
  {
    "status": "ok",
    "version": "1.0.0",
    "device": "cuda",
    "loaded_model": "paddleocr"
  }
  ```
- Use this endpoint for AWS ALB, Kubernetes liveness/readiness probes, or UptimeRobot.

### 2. Available Models Endpoint
- **URL**: `GET /api/v1/models`
- **Output**:
  ```json
  {
    "models": ["paddleocr", "lightonocr", "qwen2_vl"],
    "default_model": "paddleocr"
  }
  ```

### 3. Log Rotation
Ensure logs generated by systemd are bounded using `journald` configuration in `/etc/systemd/journald.conf`:
```ini
SystemMaxUse=2G
```

---

## 9. Performance Optimization Summary

1. **Meter Detection & Cropping (Single Image OCR)**:
   The pipeline localizes the meter body using YOLO (`meter_img_detection.pt` / `meter_img_detection.py`) and crops away unnecessary background clutter. Rather than dividing images into 6 overlapping tiles and evaluating 7 images, only the single cropped meter image is sent for OCR text extraction. This slashes processing time by **~85%** while eliminating background text hallucinations.
2. **GPU Acceleration**:
   On a modern NVIDIA GPU (T4 / L4 / RTX 40-series), warm inference latency is under **1.5 seconds per meter**.
3. **Green Light Threshold**:
   The default threshold is calibrated to **1.0%** in `utils/green_light.py` to identify low-light and faint LED indicators reliably.

API's ;


curl -X POST http://localhost:8000/api/v1/extract \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/d:/BsmartAI/Workspaces/MeterSerial-VLM/test_images/newMtrImg_10_12_2026_09_15.jpg" \
     -F "model=paddleocr" 


curl -X POST http://localhost:8000/api/v1/invoice/extract \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/d:/BsmartAI/Workspaces/MeterSerial-VLM/test_images/newMtrImg_10_12_2026_09_15.jpg" \
     -F "model=paddleocr"   


curl -X POST http://localhost:8000/api/v1/ocr/extract \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/d:/BsmartAI/Workspaces/MeterSerial-VLM/test_images/newMtrImg_10_12_2026_09_15.jpg" \
     -F "model=paddleocr"  

curl -X GET "http://localhost:8000/api/v1/health"

curl -X GET "http://localhost:8000/api/v1/queue/status"

Commands;
cd /consumer_app/HES/Bescom/MeterSerial-VLM
source .venv/bin/activate

nohup python3 -m uvicorn api.app:app \
  --host 0.0.0.0 \
  --port 9000 \
  --workers 1 \
  > /consumer_app/HES/Bescom/MeterSerial-VLM/api/meterSerial.logs 2>&1 < /dev/null &

tail -f /consumer_app/HES/Bescom/MeterSerial-VLM/api/meterSerial.logs  