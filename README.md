# PashuX AI - Autonomous Livestock Vision & Multi-Model Telemetry Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Ultralytics YOLOv11](https://img.shields.io/badge/YOLO-v11-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)
[![Keras 3](https://img.shields.io/badge/Keras-3.0%2B-D00000.svg)](https://keras.io/)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E.svg)](https://supabase.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

**PashuX AI** is an enterprise-grade, edge-deployable computer vision and cloud telemetry system designed for autonomous livestock farming. It continuously monitors live CCTV/RTSP camera feeds to track cattle and farm handlers, identify 50 indigenous breeds, diagnose clinical disease lesions (FMD, LSD), assess body condition scores (BCS), recognize behavioral patterns (estrus/mounting, feeding, resting, rumination), record highlight video clips, and stream real-time insights to a cloud dashboard.

---

## 📑 Table of Contents

- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Comprehensive AI Models Catalog](#-comprehensive-ai-models-catalog)
- [Empirical Benchmarks & Validation Results](#-empirical-benchmarks--validation-results)
- [Database & Storage Architecture](#-database--storage-architecture)
- [Repository Directory Structure](#-repository-directory-structure)
- [Step-by-Step Setup & How to Run](#-step-by-step-setup--how-to-run)
  - [1. Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
  - [2. Environment Variables (.env)](#2-environment-variables-env)
  - [3. Running the Web Dashboard UI](#3-running-the-web-dashboard-ui)
  - [4. Running the Real-Time CCTV Vision Runner](#4-running-the-real-time-cctv-vision-runner)
  - [5. Running the Model Benchmark Suite](#5-running-the-model-benchmark-suite)
  - [6. Training Breed Classifiers](#6-training-breed-classifiers)
  - [7. Docker & Container Deployment](#7-docker--container-deployment)
  - [8. Production Linux Deployment (Systemd + Nginx)](#8-production-linux-deployment-systemd--nginx)
  - [9. Cloud Deployment (Render.com)](#9-cloud-deployment-rendercom)
  - [10. Data Reset Utility](#10-data-reset-utility)
- [REST API Endpoints](#-rest-api-endpoints)
- [Security & Best Practices](#-security--best-practices)

---

## 🏗️ System Architecture

PashuX AI employs a multi-tiered pipeline: full-frame spatial detection feeds an anatomical cropper and view gating engine, which orchestrates four specialist neural networks in parallel. Detections feed an activity state machine that maintains temporal history, records rolling video clips, and syncs asynchronously to the cloud.

```mermaid
flowchart TD
    subgraph Ingestion["1. Video Ingestion Layer"]
        CCTV["📹 RTSP IP Cameras / CCTV Streams"]
        Uploads["📤 Uploaded Video Files (.mp4, .avi)"]
    end

    subgraph PrimaryVision["2. Primary Spatial Localization"]
        Det["🎯 YOLOv11 Multi-Object Detector\n(yolo11n.pt / yolo11m.pt)"]
        Tracker["🏷️ Cattle & Human Multi-Object Tracker\n(CattleTracker & HumanTracker)"]
    end

    subgraph FeatureGating["3. Quality & Anatomical View Gating"]
        Gate["📐 Aspect Ratio & Resolution Gate\n(Lateral / Hip / Muzzle / Full-Body)"]
    end

    subgraph Specialists["4. Specialist AI Inference Engines"]
        BreedMdl["🧬 50-Breed Classifier\n(best.onnx / cattle_breed_model.pt)\nONNX Runtime + YOLOv11-cls"]
        DiseaseMdl["🩺 Disease Diagnostic CNN\n(final_cattle_disease_model.keras)\nEfficientNet (FMD, LSD, Healthy)"]
        BCSMdl["⚖️ Body Condition Score CNN\n(best_bcs_cnn_v2.pth)\nBCSNetV2 (Scores 1.0 - 5.0)"]
        EstrusMdl["🔥 Estrus & Mounting Suite\n(estrusdetection.pt + CowXNet_heat_model.pkl)\nYOLO-Pose + Random Forest"]
    end

    subgraph StateMachine["5. Activity & Behavioral Engine"]
        Engine["⚙️ ActivityEngine (activity_engine.py)\n• Temporal Session Tracking\n• Rolling Ring Buffer (10s highlights)\n• In-Stall Worker Attendance"]
    end

    subgraph PresentationCloud["6. Cloud Storage & Monitoring"]
        WebUI["💻 PashuX Web Dashboard\n(FastAPI + Tailwind CSS + MJPEG Stream)"]
        Supabase["☁️ Supabase PostgreSQL & S3 Storage\n(Tables: cattle, activities, alerts, telemetry)"]
    end

    Ingestion --> Det
    Det --> Tracker
    Tracker --> Gate
    Gate --> BreedMdl
    Gate --> DiseaseMdl
    Gate --> BCSMdl
    Gate --> EstrusMdl

    BreedMdl --> Engine
    DiseaseMdl --> Engine
    BCSMdl --> Engine
    EstrusMdl --> Engine
    Tracker -. Handlers .-> Engine

    Engine --> WebUI
    Engine --> Supabase
```

---

## 💻 Technology Stack

### Backend & API Framework
* **FastAPI (0.110+)**: High-throughput asynchronous REST API for dashboard services, video upload processing, and WebSocket/telemetry feeds.
* **Uvicorn (Standard)**: Asynchronous ASGI production server.
* **Pydantic (v2)**: Strict request/response payload validation and typing.
* **PyJWT & Bcrypt**: Secure farm-level JSON Web Token authentication with salted password hashing.

### Computer Vision & Machine Learning
* **Ultralytics YOLOv11**: Real-time object detection (Cattle, Handlers) and skeletal pose estimation.
* **PyTorch (2.2+ CPU/CUDA)**: Deep learning runtime for `BCSNetV2` and tensor manipulation.
* **Keras 3 (Torch Backend)**: EfficientNet CNN execution for cattle disease classification.
* **ONNX Runtime (1.17+)**: Accelerated, low-latency inference for the 50-breed cattle classifier.
* **OpenCV (Headless 4.9+)**: Frame decoding, drawing bounding boxes, MJPEG live video streaming, and video clipping with H.264 codecs.
* **Scikit-Learn & Joblib**: Random Forest classification for kinematic estrus mounting behavior.
* **SciPy & NumPy**: Geometric coordinate normalization and temperature-scaled probability calibration.

### Cloud Database & Storage
* **Supabase (PostgreSQL 15)**: Relational schema tracking farms, cameras, users, cattle registries, worker attendance, and behavior logs.
* **Supabase Storage**: S3-compatible cloud object storage buckets for high-resolution alert snapshots and 10-second highlight `.mp4` video clips.
* **PostgREST & Gotrue**: Direct RESTful database operations and session management.

### Deployment & DevOps
* **Docker & Docker Compose**: Multi-stage containerization with resource limits (4GB limit, 1GB reservation).
* **Nginx**: Production reverse proxy handling SSL termination and proxy buffering.
* **Systemd**: Linux background daemon management with automatic restarts.
* **Render.com**: Automated cloud container deployment specification (`render.yaml`).

---

## 🧠 Comprehensive AI Models Catalog

| AI Pipeline Stage | Model Weights File | Base Architecture & Framework | Input Dimensions | Classes / Output Range | Description & Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Primary Cattle & Human Detector** | `yolo11n.pt` *(CPU)*<br>`yolo11m.pt` *(CUDA)* | **Ultralytics YOLOv11**<br>*(PyTorch)* | `640 x 640 x 3` | COCO 80 classes<br>*(Filtered: Class 19 Cattle, Class 0 Human)* | Full-frame object detector. Localizes all cattle and handlers in the frame with bounding boxes and confidence scores. |
| **2. Cattle Breed Classifier** | `best.onnx` *(Primary)*<br>`cattle_breed_model.pt` *(Ensemble)* | **YOLOv11s-cls / YOLOv11n-cls**<br>*(Ultralytics + ONNX Runtime)* | `224 x 224 x 3` | **50 Indigenous Indian Breeds**<br>*(Gir, Sahiwal, Red Sindhi, Kankrej, Ongole, Tharparkar, Rathi, etc.)* | Classifies cropped cattle images into 50 distinct breeds. Accelerated via ONNX Runtime for sub-20ms inference. |
| **3. Clinical Cattle Disease Classifier** | `final_cattle_disease_model.keras` | **EfficientNet Deep CNN**<br>*(Keras 3 with PyTorch Backend)* | `300 x 300 x 3` | **3 Diagnostic Classes:**<br>1. Foot and Mouth Disease (FMD)<br>2. Healthy Cattle<br>3. Lumpy Skin Disease (LSD) | Evaluates skin texture, muzzle, and hoof crops for lesions, blisters, and nodules. Evaluated at **92.22% overall accuracy**. |
| **4. Body Condition Scoring (BCS)** | `best_bcs_cnn_v2.pth` | **BCSNetV2**<br>*(Custom 5-Layer CNN with BatchNorm & Adaptive Pooling, PyTorch)* | `224 x 224 x 3` | **5 Veterinary Grades (1.0 to 5.0):**<br>• BCS 1: Very Thin<br>• BCS 2: Thin<br>• BCS 3: Ideal<br>• BCS 4: Fleshy<br>• BCS 5: Obese | Evaluates pelvic hook/pin bones and spinal transverse processes. Uses temperature calibration ($T=1.3$) to output continuous veterinary health scores. |
| **5. Estrus Keypoint & Pose Model** | `estrusdetection.pt` | **YOLOv11-Pose**<br>*(PyTorch)* | `640 x 640 x 3` | Cattle Skeletal Keypoints<br>*(Head, Spine, Tailhead, Pin/Hook Bones)* | Tracks 2D skeletal keypoints to monitor spinal curvature, elevation, and mounting postures. |
| **6. Mounting & Heat Cycle Classifier** | `CowXNet_heat_model.pkl` | **Random Forest Classifier**<br>*(Scikit-Learn)* | 12-Dimensional Feature Vector | Binary Heat Status<br>*(In-Heat vs. Normal Behavior)* | Classifies normalized coordinate vectors extracted from bounding box geometry and skeletal keypoints. |

---

## 📊 Empirical Benchmarks & Validation Results

Benchmarks evaluated using [`benchmark_models.py`](file:///d:/Project/Catlle%20detection/benchmark_models.py):

### 1. Cattle Disease Model (`final_cattle_disease_model.keras`)
* **Test Dataset:** `D:\Project\Disease detection\Dataset` (Stratified across classes)
* **Overall Test Accuracy:** **92.22%** (166 / 180 images correct)
* **Macro Precision / Recall / F1-Score:** **92.22%**
* **Average Latency (CPU):** `790.92 ms` (~1.3 FPS)

| Diagnostic Class | Precision | Recall | F1-Score | Evaluation Support |
| :--- | :---: | :---: | :---: | :---: |
| **Foot and Mouth Disease (FMD)** | **95.00%** | **95.00%** | **95.00%** | 60 images |
| **Healthy Cattle** | **88.33%** | **88.33%** | **88.33%** | 60 images |
| **Lumpy Skin Disease (LSD)** | **93.33%** | **93.33%** | **93.33%** | 60 images |

#### Disease Confusion Matrix
| Ground Truth \ Predicted | Foot and Mouth | Healthy | Lumpy Skin |
| :--- | :---: | :---: | :---: |
| **Foot and Mouth Disease** | **57** | 3 | 0 |
| **Healthy Cattle** | 3 | **53** | 4 |
| **Lumpy Skin Disease** | 0 | 4 | **56** |

> **Clinical Finding:** Zero cross-confusion occurred between Foot & Mouth Disease and Lumpy Skin Disease. Minor errors occurred solely in distinguishing early subtle lesions from normal healthy cattle skin.

---

### 2. Body Condition Score Model (`best_bcs_cnn_v2.pth`)
* **Architecture:** `BCSNetV2` (1,013,509 parameters, ~3.89 MB)
* **Training Checkpoint:** Epoch 27 (`Val Accuracy: 56.65%`, `Balanced Accuracy: 46.56%`)
* **Continuous Output Range:** `1.3` to `5.0` (Average: `3.77`, Mean confidence: `86.8%`)
* **Inference Latency (CPU):** **47.98 ms** (`20.8 FPS`) — suitable for real-time tracking.

---

## 🗄️ Database & Storage Architecture

The database runs on Supabase (PostgreSQL 15) with automated migrations provided in the `Supabase/` directory:

```mermaid
erDiagram
    FARMS ||--o{ USERS : "employs"
    FARMS ||--o{ CAMERAS : "operates"
    FARMS ||--o{ CATTLE : "registers"
    FARMS ||--o{ WORKERS : "tracks"
    FARMS ||--o{ FARM_ACTIVITIES : "logs"
    FARMS ||--o{ ALERTS : "triggers"
    FARMS ||--o{ TELEMETRY_LOGS : "records"

    FARMS {
        uuid id PK
        varchar farm_code UK
        varchar farm_name
        varchar owner_phone
    }
    CATTLE {
        uuid id PK
        uuid farm_id FK
        varchar cattle_id UK
        varchar breed
        varchar disease_status
        real bcs_score
        real estimated_weight_kg
        text image_url
        timestamptz last_seen
    }
    WORKERS {
        uuid id PK
        uuid farm_id FK
        varchar worker_id UK
        text image_url
        timestamptz last_seen
    }
    FARM_ACTIVITIES {
        uuid id PK
        uuid farm_id FK
        varchar camera_id FK
        varchar cattle_id
        varchar activity_type
        real confidence
        jsonb ai_detection_result
        text snapshot_url
        text video_clip_url
        timestamptz start_time
        timestamptz end_time
    }
    ALERTS {
        uuid id PK
        uuid farm_id FK
        varchar alert_type
        varchar severity
        text message
        boolean is_acknowledged
    }
```

### Storage Buckets
* `pashux-activity-images`: Stores high-resolution JPEG crops of detected cattle, suspected disease lesions, and worker check-ins.
* `pashux-activity-videos`: Stores 10-second MP4 highlight clips (5 seconds pre-event buffer + 5 seconds post-event recording).

---

## 📁 Repository Directory Structure

```text
├── activity_engine.py             # Cattle behavior, temporal deduplication & clip recorder
├── ai_model_manager.py            # Singleton AI model loader & thread-safe inference manager
├── benchmark_models.py            # Automated accuracy & latency benchmarking suite
├── camera_test.py                 # Multi-camera RTSP ingestion & visual overlay pipeline
├── cattle_tracker.py              # Multi-object tracker for cattle & farm workers
├── performance_telemetry.py       # Live FPS, memory, and latency telemetry profiler
├── pashux_supabase.py             # Supabase cloud database & storage client
├── pashux_web_dashboard.py        # FastAPI web application & monitoring dashboard
├── video_job_queue.py             # Asynchronous background video analysis worker
├── wipe_all_data.py               # Clean database & capture reset utility
│
├── train_cattle_breed.py          # Local/Colab training script for 50-breed YOLO classifier
├── train_jarvis.py                # Cloud GPU training script for JarvisLabs / RunPod
│
├── best.onnx                      # 50-breed classifier exported to ONNX Runtime (41/50 classes)
├── cattle_breed_model.pt          # PyTorch weights for 50-breed YOLO classifier
├── final_cattle_disease_model.keras # Keras 3 EfficientNet cattle disease classifier
├── best_bcs_cnn_v2.pth            # PyTorch checkpoint for 5-class BCSNetV2
├── estrusdetection.pt             # YOLOv11 cattle pose estimation model
├── CowXNet_heat_model.pkl         # Scikit-learn Random Forest model for estrus scoring
├── yolo11n.pt / yolo11m.pt        # Primary YOLOv11 object detectors
│
├── captures/                      # Local storage for generated images & video clips
│   ├── images/                    # Highlight snapshot crops
│   └── videos/                    # 10-second MP4 event clips
├── uploads/                       # Temp storage for dashboard video uploads
├── deploy/                        # Production deployment configurations
│   ├── deploy_production.sh       # Automated Linux server deployment script
│   ├── nginx/pashux.conf          # Nginx reverse proxy & buffering configuration
│   └── systemd/pashux.service     # Systemd background daemon service unit
│
├── Supabase/                      # SQL database schema and migrations
│   ├── supabase_schema.sql        # Full database schema with public table definitions
│   └── create_cattle_table.sql    # Dedicated cattle & worker registry tables
│
├── requirements.txt               # Production Python dependencies
├── Dockerfile                     # Containerization blueprint
├── docker-compose.yml             # Docker Compose orchestration
├── render.yaml                    # Render.com Cloud deploy spec
├── .env.example                   # Environment configuration template
└── README.md                      # Comprehensive project documentation
```

---

## 🚀 Step-by-Step Setup & How to Run

### 1. Prerequisites & Environment Setup

* **Python 3.10+** (Tested on Python 3.10, 3.11, 3.12, 3.13)
* **FFmpeg** installed and accessible in your system `PATH`
* **Git** installed

```bash
# Clone the repository
git clone https://github.com/silambucse26/Cattle-data-collection.git
cd Cattle-data-collection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (cmd):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Environment Variables (.env)

Copy `.env.example` to create your active configuration:
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```ini
# Supabase Cloud Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-anon-or-service-role-key
SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key

# Farm & Camera Identifiers
FARM_ID=00000000-0000-0000-0000-000000000001
FARM_CODE=FARM001
CAMERA_ID=CAM01

# Video & Camera Ingestion Sources
# Use 0 for local USB webcam, or RTSP URL for CCTV stream:
RTSP_CAMERA_URL=rtsp://admin:password@192.168.1.100:554/stream1

# Hardware Acceleration
USE_CUDA=true                # Set to false for CPU-only mode

# Server & Dashboard
WEB_DASHBOARD_PORT=8000
JWT_SECRET=your-random-secret-key-32-chars-long
```

---

### 3. Running the Web Dashboard UI

The web dashboard provides live camera streaming, analytics charts, activity history, and video upload capabilities:

```bash
# Run with python directly:
python pashux_web_dashboard.py

# Or run with uvicorn:
uvicorn pashux_web_dashboard:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser and navigate to: **`http://localhost:8000`**

---

### 4. Running the Real-Time CCTV Vision Runner

To run the camera processing pipeline directly with OpenCV visual overlays and live bounding box rendering:

```bash
python camera_test.py
```
* Press `q` to safely terminate the video loop and flush session data to Supabase.
* Press `s` to save an immediate manual snapshot crop.

---

### 5. Running the Model Benchmark Suite

Measure accuracy, precision, recall, F1, confusion matrices, and latency on your local hardware:

```bash
# Benchmark all models (Disease CNN + BCS CNN):
python benchmark_models.py --model all --samples 60

# Benchmark only the Disease Diagnostic Model:
python benchmark_models.py --model disease --samples 100

# Benchmark only the Body Condition Scoring Model:
python benchmark_models.py --model bcs --runs 50
```

---

### 6. Training Breed Classifiers

To retrain the 50-breed Indian cattle classification model:

* **Local or Google Colab Training:**
  ```bash
  python train_cattle_breed.py
  ```
* **Cloud GPU Training (JarvisLabs / RunPod / LambdaLabs):**
  ```bash
  python train_jarvis.py
  ```

---

### 7. Docker & Container Deployment

Build and launch the containerized application with Docker Compose:

```bash
# Build and run in detached mode:
docker compose up -d --build

# View real-time container logs:
docker compose logs -f

# Stop container:
docker compose down
```

---

### 8. Production Linux Deployment (Systemd + Nginx)

Automated scripts are included in `deploy/`:

1. Run the deployment script on Ubuntu/Debian:
   ```bash
   chmod +x deploy/deploy_production.sh
   sudo ./deploy/deploy_production.sh
   ```
2. Enable and start the systemd service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable pashux
   sudo systemctl start pashux
   sudo systemctl status pashux
   ```
3. Set up the Nginx reverse proxy configuration:
   ```bash
   sudo cp deploy/nginx/pashux.conf /etc/nginx/sites-available/
   sudo ln -s /etc/nginx/sites-available/pashux.conf /etc/nginx/sites-enabled/
   sudo nginx -t && sudo systemctl restart nginx
   ```

---

### 9. Cloud Deployment (Render.com)

The repository includes a ready-to-use [`render.yaml`](file:///d:/Project/Catlle%20detection/render.yaml) blueprint:
1. Link your GitHub repository to Render.
2. Select **New Blueprint Instance**.
3. Supply `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, and `FARM_ID` in the Render environment variables dashboard.
4. Render builds the Docker image and launches the service with automatic health checks on `/`.

---

### 10. Data Reset Utility

To clear stale demo records, local video clips, and reset the local roster:

```bash
python wipe_all_data.py
```

---

## 📡 REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/auth/register` | Register new farm manager with phone & bcrypt password |
| `POST` | `/api/auth/login` | Authenticate user and receive JWT bearer token |
| `GET` | `/api/auth/me` | Fetch active user profile and farm assignment |
| `GET` | `/api/farm/cameras` | List all configured CCTV cameras for the farm |
| `POST` | `/api/cctv/connect` | Connect to an RTSP IP camera feed |
| `GET` | `/api/cctv/status` | Current streaming status, FPS, and frame dimensions |
| `GET` | `/api/cctv/stream` | Multi-part MJPEG live video stream for web display |
| `POST` | `/api/analyze/upload-video` | Upload a recorded video for asynchronous multi-model processing |
| `GET` | `/api/jobs/{job_id}` | Poll progress and results of background video analysis jobs |
| `GET` | `/api/performance/telemetry` | Real-time CPU, RAM, GPU, and per-model latency telemetry |
| `GET` | `/api/activities` | Fetch historical cattle activity logs with pagination and filters |
| `DELETE`| `/api/activities/clear-all` | Clear activity logs from the dashboard |
| `POST` | `/api/supabase/sync` | Force synchronization between local queue and Supabase |

---

## 🔒 Security & Best Practices

* **Credentials Protection:** Model weights (`*.pt`, `*.keras`, `*.onnx`) and secrets (`.env`) are excluded from Git via `.gitignore`.
* **Row-Level Security (RLS):** Supabase tables enforce tenant isolation so each farm can only view its own cameras, cattle, and alerts.
* **Stream Resilience:** The RTSP reader automatically attempts reconnection with exponential backoff if a network drop occurs.
* **Thread Safety:** The singleton `AIModelManager` uses execution locks to prevent concurrent CUDA/CPU memory contention between live feeds and uploaded video worker threads.
