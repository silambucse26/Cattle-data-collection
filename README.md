# Cattle Data Collection & PashuX AI

AI-powered real-time CCTV monitoring, activity recognition, breed identification, and automated data collection pipeline for livestock management.

## 🚀 Key Features

- **Real-Time Cattle Detection & Tracking**: Deep-learning powered identification and tracking across multiple camera streams.
- **Activity & Behavior Monitoring**: Real-time detection of feeding, walking, lying, standing, mounting/estrus, and rumination behavior.
- **AI Model Pipeline**: Integrated breed classification, body condition scoring (BCS), estrus detection, and health monitoring.
- **Supabase Cloud Sync**: Real-time synchronization of telemetry, activity logs, video clip highlights, and farm alerts.
- **Web Dashboard**: Interactive monitoring UI with live analytics, camera management, and historical logs.
- **Telemetry & Edge Queuing**: Async video job queue and performance telemetry for smooth edge execution.

## 📁 Project Structure

```text
├── activity_engine.py          # Activity detection logic & heuristics
├── ai_model_manager.py         # Multi-model inference orchestrator
├── camera_test.py              # Camera feed validation and live vision runner
├── cattle_tracker.py           # Multi-object tracker for cattle
├── pashux_supabase.py          # Supabase cloud backend integration
├── pashux_web_dashboard.py     # Local web dashboard UI
├── performance_telemetry.py    # FPS, latency, and resource metrics
├── video_job_queue.py          # Asynchronous video clip processor
├── test_pashux_pipeline.py     # End-to-end integration test
├── Supabase/                   # Database schemas and SQL migrations
└── .env.example                # Template for configuration and credentials
```

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/silambucse26/Cattle-data-collection.git
   cd Cattle-data-collection
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install ultralytics opencv-python supabase python-dotenv
   ```

4. **Configure Environment Variables:**
   Copy `.env.example` to `.env` and fill in your Supabase credentials and camera RTSP/device details:
   ```bash
   cp .env.example .env
   ```

5. **Run the Dashboard / Pipeline:**
   ```bash
   python pashux_web_dashboard.py
   ```

## 🔒 Security Note

Model weights (`*.pt`, `*.keras`, `*.pkl`) and `.env` credentials are excluded from source control via `.gitignore`.
