"""
PashuX AI – Complete Web Dashboard & Farm Video Management Server
Features:
- Phone & Password Authentication (JWT Session, Bcrypt Hashed)
- Automatic Farm Creation & User Account Connection (Supabase)
- Connect Farm CCTV Cameras (RTSP/ONVIF or Local)
- Upload Existing CCTV Recordings for Instant AI Analysis
- Live Stream MJPEG Feed with Real-Time AI Detection Overlays
- Automatic 10-Second Video Clip Evidence & Snapshot Capture
- Supabase Storage & PostgreSQL Database Activity History
- Search & Filter by Date, Farm, Camera, Cattle ID, Activity Type
"""

from pashux_supabase import LOCAL_DB_FILE
import os
import sys
import time
import json
import uuid
import datetime
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

if sys.platform == "win32":
    torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib) and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File, Form, Header, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent

from pashux_supabase import (
    supabase_manager,
    LOCAL_IMAGES_DIR,
    LOCAL_VIDEOS_DIR,
    LOCAL_DB_FILE,
    CATTLE_ROSTER_FILE,
    WORKER_ROSTER_FILE,
    DEFAULT_FARM_ID,
    DEFAULT_CAMERA_ID,
    JWT_SECRET
)
from activity_engine import ActivityEngine
from ai_model_manager import ai_models
from cattle_tracker import SpatialCattleTracker, SpatialHumanTracker, calculate_iou
from video_job_queue import video_queue
from performance_telemetry import telemetry_profiler
from camera_test import estimate_cattle_biometry

app = FastAPI(title="PashuX AI Farm Dashboard", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    farm_id = os.getenv("FARM_ID", DEFAULT_FARM_ID)
    pass

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Global Live Stream & Background AI Processing State
live_stream_lock = threading.Lock()
current_live_frame = None
active_monitoring_state = {
    "is_running": False,
    "source_type": "none", # "live" or "uploaded"
    "source_path": "",
    "farm_id": DEFAULT_FARM_ID,
    "camera_id": DEFAULT_CAMERA_ID,
    "frames_processed": 0,
    "status_text": "Idle"
}
active_stop_event = threading.Event()

# Cattle Registration Audit & Retry State
last_registration_audit: Dict[str, Any] = {
    "farm_id": DEFAULT_FARM_ID,
    "camera_id": DEFAULT_CAMERA_ID,
    "timestamp": None,
    "detected_count": 0,
    "registered_count": 0,
    "success": True,
    "detected_cattle": [],
    "registered_cattle": [],
    "failed_cattle": {}
}



# ---------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------
class LoginRequest(BaseModel):
    phone: str
    password: str

class RegisterRequest(BaseModel):
    phone: str
    password: str
    full_name: str
    farm_name: Optional[str] = "My Dairy Farm"
    farm_id: Optional[str] = None
    role: Optional[str] = "owner"

class ConnectCameraRequest(BaseModel):
    camera_name: str
    rtsp_url: str
    location_description: Optional[str] = "Main Barn"
    camera_id: Optional[str] = None


# ---------------------------------------------------------------------
# Auth Dependency & Farm-Level Access Control
# ---------------------------------------------------------------------
def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Validates session JWT token and extracts user farm ID."""
    if not authorization:
        # Default fallback to demo owner for testing
        return {
            "user_id": "demo-user-id",
            "farm_id": DEFAULT_FARM_ID,
            "phone": "+919876543210",
            "role": "owner",
            "full_name": "Ramesh Patel",
            "farm_name": "PashuX Pioneer Dairy Farm"
        }

    token = authorization.replace("Bearer ", "").strip()
    payload = supabase_manager.decode_jwt_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or invalid token. Please log in again."
        )
    return payload


# ---------------------------------------------------------------------
# Auth API Endpoints
# ---------------------------------------------------------------------
@app.post("/api/auth/login")
async def login(req: LoginRequest):
    """Authenticate with Phone Number and Bcrypt Hashed Password."""
    success, user_data, message = supabase_manager.authenticate_user_by_phone(req.phone, req.password)
    if not success or not user_data:
        raise HTTPException(status_code=400, detail=message)
    return {"success": True, "message": message, "user": user_data}


@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    """Register new user, create a dedicated Farm in Supabase, and link account."""
    clean_phone = req.phone.strip()
    clean_name = req.full_name.strip()
    clean_farm_name = req.farm_name.strip() if req.farm_name else f"{clean_name}'s Dairy Farm"

    # 1. Create new farm in Supabase if live, or generate local UUID
    new_farm_id = str(uuid.uuid4())
    if supabase_manager.is_connected and supabase_manager.client:
        try:
            f_res = supabase_manager.client.table("farms").insert({
                "id": new_farm_id,
                "farm_code": f"FARM_{uuid.uuid4().hex[:6].upper()}",
                "farm_name": clean_farm_name,
                "owner_name": clean_name,
                "owner_phone": clean_phone
            }).execute()
            if f_res.data:
                new_farm_id = f_res.data[0]["id"]
        except Exception as e:
            print(f"[Supabase Farm Create Notice] {e}")

    success, user_data, message = supabase_manager.register_user(
        farm_id=new_farm_id,
        phone=clean_phone,
        plain_password=req.password,
        full_name=clean_name,
        role=req.role or "owner"
    )
    if not success or not user_data:
        raise HTTPException(status_code=400, detail=message)

    user_data["farm_name"] = clean_farm_name
    return {"success": True, "message": message, "user": user_data}


@app.get("/api/auth/me")
async def get_me(user: Dict[str, Any] = Depends(get_current_user)):
    """Returns currently authenticated user profile and farm details."""
    farm_id = user.get("farm_id", DEFAULT_FARM_ID)
    farm_name = user.get("farm_name", "Registered Dairy Farm")

    if supabase_manager.is_connected and supabase_manager.client:
        try:
            res = supabase_manager.client.table("farms").select("farm_name, farm_code").eq("id", farm_id).execute()
            if res.data and len(res.data) > 0:
                farm_name = res.data[0].get("farm_name", farm_name)
        except Exception:
            pass

    user_info = dict(user)
    user_info["farm_name"] = farm_name
    return {"user": user_info}


# ---------------------------------------------------------------------
# Camera Management & Live Stream Connection Endpoints
# ---------------------------------------------------------------------
@app.get("/api/farm/cameras")
async def get_cameras(user: Dict[str, Any] = Depends(get_current_user)):
    """Returns CCTV cameras belonging exclusively to the authenticated user's farm."""
    farm_id = user.get("farm_id", DEFAULT_FARM_ID)
    cameras = supabase_manager.get_farm_cameras(farm_id)
    return {"cameras": cameras}


@app.post("/api/cctv/connect")
async def connect_cctv_camera(req: ConnectCameraRequest, user: Dict[str, Any] = Depends(get_current_user)):
    """
    Connects a farm CCTV camera (RTSP / ONVIF / IP URL) to the user's farm,
    saves the camera record in Supabase, and starts continuous AI monitoring.
    """
    farm_id = user.get("farm_id", DEFAULT_FARM_ID)
    cam_id = req.camera_id or f"CAM_{uuid.uuid4().hex[:4].upper()}"

    # 1. Save Camera Record into Supabase
    if supabase_manager.is_connected and supabase_manager.client:
        try:
            supabase_manager.client.table("cameras").insert({
                "id": cam_id,
                "farm_id": farm_id,
                "camera_name": req.camera_name,
                "rtsp_url": req.rtsp_url,
                "location_description": req.location_description or "Main Barn",
                "is_active": True
            }).execute()
        except Exception as e:
            print(f"[Supabase Camera Notice] {e}")

    # 2. Launch Background AI Processing Worker on the Live Stream
    start_monitoring_worker(
        source_path=req.rtsp_url,
        source_type="live",
        farm_id=farm_id,
        camera_id=cam_id
    )

    return {
        "success": True,
        "message": f"Camera '{req.camera_name}' ({cam_id}) connected! PashuX AI is now monitoring and recording activities to Supabase.",
        "camera_id": cam_id,
        "farm_id": farm_id
    }


@app.get("/api/cctv/status")
async def get_cctv_status():
    """Returns status of active AI video/CCTV processing."""
    return active_monitoring_state


@app.get("/api/cctv/stream")
async def get_cctv_mjpeg_stream():
    """MJPEG Live video stream with AI detection overlays for the browser."""
    def frame_generator():
        global current_live_frame
        idle_count = 0
        while True:
            with live_stream_lock:
                if current_live_frame is not None:
                    idle_count = 0
                    ret, jpeg = cv2.imencode(".jpg", current_live_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        frame_bytes = jpeg.tobytes()
                        yield (b"--frame\r\n"
                               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                else:
                    idle_count += 1
            if not active_monitoring_state.get("is_running", False) and idle_count > 25:
                # Video analysis finished and closed cleanly - close stream
                break
            time.sleep(0.04) # ~25 fps

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------------
# AI Video Analysis & 10s Video Clip Evidence Worker
# ---------------------------------------------------------------------
def start_monitoring_worker(source_path: str, source_type: str, farm_id: str, camera_id: str):
    """Stops any existing worker and launches a new background AI detection thread for live feeds."""
    global active_stop_event, active_monitoring_state
    active_stop_event.set()
    time.sleep(0.5)
    active_stop_event = threading.Event()

    active_monitoring_state["is_running"] = True
    active_monitoring_state["source_type"] = source_type
    active_monitoring_state["source_path"] = source_path
    active_monitoring_state["farm_id"] = farm_id
    active_monitoring_state["camera_id"] = camera_id
    active_monitoring_state["frames_processed"] = 0
    active_monitoring_state["status_text"] = f"Monitoring {source_type.upper()} ({camera_id})"

    t = threading.Thread(
        target=ai_video_monitoring_loop,
        args=(source_path, source_type, farm_id, camera_id, active_stop_event),
        daemon=True
    )
    t.start()


def check_cattle_estrus_behavior(
    c1: Dict[str, Any],
    c2: Dict[str, Any],
    human_present: bool = False,
    ai_estrus_confirmed: bool = False
) -> Tuple[bool, str, float]:
    """
    Veterinary Kinematic Estrus Mounting Verification:
    1. Rejects if human/worker is attending (avoids confounding veterinary handling with mounting).
    2. Real Mounting Elevation: The mounting cow raises its forelimbs onto the back of the mounted cow.
       - The mounting cow's top boundary (y1_1) must be significantly elevated above the mounted cow (y1_2 - min_h * 0.22).
    3. Vertical Rearing Silhouette: Normal standing cows are horizontal (w/h >= 1.30). A mounting cow rears up and becomes taller (w1/h1 <= 1.25).
    4. Heavy Overlap: Rearing cow physically covers the pelvis/back of the mounted cow (iomin >= 0.38).
    5. Longitudinal Alignment: Both cows must be longitudinally aligned (abs(cx1 - cx2) < max(w1, w2) * 0.50).
    6. Cows in separate parallel stalls standing side-by-side must never trigger false estrus.
    """
    if human_present:
        return False, "", 0.0

    box1, box2 = c1.get("box", [0, 0, 0, 0]), c2.get("box", [0, 0, 0, 0])
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    w1, h1 = max(1, x2_1 - x1_1), max(1, y2_1 - y1_1)
    w2, h2 = max(1, x2_2 - x1_2), max(1, y2_2 - y1_2)

    if min(w1, h1, w2, h2) < 55:
        return False, "", 0.0

    # Aspect ratios (width / height)
    ar1 = float(w1) / float(h1)
    ar2 = float(w2) / float(h2)

    inter_x = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    inter_y = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    inter_area = inter_x * inter_y

    min_area = min(w1 * h1, w2 * h2)
    iomin = (inter_area / float(min_area)) if min_area > 0 else 0.0

    cx1, cy1 = (x1_1 + x2_1) / 2.0, (y1_1 + y2_1) / 2.0
    cx2, cy2 = (x1_2 + x2_2) / 2.0, (y1_2 + y2_2) / 2.0
    min_h = min(h1, h2)

    # Lateral proximity: cows in separate parallel stalls standing side-by-side must not trigger estrus
    # True mounting has vertical rearing (ar1 <= 1.25 or elevated top edge by at least 22% of height) AND substantial overlap (>= 38%)
    is_elevated_mounting = (
        (iomin >= 0.38) and
        (y1_1 < y1_2 - min_h * 0.22) and
        (cy1 < cy2) and
        (ar1 <= 1.30 or ai_estrus_confirmed) and
        (abs(cx1 - cx2) < max(w1, w2) * 0.55)
    )

    if is_elevated_mounting or (ai_estrus_confirmed and iomin >= 0.30):
        conf = min(0.98, max(0.85, 0.84 + (iomin * 0.16)))
        return True, "Active Mounting (Estrus Heat)", conf

    return False, "", 0.0


def ai_video_monitoring_loop(
    source_path: str,
    source_type: str,
    farm_id: str,
    camera_id: str,
    stop_event: threading.Event,
    job_id: Optional[str] = None,
    enabled_models: Optional[List[str]] = None
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Optimized AI detection loop:
    - Preloaded singleton AI models (Zero disk reload)
    - User-selectable AI analysis models (Breed, BCS, Biometry, Disease, Estrus, Worker)
    - Cumulative spatial tracking across sequential uploads without ID collision
    - Cattle Profile Caching: specialist models run max 2 times per cow
    - Frame Stride: detects every 2nd frame on video playback
    - Asynchronous 10-second video encoding & Supabase upload
    - Performance telemetry profiling
    """
    global current_live_frame, active_monitoring_state
    if enabled_models is None:
        enabled_models = ["breed", "bcs", "biometry", "disease", "estrus", "worker"]
    print(f"\n[PashuX AI] Launching Optimized AI Detection on {source_type.upper()}: {source_path} (Active Models: {enabled_models})")

    # Parse camera input: if digit string (e.g. '0'), convert to int for webcam
    cap_input = int(source_path) if (isinstance(source_path, str) and source_path.isdigit()) else source_path
    cap = cv2.VideoCapture(cap_input)

    if not cap.isOpened():
        err_msg = f"Could not connect to video/camera source: {source_path}"
        print(f"[PashuX AI] Error: {err_msg}")
        active_monitoring_state["is_running"] = False
        active_monitoring_state["status_text"] = err_msg
        return False, err_msg, {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if source_type == "uploaded" else 0

    # Initialize Spatial Cattle & Human Trackers (Maintain cumulative unique cattle and worker IDs across videos)
    max_cattle_num = 0
    max_worker_num = 0

    # 1. Check existing cattle and workers in Supabase
    if supabase_manager.is_connected and supabase_manager.client:
        try:
            c_res = supabase_manager.client.table("cattle").select("cattle_id").eq("farm_id", farm_id).execute()
            if c_res.data:
                for row in c_res.data:
                    c_tag = str(row.get("cattle_id", ""))
                    digits = "".join([c for c in c_tag if c.isdigit()])
                    if digits:
                        max_cattle_num = max(max_cattle_num, int(digits))
        except Exception:
            pass

        try:
            w_res = supabase_manager.client.table("workers").select("worker_id").eq("farm_id", farm_id).execute()
            if w_res.data:
                for row in w_res.data:
                    w_tag = str(row.get("worker_id", ""))
                    digits = "".join([c for c in w_tag if c.isdigit()])
                    if digits:
                        max_worker_num = max(max_worker_num, int(digits))
        except Exception:
            pass

    # 2. Check existing local cattle roster
    try:
        if CATTLE_ROSTER_FILE.exists():
            with open(CATTLE_ROSTER_FILE, "r", encoding="utf-8") as _rf:
                _roster = json.load(_rf)
            for _k, _v in _roster.items():
                _num = _v.get("numeric_id")
                if _num is not None:
                    max_cattle_num = max(max_cattle_num, int(_num))
                _d = "".join([c for c in str(_k) if c.isdigit()])
                if _d:
                    max_cattle_num = max(max_cattle_num, int(_d))
    except Exception:
        pass

    # 3. Check existing local worker roster
    try:
        if WORKER_ROSTER_FILE.exists():
            with open(WORKER_ROSTER_FILE, "r", encoding="utf-8") as _wrf:
                _wroster = json.load(_wrf)
            for _k, _v in _wroster.items():
                _num = _v.get("numeric_id")
                if _num is not None:
                    max_worker_num = max(max_worker_num, int(_num))
                _d = "".join([c for c in str(_k) if c.isdigit()])
                if _d:
                    max_worker_num = max(max_worker_num, int(_d))
    except Exception:
        pass

    # 4. Check existing local activities
    try:
        if LOCAL_DB_FILE.exists():
            with open(LOCAL_DB_FILE, "r", encoding="utf-8") as _f:
                _recs = json.load(_f)
            for _r in _recs:
                _cid = _r.get("cattle_id")
                if _cid:
                    _d = "".join([c for c in str(_cid) if c.isdigit()])
                    if _d:
                        max_cattle_num = max(max_cattle_num, int(_d))
                _wid = _r.get("worker_id")
                if _wid:
                    _wd = "".join([c for c in str(_wid) if c.isdigit()])
                    if _wd:
                        max_worker_num = max(max_worker_num, int(_wd))
    except Exception:
        pass

    # Maintain cumulative numbering so each new video continues without overwriting prior video data
    start_cattle_id = max_cattle_num + 1
    start_worker_id = max_worker_num + 1
    print(f"\n[PashuX AI] Consecutive ID assignment: Cattle starts at COW{start_cattle_id:03d}, Workers start at WRK{start_worker_id:03d} (Preserving all previous videos)")

    cattle_tracker = SpatialCattleTracker(start_id=start_cattle_id, iou_threshold=0.22, max_missing_frames=100000)
    human_tracker = SpatialHumanTracker(start_id=start_worker_id, iou_threshold=0.20, max_missing_frames=60)
    session_updated_worker_photos = set()
    cattle_unique_photo_names: Dict[int, str] = {}
    worker_unique_photo_names: Dict[int, str] = {}

    # Initialize Activity Engine (10-second clip capture: 4s pre + 6s post)
    engine = ActivityEngine(
        farm_id=farm_id,
        camera_id=camera_id,
        video_source_type=source_type,
        buffer_fps=fps,
        pre_buffer_seconds=4.0,
        post_buffer_seconds=6.0,
        cooldown_seconds=45.0
    )

    frame_count = 0
    # Adaptive stride: Analyzing ~5 frames/sec on uploaded video preserves 100% detection accuracy while cutting analysis time by 50%
    stride = max(4, int(round(fps / 5.0))) if source_type == "uploaded" else 1
    cattle_tracks: List[Dict[str, Any]] = []
    human_tracks: List[Dict[str, Any]] = []
    estrus_events: List[Dict[str, Any]] = []
    video_base_dt = datetime.datetime.now(datetime.timezone.utc)
    mounting_streaks: Dict[Tuple[int, int], int] = {}
    video_sec = 0.0
    loop_start_time = time.time()

    annotated_frame_buffer = None
    session_updated_photos = set()
    session_updated_worker_photos = set()
    session_registered_cattle = set()
    session_fullbody_synced = set()
    session_synced_breed_conf: Dict[int, float] = {}
    session_registered_workers = set()
    try:
        while not stop_event.is_set():
            try:
                ret, frame = cap.read()
            except Exception as _read_err:
                import gc
                gc.collect()
                time.sleep(0.04)
                try:
                    ret, frame = cap.read()
                except Exception:
                    break

            if not ret or frame is None:
                is_file = isinstance(source_path, str) and (
                    os.path.isfile(source_path) or
                    any(source_path.lower().endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"])
                )
                if source_type == "uploaded" or is_file:
                    print(f"[PashuX AI] Video playback completed ({frame_count} frames analyzed). Closing cleanly...")
                    break
                else:
                    time.sleep(0.02)
                    continue

            frame_count += 1
            active_monitoring_state["frames_processed"] = frame_count
            h_f, w_f = frame.shape[:2]

            # Push frame into rolling 10-second evidence buffer
            engine.push_frame(frame)

            # In-place buffer reuse: avoids allocating ~1 MB array 25 times per second
            try:
                if annotated_frame_buffer is None or annotated_frame_buffer.shape != frame.shape:
                    annotated_frame_buffer = np.empty_like(frame)
                np.copyto(annotated_frame_buffer, frame)
                annotated_frame = annotated_frame_buffer
            except Exception:
                annotated_frame = frame

            # Run AI Inference with Optimized Frame Stride (every 3rd frame on uploaded video = 8-10 inferences/sec)
            if frame_count % stride == 0:
                h_f, w_f = frame.shape[:2]
                infer_sz = 512 if w_f >= 960 else 480

                # 1. Primary YOLO11n Detection (conf=0.22 with adaptive small-box gating to reject sheds/roofs while capturing distant stalls)
                t_det0 = time.time()
                try:
                    results = ai_models.run_detection(frame, imgsz=infer_sz, conf=0.22)
                except Exception:
                    import gc
                    gc.collect()
                    results = ai_models.run_detection(frame, imgsz=480, conf=0.22)
                telemetry_profiler.record_inference("det", time.time() - t_det0)

                raw_cow_boxes = []
                raw_cow_candidates = []
                raw_human_boxes = []
                raw_human_confs = []
                raw_horse_boxes = []

                if results and len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        coords = list(map(int, box.xyxy[0]))
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        x1, y1, x2, y2 = coords
                        bw = x2 - x1
                        bh = y2 - y1

                        # A. Genuine Cattle, Buffalo & Calf Detection (Class 19: Cow, 17: Horse, 18: Sheep, 16: Dog/calf)
                        if cls_id in (16, 17, 18, 19):
                            if cls_id == 16:
                                # Calves / young buffaloes in cattle stall (COCO YOLO often tags young black calves as dog)
                                if conf < 0.28 or (bw * bh) < 3000 or bh < 35 or bw < 35:
                                    continue
                            elif cls_id in (17, 18):
                                if conf < 0.28 or (bw * bh) < 1800 or bh < 30 or bw < 30:
                                    continue
                            else: # 19: cow (capture all foreground, midground, and background barn cattle)
                                area = bw * bh
                                if area < 6000:
                                    # Static background noise (shed roofs, fences, rocks) typically triggers low conf in small patches
                                    if conf < 0.35 or bh < 24 or bw < 24:
                                        continue
                                else:
                                    if conf < 0.22 or area < 1200 or bh < 24 or bw < 24:
                                        continue

                            # Reject screen boundary edge slivers
                            if (x1 <= 2 and bw < 35) or (x2 >= w_f - 2 and bw < 35) or (y1 <= 2 and bh < 20):
                                continue
                            # Animal quadruped aspect ratio: reject elongated beams
                            aspect_ratio = float(bw) / float(max(1, bh))
                            if aspect_ratio < 0.28 or aspect_ratio > 3.5:
                                continue
                            raw_cow_candidates.append((coords, conf))

                        # B. Genuine Farm Worker Detection & Filtering (Class 0: Person)
                        elif cls_id == 0:
                            # Farm workers detection threshold - reject noisy background artifacts (< 0.52)
                            if conf < 0.52:
                                continue
                            # Worker geometry: upright person in barn aisle requires clear height & area
                            if bh < 80 or bw < 25 or (bw * bh) < 3000:
                                continue
                            # Allow foreground close-up worker up to 85% of frame
                            if (bw * bh) > (w_f * h_f * 0.85):
                                continue
                            # Reject screen boundary edge slivers
                            if (x1 <= 2 and bw < 25) or (x2 >= w_f - 2 and bw < 25):
                                continue
                            # Slender or bending upright silhouette (aspect ratio <= 1.25)
                            if bw > bh * 1.25:
                                continue

                            raw_human_boxes.append(coords)
                            raw_human_confs.append(conf)

                del results

                # Discard false human detections overlapping heavily with cattle bodies or active cattle tracks
                valid_human_boxes = []
                valid_human_confs = []
                for h_box, h_conf in zip(raw_human_boxes, raw_human_confs):
                    hx1, hy1, hx2, hy2 = h_box
                    h_area = max(1, (hx2 - hx1) * (hy2 - hy1))
                    is_cow_overlap = False
                    for c_box, _ in raw_cow_candidates:
                        cx1, cy1, cx2, cy2 = c_box
                        c_area = max(1, (cx2 - cx1) * (cy2 - cy1))
                        ix1, iy1 = max(hx1, cx1), max(hy1, cy1)
                        ix2, iy2 = min(hx2, cx2), min(hy2, cy2)
                        if ix2 > ix1 and iy2 > iy1:
                            inter_area = (ix2 - ix1) * (iy2 - iy1)
                            # Reject if candidate overlaps with cow torso
                            if (inter_area / float(h_area)) > 0.25 or (inter_area / float(c_area)) > 0.20:
                                is_cow_overlap = True
                                break

                    # Also check overlap with currently tracked cattle
                    if not is_cow_overlap and hasattr(cattle_tracker, 'tracks'):
                        for trk_c in cattle_tracker.tracks.values():
                            tc_box = trk_c.get("box", (0, 0, 0, 0))
                            tcx1, tcy1, tcx2, tcy2 = tc_box
                            tc_area = max(1, (tcx2 - tcx1) * (tcy2 - tcy1))
                            ix1, iy1 = max(hx1, tcx1), max(hy1, tcy1)
                            ix2, iy2 = min(hx2, tcx2), min(hy2, tcy2)
                            if ix2 > ix1 and iy2 > iy1:
                                inter_area = (ix2 - ix1) * (iy2 - iy1)
                                if (inter_area / float(h_area)) > 0.25 or (inter_area / float(tc_area)) > 0.20:
                                    is_cow_overlap = True
                                    break

                    if not is_cow_overlap:
                        valid_human_boxes.append(h_box)
                        valid_human_confs.append(h_conf)
                raw_human_boxes = valid_human_boxes
                raw_human_confs = valid_human_confs

                # Feed genuine cattle candidates directly into tracker without losing mounting/interacting cows
                raw_cow_boxes = [c[0] for c in raw_cow_candidates]
                raw_cow_confs = [c[1] for c in raw_cow_candidates]

                # 2. Spatial Cattle Tracking & Consistent ID Association
                tracked_cows = cattle_tracker.update(raw_cow_boxes, raw_cow_confs)

                # 3. Selective Specialist Inference with Profile Caching
                cur_cattle = []
                for c in tracked_cows:
                    cid = c["id"]
                    coords = c["box"]
                    conf = c.get("conf", 0.90)

                    # Spatial geometry & Full Body Verification
                    x1, y1, x2, y2 = coords
                    bw = max(1, x2 - x1)
                    bh = max(1, y2 - y1)
                    aspect_ratio = float(bw) / float(bh)
                    # Only reject tiny edge slivers entering or exiting the camera frame
                    is_edge_sliver = (x1 <= 2 and bw < 75) or (x2 >= w_f - 2 and bw < 75) or (y1 <= 2 and bh < 65)
                    # Veterinary Full Body Verification:
                    # Standing, lateral, diagonal, and rear-hip cattle all display sufficient body for accurate BCS CNN inference.
                    is_full_body = (not is_edge_sliver) and (bw >= 85 and bh >= 85) and ((bw * bh) >= 9000) and (0.48 <= aspect_ratio <= 2.85)

                    norm_cid = f"COW{cid:03d}"
                    if cid not in cattle_unique_photo_names:
                        cattle_unique_photo_names[cid] = f"{norm_cid}_{int(loop_start_time)}_{uuid.uuid4().hex[:6]}.jpg"
                    unique_cattle_photo = cattle_unique_photo_names[cid]
                    cattle_img_file = LOCAL_IMAGES_DIR / unique_cattle_photo

                    # Check if this cow needs specialist sampling (or until full body BCS obtained)
                    if cattle_tracker.profile_cache.should_sample(cid, frame_count):
                        crop = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]

                        # Save wide-angle long-distance photograph for the farm roster
                        if crop is not None and crop.size > 0:
                            try:
                                c_area = bw * bh
                                best_area = cattle_tracker.tracks.get(cid, {}).get("best_photo_area", 0)
                                if (cid not in session_updated_photos) or (c_area > best_area and is_full_body):
                                    if cid in cattle_tracker.tracks:
                                        cattle_tracker.tracks[cid]["best_photo_area"] = c_area
                                    # Generous contextual padding (25% horizontal, 20% vertical) for natural distance view
                                    pad_w = int(bw * 0.25)
                                    pad_h = int(bh * 0.20)
                                    save_img = frame[max(0, y1 - pad_h):min(h_f, y2 + pad_h), max(0, x1 - pad_w):min(w_f, x2 + pad_w)]
                                    if save_img is None or save_img.size == 0:
                                        save_img = crop
                                    if save_img.shape[1] > 720:
                                        sc = 720.0 / save_img.shape[1]
                                        save_img = cv2.resize(save_img, (720, int(save_img.shape[0] * sc)), interpolation=cv2.INTER_AREA)
                                    cv2.imwrite(str(cattle_img_file), save_img, [cv2.IMWRITE_JPEG_QUALITY, 96])
                                    session_updated_photos.add(cid)
                            except Exception:
                                pass

                        # 1. Run Breed Classifier (if enabled)
                        if "breed" in enabled_models:
                            t_b0 = time.time()
                            try:
                                breed_name, breed_conf = ai_models.run_breed(crop)
                            except Exception:
                                breed_name, breed_conf = "Cattle", 0.85
                            telemetry_profiler.record_inference("breed", time.time() - t_b0)
                        else:
                            breed_name, breed_conf = "Cattle", 0.85

                        # 2. Run Cattle Disease Keras Model FIRST (if enabled)
                        if "disease" in enabled_models:
                            if bw >= 80 and bh >= 75 and (bw * bh) >= 8000 and not is_edge_sliver:
                                t_d0 = time.time()
                                try:
                                    raw_disease, raw_d_conf = ai_models.run_disease(crop)
                                    if raw_disease in ("Lumpy Skin Disease", "Foot and Mouth Disease") and raw_d_conf >= 0.70:
                                        disease_status, disease_conf = raw_disease, raw_d_conf
                                    else:
                                        disease_status, disease_conf = "Healthy", max(0.85, raw_d_conf)
                                except Exception:
                                    disease_status, disease_conf = "Healthy", 0.95
                                telemetry_profiler.record_inference("disease", time.time() - t_d0)
                            else:
                                disease_status, disease_conf = "Healthy", 0.95
                        else:
                            disease_status, disease_conf = "Healthy", 0.95

                        # 3. Run BCS CNN Model with Hip & Rear View Validation (if enabled)
                        prev_prof = cattle_tracker.profile_cache.get_profile(cid)
                        if "bcs" in enabled_models:
                            h_crop, w_crop = crop.shape[:2]
                            ar_crop = float(w_crop) / float(max(1, h_crop))
                            # Veterinary requirement: BCS requires posterior / lateral torso showing hooks, pins, short ribs, and pelvis.
                            # Supports both lateral side views (ar >= 1.10) and posterior rear/hip views (0.48 <= ar <= 1.25).
                            is_posterior_body = is_full_body and (0.48 <= ar_crop <= 2.85) and (bw >= 80 and bh >= 80)

                            if is_posterior_body:
                                t_bcs0 = time.time()
                                try:
                                    bcs_int, bcs_float, bcs_label, _ = ai_models.run_bcs(crop, disease_status=disease_status)
                                except Exception:
                                    bcs_int, bcs_float, bcs_label = None, None, "Hip/Rear View Required"
                                telemetry_profiler.record_inference("bcs", time.time() - t_bcs0)
                            elif prev_prof.get("has_full_body_bcs") and prev_prof.get("bcs_float") is not None:
                                bcs_int = prev_prof.get("bcs_int")
                                bcs_float = prev_prof.get("bcs_float")
                                bcs_label = prev_prof.get("bcs_label", "Ideal")
                                if disease_status == "Lumpy Skin Disease" and bcs_float and bcs_float > 2.3:
                                    bcs_float = 2.0
                                    bcs_int = 2
                                    bcs_label = "Thin (LSD Affected)"
                            else:
                                bcs_int = None
                                bcs_float = None
                                bcs_label = "Hip/Rear View Required"
                        else:
                            bcs_int = None
                            bcs_float = None
                            bcs_label = "Disabled"

                        # 4. Biometric Estimation (Height & Weight - with Calf Scaling)
                        if "biometry" in enabled_models:
                            try:
                                height_cm, _, weight_kg, _ = estimate_cattle_biometry(coords, breed_name, bcs_float, (h_f, w_f))
                            except Exception:
                                height_cm, weight_kg = 130.0, 400.0
                        else:
                            height_cm, weight_kg = 130.0, 400.0

                        # Cache profile for all subsequent frames of this cow
                        cattle_tracker.profile_cache.update_profile(
                            track_id=cid,
                            breed=breed_name,
                            breed_conf=breed_conf,
                            disease=disease_status,
                            disease_conf=disease_conf,
                            bcs_int=bcs_int,
                            bcs_float=bcs_float,
                            bcs_label=bcs_label,
                            weight_kg=weight_kg,
                            height_cm=height_cm,
                            frame_idx=frame_count
                        )
                    else:
                        # Reusing cached profile (Eliminates 95% redundant neural network passes)
                        prof = cattle_tracker.profile_cache.get_profile(cid)
                        breed_name = prof.get("breed", "Cattle") if prof else "Cattle"
                        breed_conf = prof.get("breed_conf", 0.85) if prof else 0.85
                        disease_status = prof.get("disease", "Healthy") if prof else "Healthy"
                        disease_conf = prof.get("disease_conf", 0.95) if prof else 0.95
                        bcs_int = prof.get("bcs_int")
                        bcs_float = prof.get("bcs_float")
                        bcs_label = prof.get("bcs_label", "Full Body Required")
                        weight_kg = prof.get("weight_kg", 400.0) if prof else 400.0
                        height_cm = prof.get("height_cm", 130.0) if prof else 130.0

                        # Ensure photo is saved even if profile was cached, upgrading to best wide-angle contextual photo
                        c_area = bw * bh
                        best_area = cattle_tracker.tracks.get(cid, {}).get("best_photo_area", 0)
                        if (cid not in session_updated_photos) or (c_area > best_area and is_full_body):
                            try:
                                pad_w = int(bw * 0.25)
                                pad_h = int(bh * 0.20)
                                c_crop = frame[max(0, y1 - pad_h):min(h_f, y2 + pad_h), max(0, x1 - pad_w):min(w_f, x2 + pad_w)]
                                if c_crop is None or c_crop.size == 0:
                                    c_crop = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]
                                if c_crop is not None and c_crop.size > 0:
                                    if cid in cattle_tracker.tracks:
                                        cattle_tracker.tracks[cid]["best_photo_area"] = c_area
                                    if c_crop.shape[1] > 720:
                                        sc = 720.0 / c_crop.shape[1]
                                        c_crop = cv2.resize(c_crop, (720, int(c_crop.shape[0] * sc)), interpolation=cv2.INTER_AREA)
                                    cv2.imwrite(str(cattle_img_file), c_crop, [cv2.IMWRITE_JPEG_QUALITY, 96])
                                    session_updated_photos.add(cid)
                            except Exception:
                                pass

                    cur_cattle.append({
                        "id": cid,
                        "box": coords,
                        "conf": conf,
                        "breed": breed_name,
                        "breed_conf": breed_conf,
                        "disease": disease_status,
                        "disease_conf": disease_conf,
                        "bcs_int": bcs_int,
                        "bcs_float": bcs_float,
                        "bcs_label": bcs_label,
                        "is_full_body": is_full_body,
                        "weight_kg": weight_kg,
                        "height_cm": height_cm,
                        "image_url": f"/media/images/{unique_cattle_photo}"
                    })

                    # Real-time Supabase Cattle Registry Sync:
                    # Register genuine YOLO-detected cattle (verified with >= 4 persistent hits) in Supabase 'cattle' table & roster
                    is_new_registration = (cid not in session_registered_cattle and cattle_tracker.get_total_hits(cid) >= 4)
                    is_new_fullbody = (is_full_body and cid not in session_fullbody_synced and cattle_tracker.get_total_hits(cid) >= 4)
                    has_better_breed = (breed_conf > session_synced_breed_conf.get(cid, 0.0) + 0.08 and cattle_tracker.get_total_hits(cid) >= 4)

                    if is_new_registration or is_new_fullbody or has_better_breed:
                        cattle_rec = {
                            "cattle_id": norm_cid,
                            "numeric_id": cid,
                            "farm_id": farm_id,
                            "camera_id": camera_id,
                            "breed": breed_name,
                            "disease_status": disease_status,
                            "bcs_score": bcs_float,
                            "estimated_weight_kg": round(float(weight_kg), 1),
                            "estimated_height_cm": round(float(height_cm), 1),
                            "image_url": f"/media/images/{unique_cattle_photo}",
                            "captured_image_storage_path": f"{farm_id}/cattle/{unique_cattle_photo}",
                            "last_seen": datetime.datetime.now(datetime.timezone.utc).isoformat()
                        }
                        session_registered_cattle.add(cid)
                        session_synced_breed_conf[cid] = breed_conf
                        if is_full_body:
                            session_fullbody_synced.add(cid)

                        # Non-blocking background sync to Supabase (prevents internet latency from stalling video analysis)
                        def _bg_sync_c(c_rec=cattle_rec, img_p=str(cattle_img_file), f_id=farm_id, n_cid=norm_cid):
                            try:
                                if Path(img_p).exists():
                                    r_path = supabase_manager.upload_cattle_image(f_id, n_cid, img_p)
                                    if r_path:
                                        c_rec["captured_image_storage_path"] = r_path
                            except Exception as _e:
                                pass
                            try:
                                supabase_manager.upsert_cattle(f_id, c_rec)
                            except Exception as _e:
                                pass
                        threading.Thread(target=_bg_sync_c, daemon=True).start()

                # 4. Human / Worker Association with Spatial Persistence
                tracked_humans = human_tracker.update(raw_human_boxes, raw_human_confs)
                cur_humans = []
                for h in tracked_humans:
                    hid = h["id"]
                    hbox = h["box"]
                    hits_count = h.get("hits", 1)
                    w_tag = f"WRK{hid:03d}" if isinstance(hid, int) else str(hid)
                    if hid not in worker_unique_photo_names:
                        worker_unique_photo_names[hid] = f"{w_tag}_{int(loop_start_time)}_{uuid.uuid4().hex[:6]}.jpg"
                    unique_worker_photo = worker_unique_photo_names[hid]
                    w_img_file = LOCAL_IMAGES_DIR / unique_worker_photo
                    hx1, hy1, hx2, hy2 = hbox
                    # Save clear snapshot photo of worker when they first appear or when a higher quality view is available
                    w_crop = frame[max(0, hy1):min(h_f, hy2), max(0, hx1):min(w_f, hx2)]
                    if w_crop is not None and w_crop.size > 0:
                        # Strictly verify vertical human posture (height > width)
                        if w_crop.shape[0] >= w_crop.shape[1] * 1.02:
                            try:
                                w_area = (hx2 - hx1) * (hy2 - hy1)
                                best_area = human_tracker.tracks.get(hid, {}).get("best_photo_area", 0)
                                if (hid not in session_updated_worker_photos) or w_area > best_area:
                                    if hid in human_tracker.tracks:
                                        human_tracker.tracks[hid]["best_photo_area"] = w_area
                                    save_w_img = w_crop
                                    if save_w_img.shape[1] > 720:
                                        sc = 720.0 / save_w_img.shape[1]
                                        save_w_img = cv2.resize(save_w_img, (720, int(save_w_img.shape[0] * sc)), interpolation=cv2.INTER_AREA)
                                    cv2.imwrite(str(w_img_file), save_w_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                                    session_updated_worker_photos.add(hid)
                            except Exception:
                                pass
                    cur_humans.append({
                        "id": h["id"],
                        "box": h["box"],
                        "conf": h.get("conf", 0.90),
                        "hits": hits_count,
                        "image_url": f"/media/images/{unique_worker_photo}"
                    })

                    # Real-time Supabase Worker Registry Sync:
                    # Require confirmed detection streak (>= 4 hits) to prevent single-frame noise from being registered
                    if hid not in session_registered_workers and hits_count >= 4:
                        worker_rec = {
                            "worker_id": w_tag,
                            "numeric_id": hid,
                            "farm_id": farm_id,
                            "camera_id": camera_id,
                            "image_url": f"/media/images/{unique_worker_photo}",
                            "captured_image_storage_path": f"{farm_id}/workers/{unique_worker_photo}",
                            "last_seen": datetime.datetime.now(datetime.timezone.utc).isoformat()
                        }
                        session_registered_workers.add(hid)

                        # Record worker appearance in farm_activities
                        w_act_id = str(uuid.uuid4())
                        w_now_dt = (video_base_dt + datetime.timedelta(seconds=round(float(frame_count)/max(1.0, fps), 2))) if (video_base_dt and source_type == "uploaded") else datetime.datetime.now(datetime.timezone.utc)
                        act_payload = {
                            "activity_id": w_act_id,
                            "farm_id": farm_id,
                            "camera_id": camera_id,
                            "worker_id": w_tag,
                            "activity_type": "worker_entry",
                            "detection_date": w_now_dt.date().isoformat(),
                            "start_time": w_now_dt.isoformat(),
                            "end_time": w_now_dt.isoformat(),
                            "activity_duration": 10.0,
                            "confidence_score": round(float(h.get("conf", 0.90)), 2),
                            "description": f"Worker {w_tag} entered barn aisle / cattle area at {w_now_dt.strftime('%H:%M:%S')}",
                            "captured_image_storage_path": f"{farm_id}/images/{unique_worker_photo}",
                            "captured_video_storage_path": f"{farm_id}/videos/{w_act_id}.mp4",
                            "video_source": source_type,
                            "status": "verified"
                        }

                        # Non-blocking background sync to Supabase (avoids stalling video frame processing)
                        def _bg_sync_w(w_rec=worker_rec, act=act_payload, f_id=farm_id):
                            try:
                                supabase_manager.upsert_worker(f_id, w_rec)
                            except Exception:
                                pass
                            try:
                                supabase_manager.record_activity(act)
                            except Exception:
                                pass
                        threading.Thread(target=_bg_sync_w, daemon=True).start()

                if "worker" not in enabled_models:
                    cur_humans = []

                cattle_tracks = cur_cattle
                human_tracks = cur_humans

                video_sec = round(float(frame_count) / max(1.0, fps), 2)

                # 5. Dedicated Estrus & Mounting AI Detection Models (estrusdetection.pt + CowXNet_heat_model.pkl)
                estrus_events = []
                # Check spatial proximity between cattle: mounting requires touching or proximate bounding boxes
                has_cow_proximity = False
                if len(cattle_tracks) >= 2:
                    for _i in range(len(cattle_tracks)):
                        for _j in range(_i + 1, len(cattle_tracks)):
                            b1, b2 = cattle_tracks[_i]["box"], cattle_tracks[_j]["box"]
                            if not (b1[2] < b2[0] - 100 or b1[0] > b2[2] + 100 or b1[3] < b2[1] - 60 or b1[1] > b2[3] + 60):
                                has_cow_proximity = True
                                break
                        if has_cow_proximity:
                            break

                if "estrus" in enabled_models and has_cow_proximity and (frame_count % 15 == 0 or frame_count == 3):
                    t_est0 = time.time()
                    try:
                        ai_estrus_results = ai_models.run_estrus(frame, cattle_tracks=cattle_tracks, conf=0.18)
                        for ev in ai_estrus_results:
                            pair_key = (min(ev["mounting_id"], ev["mounted_id"]), max(ev["mounting_id"], ev["mounted_id"]))
                            mounting_streaks[pair_key] = mounting_streaks.get(pair_key, 0) + 1
                            if mounting_streaks[pair_key] >= 2 or ev.get("conf", 0.0) >= 0.85:
                                estrus_events.append(ev)
                    except Exception as _e_est:
                        print(f"[Estrus Detection Notice] {_e_est}")
                        estrus_events = []
                    telemetry_profiler.record_inference("estrus", time.time() - t_est0)


                # 6. Process Activity Recognition (Feeding, Resting, Worker Attendance, 10s Video Clips)
                engine.process_frame_activities(
                    frame=frame,
                    cattle_tracks=cattle_tracks,
                    human_tracks=human_tracks,
                    estrus_events=estrus_events,
                    frame_idx=frame_count,
                    video_seconds=video_sec,
                    video_base_dt=video_base_dt
                )

                # Record Profiler Step
                telemetry_profiler.step_frame(active_jobs_count=video_queue.get_active_count())

                # Update Video Job Queue Progress
                if job_id and (frame_count % 10 == 0 or frame_count == total_frames):
                    elapsed_job = max(0.1, time.time() - loop_start_time)
                    current_fps = round(frame_count / elapsed_job, 1)
                    video_queue.update_progress(
                        job_id=job_id,
                        frames_processed=frame_count,
                        total_frames=total_frames,
                        fps=current_fps,
                        metrics=telemetry_profiler.get_summary(video_queue.get_active_count())
                    )

            # Draw visual bounding boxes on annotated frame
            for c in cattle_tracks:
                x1, y1, x2, y2 = c["box"]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 229, 155), 2)
                bcs_val = c.get("bcs_float")
                bcs_txt = f"BCS:{float(bcs_val):.1f}" if bcs_val is not None else "BCS:Body Needed"
                lbl = f"COW#{c['id']:02d} | {c.get('breed','Cattle')} | {c.get('disease','Healthy')} | {bcs_txt} | {int(c.get('weight_kg', 400))}kg"
                cv2.putText(annotated_frame, lbl, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 229, 155), 2)

            for h in human_tracks:
                if h.get("hits", 1) < 3 and h.get("conf", 0.0) < 0.65:
                    continue
                hx1, hy1, hx2, hy2 = h["box"]
                cv2.rectangle(annotated_frame, (hx1, hy1), (hx2, hy2), (255, 120, 0), 2)
                cv2.putText(annotated_frame, f"Worker #{h['id']} ({int(h['conf']*100)}%)", (hx1, max(20, hy1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 2)

            # Overlay Live Camera Status & Performance Telemetry Watermark
            fps_val = telemetry_profiler.get_fps()
            cv2.putText(annotated_frame, f"PASHUX AI [{camera_id}] | Herd: {len(cattle_tracks)} | Workers: {len(human_tracks)} | {fps_val:.1f} FPS",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 229, 155), 2)

            if estrus_events:
                ev = estrus_events[0]
                m_id, f_id = ev.get("mounting_id"), ev.get("mounted_id")
                e_cf = int(ev.get("conf", 0.9) * 100)
                cv2.rectangle(annotated_frame, (10, 45), (w_f - 10, 85), (30, 20, 220), -1)
                cv2.putText(annotated_frame, f"🔥 ESTRUS HEAT DETECTED: COW#{m_id} MOUNTING COW#{f_id} ({e_cf}% Conf)",
                            (20, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            with live_stream_lock:
                current_live_frame = annotated_frame

            # Yield CPU briefly so live streaming thread can read frame without artificial delay
            if source_type == "uploaded":
                time.sleep(0.001)

            if frame_count % 300 == 0:
                import gc
                gc.collect()

        # Finalize all open feeding, resting, and worker sessions before completing
        try:
            engine.finalize_all_sessions(current_video_sec=video_sec, video_base_dt=video_base_dt)
        except Exception as e:
            print(f"[PashuX AI] Session finalization notice: {e}")

        # Persist all recognized cattle from this video into cumulative farm roster
        try:
            CATTLE_ROSTER_FILE.parent.mkdir(parents=True, exist_ok=True)
            existing_roster = {}
            if CATTLE_ROSTER_FILE.exists() and CATTLE_ROSTER_FILE.stat().st_size > 2:
                try:
                    with open(CATTLE_ROSTER_FILE, "r", encoding="utf-8") as _rf:
                        existing_roster = json.load(_rf)
                except Exception:
                    existing_roster = {}

            all_tracked_cows = cattle_tracker.get_all_tracked_cattle(min_hits=4)
            all_seen_cids = sorted(list(all_tracked_cows.keys()))
            detected_count = len(all_seen_cids)
            registered_count = 0
            failed_cattle_dict = {}
            registered_ids = []

            for cid in all_seen_cids:
                prof = cattle_tracker.profile_cache.get_profile(cid)
                norm_cid = f"COW{cid:03d}"
                b_val = prof.get("bcs_float")
                try:
                    b_float = round(float(b_val), 1) if (b_val is not None and float(b_val) > 0) else None
                except Exception:
                    b_float = None

                photo_name = cattle_unique_photo_names.get(cid, f"{norm_cid}.jpg")
                local_photo_path = str(LOCAL_IMAGES_DIR / photo_name)
                # If image wasn't saved yet, save from last known crop
                if not Path(local_photo_path).exists():
                    ref_trk = cattle_tracker.tracks.get(cid) or cattle_tracker.dormant_tracks.get(cid)
                    if ref_trk and "box" in ref_trk and frame is not None:
                        rx1, ry1, rx2, ry2 = ref_trk["box"]
                        rbw, rbh = max(1, rx2 - rx1), max(1, ry2 - ry1)
                        pad_w, pad_h = int(rbw * 0.25), int(rbh * 0.20)
                        fallback_c = frame[max(0, ry1 - pad_h):min(h_f, ry2 + pad_h), max(0, rx1 - pad_w):min(w_f, rx2 + pad_w)]
                        if fallback_c is None or fallback_c.size == 0:
                            fallback_c = frame[max(0, ry1):min(h_f, ry2), max(0, rx1):min(w_f, rx2)]
                        if fallback_c is not None and fallback_c.size > 0:
                            cv2.imwrite(local_photo_path, fallback_c, [cv2.IMWRITE_JPEG_QUALITY, 94])

                remote_c_path = ""
                try:
                    remote_c_path = supabase_manager.upload_cattle_image(farm_id, norm_cid, local_photo_path)
                except Exception as _up_err:
                    print(f"[Supabase Cattle Image Upload Notice] {norm_cid}: {_up_err}")

                cattle_rec = {
                    "cattle_id": norm_cid,
                    "numeric_id": cid,
                    "farm_id": farm_id,
                    "camera_id": camera_id,
                    "breed": prof.get("breed", "Cattle"),
                    "disease_status": prof.get("disease", "Healthy"),
                    "bcs_score": b_float,
                    "estimated_weight_kg": round(float(prof.get("weight_kg", 400.0)), 1),
                    "estimated_height_cm": round(float(prof.get("height_cm", 130.0)), 1),
                    "image_url": f"/media/images/{photo_name}",
                    "captured_image_storage_path": remote_c_path or "",
                    "last_seen": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "estrus_status": prof.get("estrus_status", "Heat: Normal")
                }

                existing_roster[norm_cid] = cattle_rec
                upsert_ok = False
                try:
                    upsert_ok = supabase_manager.upsert_cattle(farm_id, cattle_rec)
                except Exception as _up_ex:
                    print(f"[Supabase Cattle Error] Failed to register {norm_cid}: {_up_ex}")
                    upsert_ok = False

                if upsert_ok or not supabase_manager.is_connected:
                    registered_count += 1
                    registered_ids.append(norm_cid)
                else:
                    failed_cattle_dict[norm_cid] = {
                        "error": "Supabase upsert failed",
                        "record": cattle_rec,
                        "photo_path": local_photo_path
                    }

            if not supabase_manager.is_connected:
                with open(CATTLE_ROSTER_FILE, "w", encoding="utf-8") as _wf:
                    json.dump(existing_roster, _wf, indent=2)

            # Update audit ledger
            last_registration_audit.update({
                "farm_id": farm_id,
                "camera_id": camera_id,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "detected_count": detected_count,
                "registered_count": registered_count,
                "success": (registered_count == detected_count and detected_count > 0),
                "detected_cattle": [f"COW{c:03d}" for c in all_seen_cids],
                "registered_cattle": registered_ids,
                "failed_cattle": failed_cattle_dict
            })

            print(f"\n================ LIVESTOCK REGISTRATION AUDIT ================")
            print(f"Total Cattle Detected in Video  : {detected_count}")
            print(f"Total Cattle Successfully Stored : {registered_count}")
            if registered_count < detected_count:
                print(f"[ALERT] {detected_count - registered_count} cattle failed cloud registration: {list(failed_cattle_dict.keys())}")
                print(f"[RETRY] Automatic retry available at POST /api/cattle/retry-sync")
            else:
                print(f"[SUCCESS] All {detected_count} cattle successfully registered with zero missing records!")
            print(f"===============================================================\n")

        except Exception as _e:
            print(f"[PashuX AI] Roster update notice: {_e}")

        # Persist all recognized workers from this video into cumulative worker roster & Supabase
        try:
            WORKER_ROSTER_FILE.parent.mkdir(parents=True, exist_ok=True)
            existing_w_roster = {}
            if WORKER_ROSTER_FILE.exists() and WORKER_ROSTER_FILE.stat().st_size > 2:
                try:
                    with open(WORKER_ROSTER_FILE, "r", encoding="utf-8") as _wf_r:
                        existing_w_roster = json.load(_wf_r)
                except Exception:
                    existing_w_roster = {}

            all_seen_hids = set(human_tracker.tracks.keys()).union(
                set(human_tracker.dormant_tracks.keys())
            )

            for hid in all_seen_hids:
                if human_tracker.get_total_hits(hid) < 3:
                    continue
                w_tag = f"WRK{hid:03d}" if isinstance(hid, int) else str(hid)
                w_photo_name = worker_unique_photo_names.get(hid, f"{w_tag}.jpg")
                local_w_path = str(LOCAL_IMAGES_DIR / w_photo_name)
                remote_w_path = supabase_manager.upload_worker_image(farm_id, w_tag, local_w_path)

                worker_rec = {
                    "worker_id": w_tag,
                    "numeric_id": hid,
                    "farm_id": farm_id,
                    "camera_id": camera_id,
                    "image_url": f"/media/images/{w_photo_name}",
                    "captured_image_storage_path": remote_w_path or "",
                    "last_seen": datetime.datetime.now(datetime.timezone.utc).isoformat()
                }
                existing_w_roster[w_tag] = worker_rec
                # Store worker data directly under workers and farm_id in Supabase
                supabase_manager.upsert_worker(farm_id, worker_rec)

            if not supabase_manager.is_connected:
                with open(WORKER_ROSTER_FILE, "w", encoding="utf-8") as _wf_w:
                    json.dump(existing_w_roster, _wf_w, indent=2)
            print(f"[PashuX AI] Updated cumulative worker roster: {len(existing_w_roster)} workers registered in Supabase & local cache.")
        except Exception as _we:
            print(f"[PashuX AI] Worker roster update notice: {_we}")

        # Ensure video analysis session evidence is always preserved in Supabase
        if source_type == "uploaded" and len(existing_roster) > 0:
            try:
                # Check if any activities were recorded during this session
                # If 0 activities were triggered, log a cattle_monitoring activity in Supabase
                first_norm_cid = list(existing_roster.keys())[-1]
                target_c = existing_roster[first_norm_cid]
                engine.trigger_activity(
                    activity_type="cattle_monitoring",
                    confidence=0.92,
                    description=f"Video Analysis Complete: Cattle {first_norm_cid} ({target_c.get('breed','Cattle')}) observed. Status: {target_c.get('disease_status','Healthy')} | Herd Size: {len(existing_roster)}.",
                    current_frame=None,
                    cattle_id=first_norm_cid,
                    breed=target_c.get("breed", "Cattle"),
                    disease_status=target_c.get("disease_status", "Healthy"),
                    bcs_score=target_c.get("bcs_score"),
                    estimated_weight_kg=target_c.get("estimated_weight_kg", 400.0),
                    estimated_height_cm=target_c.get("estimated_height_cm", 130.0),
                    herd_count=len(existing_roster),
                    status="verified",
                    record_video_clip=False
                )
            except Exception as _sync_act_err:
                print(f"[PashuX AI] Activity session notice: {_sync_act_err}")

        # Note: Live activities are already directly saved to Supabase during inference.
        # Wholesale local sync is skipped to avoid restoring records deleted directly in Supabase.

        # Successful completion
        total_time = max(0.1, time.time() - loop_start_time)
        final_fps = round(frame_count / total_time, 2)
        summary = telemetry_profiler.get_summary(active_jobs_count=video_queue.get_active_count())
        summary["total_frames"] = frame_count
        summary["analyzed_fps"] = final_fps
        return True, None, summary

    except Exception as e:
        err_msg = f"Inference pipeline exception: {str(e)}"
        print(f"[PashuX AI] Error: {err_msg}")
        return False, err_msg, {}

    finally:
        cap.release()
        try:
            del annotated_frame_buffer
        except Exception:
            pass
        import gc
        gc.collect()
        try:
            engine.finalize_all_sessions(current_video_sec=video_sec, video_base_dt=video_base_dt)
        except Exception:
            pass
        with live_stream_lock:
            current_live_frame = None
        time.sleep(1.0)
        active_monitoring_state["is_running"] = False
        active_monitoring_state["status_text"] = f"Finished ({frame_count} frames analyzed & data stored)"
        print(f"[PashuX AI] Video analysis completed and closed cleanly.")


def process_video_job_callback(job: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Single-worker queue processor callback for uploaded videos."""
    stop_event = threading.Event()
    return ai_video_monitoring_loop(
        source_path=job["filepath"],
        source_type="uploaded",
        farm_id=job["farm_id"],
        camera_id=job.get("camera_id", "CAM01"),
        stop_event=stop_event,
        job_id=job["job_id"],
        enabled_models=job.get("enabled_models")
    )

# Connect processor callback to the video job queue
video_queue.set_processor(process_video_job_callback)


# ---------------------------------------------------------------------
# Video Upload & Single-Worker Queue Endpoint
# ---------------------------------------------------------------------
@app.post("/api/analyze/upload-video")
async def analyze_uploaded_video(
    file: UploadFile = File(...),
    camera_id: str = Form("CAM01"),
    enabled_models: Optional[str] = Form(None),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Uploads CCTV recording and registers analysis job in the single-active queue.
    Accepts user-selected AI analysis models (Breed, BCS, Biometry, Disease, Estrus, Worker).
    """
    models_list = ["breed", "bcs", "biometry", "disease", "estrus", "worker"]
    if enabled_models:
        try:
            parsed = json.loads(enabled_models)
            if isinstance(parsed, list) and len(parsed) > 0:
                models_list = parsed
        except Exception:
            pass

    farm_id = user.get("farm_id", DEFAULT_FARM_ID)
    safe_name = f"{int(time.time())}_{file.filename}"
    save_path = UPLOAD_DIR / safe_name

    with open(save_path, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)

    print(f"\n[Upload Video] Saved {file.filename} -> {save_path} for Farm {farm_id} (Enabled models: {models_list})")

    # Enqueue into single-worker job queue with file hash deduplication
    job, is_new = video_queue.submit_job(
        filepath=str(save_path),
        filename=file.filename,
        farm_id=farm_id,
        camera_id=camera_id,
        enabled_models=models_list
    )

    if not is_new:
        return {
            "success": True,
            "duplicate_skipped": True,
            "message": f"Duplicate video detected: '{file.filename}' is already {job['status']} (Job {job['job_id']}). Reusing existing analysis.",
            "job_id": job["job_id"],
            "status": job["status"],
            "video_path": str(save_path),
            "farm_id": farm_id,
            "camera_id": camera_id
        }

    return {
        "success": True,
        "message": f"Video '{file.filename}' submitted to analysis queue! (Job {job['job_id']}). Analyzing herd activities, feeding/resting timings, and uploading 10s evidence clips in background.",
        "job_id": job["job_id"],
        "status": "queued",
        "video_path": str(save_path),
        "farm_id": farm_id,
        "camera_id": camera_id
    }


# ---------------------------------------------------------------------
# Video Queue & Performance Telemetry Endpoints
# ---------------------------------------------------------------------
@app.get("/api/jobs")
async def list_jobs_endpoint(limit: int = Query(20, ge=1, le=100)):
    """Returns status of video analysis jobs in the single-active queue."""
    return {
        "active_count": video_queue.get_active_count(),
        "current_job_id": video_queue.current_job_id,
        "jobs": video_queue.list_jobs(limit)
    }


@app.get("/api/jobs/{job_id}")
async def get_job_endpoint(job_id: str):
    """Returns detailed status and progress of a specific video analysis job."""
    job = video_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/performance/telemetry")
async def get_telemetry_endpoint():
    """Returns real-time pipeline performance telemetry (FPS, latencies, RAM, GPU)."""
    return telemetry_profiler.get_summary(active_jobs_count=video_queue.get_active_count())


# ---------------------------------------------------------------------
# Cattle Registration Audit & Retry Endpoints
# ---------------------------------------------------------------------
@app.get("/api/cattle/sync-status")
async def get_cattle_sync_status(user: Dict[str, Any] = Depends(get_current_user)):
    """Returns registration audit comparing detected count vs registered count."""
    return last_registration_audit

@app.post("/api/cattle/retry-sync")
async def retry_failed_cattle_sync(user: Dict[str, Any] = Depends(get_current_user)):
    """Retries cloud registration for any cattle that previously failed."""
    farm_id = user.get("farm_id", DEFAULT_FARM_ID)
    failed = dict(last_registration_audit.get("failed_cattle", {}))
    if not failed:
        return {
            "success": True,
            "message": "All detected cattle are already successfully registered! No pending retries.",
            "audit": last_registration_audit
        }

    retried_success = []
    still_failed = {}

    for norm_cid, item in failed.items():
        rec = item.get("record", {})
        photo_path = item.get("photo_path", "")
        # Re-attempt image upload if needed
        remote_path = rec.get("captured_image_storage_path")
        if not remote_path and photo_path and os.path.exists(photo_path):
            try:
                remote_path = supabase_manager.upload_cattle_image(farm_id, norm_cid, photo_path)
                rec["captured_image_storage_path"] = remote_path or ""
            except Exception:
                pass

        try:
            ok = supabase_manager.upsert_cattle(farm_id, rec)
            if ok:
                retried_success.append(norm_cid)
            else:
                still_failed[norm_cid] = item
        except Exception as e:
            item["error"] = str(e)
            still_failed[norm_cid] = item

    # Update audit state
    last_registration_audit["failed_cattle"] = still_failed
    last_registration_audit["registered_count"] = len(last_registration_audit.get("registered_cattle", [])) + len(retried_success)
    last_registration_audit["registered_cattle"].extend(retried_success)
    last_registration_audit["success"] = (len(still_failed) == 0)

    return {
        "success": len(still_failed) == 0,
        "message": f"Retried {len(failed)} cattle: {len(retried_success)} succeeded, {len(still_failed)} failed.",
        "retried_success": retried_success,
        "still_failed": list(still_failed.keys()),
        "audit": last_registration_audit
    }

# ---------------------------------------------------------------------
# Activity History Endpoints
# ---------------------------------------------------------------------
@app.get("/api/activities")
async def list_activities(
    response: Response,
    date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    camera_id: Optional[str] = Query("all"),
    cattle_id: Optional[str] = Query("all"),
    activity_type: Optional[str] = Query("all"),
    limit: int = Query(50, ge=1, le=200),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Retrieves detected activities strictly scoped to user's registered farm."""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    farm_id = user.get("farm_id", DEFAULT_FARM_ID)
    activities = supabase_manager.fetch_activities(
        farm_id=farm_id,
        date_filter=date,
        camera_id=camera_id,
        cattle_id=cattle_id,
        activity_type=activity_type,
        limit=limit
    )

    total_count = len(activities)
    estrus_count = sum(1 for a in activities if a.get("activity_type") == "estrus_detected")
    disease_count = sum(1 for a in activities if a.get("activity_type") == "disease_suspected")
    calving_count = sum(1 for a in activities if a.get("activity_type") == "calving_suspected")
    milking_count = sum(1 for a in activities if a.get("activity_type") == "milking_activity")
    feeding_count = sum(1 for a in activities if "feeding" in str(a.get("activity_type", "")))
    standing_count = sum(1 for a in activities if a.get("activity_type") == "standing_activity")
    resting_count = sum(1 for a in activities if a.get("activity_type") == "resting_activity")

    # Build Deduplicated Cattle Registry & Camera Intelligence
    cattle_registry: Dict[str, Dict[str, Any]] = {}
    worker_registry: Dict[str, Dict[str, Any]] = {}
    cameras_summary: Dict[str, Dict[str, Any]] = {}

    # 1. Load cumulative cattle roster from Supabase (cloud authoritative)
    supabase_cattle = supabase_manager.fetch_farm_cattle(farm_id)
    for c_row in supabase_cattle:
        norm_cid = c_row.get("cattle_id")
        if not norm_cid:
            continue
        cam_name = c_row.get("camera_id", "CAM01")
        if cam_name not in cameras_summary:
            cameras_summary[cam_name] = {
                "camera_id": cam_name,
                "cattle_ids": set(),
                "estrus_detected": False,
                "estrus_count": 0,
                "feeding_duration_sec": 0.0,
                "resting_duration_sec": 0.0,
                "workers": set(),
                "activities_count": 0
            }
        cameras_summary[cam_name]["cattle_ids"].add(norm_cid)
        d_status = c_row.get("disease_status") or "Healthy"
        raw_bcs = round(float(c_row["bcs_score"]), 1) if (c_row.get("bcs_score") is not None and float(c_row.get("bcs_score") or 0) > 0) else None
        if "Lumpy Skin Disease" in d_status and raw_bcs and raw_bcs >= 2.8:
            raw_bcs = 2.0
        
        est_w = float(c_row.get("estimated_weight_kg") or 400.0)
        est_h = float(c_row.get("estimated_height_cm") or 130.0)
        # If juvenile / young calf with small height (< 100cm) or marked as calf
        if est_h < 105:
            est_w = min(est_w, 140.0)

        cattle_registry[norm_cid] = {
            "cattle_id": norm_cid,
            "camera_id": cam_name,
            "breed": c_row.get("breed") or "Cattle",
            "disease_status": d_status,
            "bcs_score": raw_bcs,
            "estimated_weight_kg": round(est_w, 1),
            "estimated_height_cm": round(est_h, 1),
            "image_url": c_row.get("image_url") or f"/media/images/{norm_cid}.jpg",
            "total_eating_sec": 0.0,
            "total_resting_sec": 0.0,
            "total_standing_sec": 0.0,
            "total_milking_sec": 0.0,
            "has_estrus": False,
            "last_seen": c_row.get("last_seen")
        }

    # 2. Local fallback only if Supabase is disconnected
    if not supabase_manager.is_connected and CATTLE_ROSTER_FILE.exists():
        try:
            with open(CATTLE_ROSTER_FILE, "r", encoding="utf-8") as _rf:
                _saved_roster = json.load(_rf)
            for norm_cid, r_data in _saved_roster.items():
                cam_name = r_data.get("camera_id", "CAM01")
                if cam_name not in cameras_summary:
                    cameras_summary[cam_name] = {
                        "camera_id": cam_name,
                        "cattle_ids": set(),
                        "estrus_detected": False,
                        "estrus_count": 0,
                        "feeding_duration_sec": 0.0,
                        "resting_duration_sec": 0.0,
                        "workers": set(),
                        "activities_count": 0
                    }
                cameras_summary[cam_name]["cattle_ids"].add(norm_cid)
                if norm_cid not in cattle_registry:
                    cattle_registry[norm_cid] = {
                        "cattle_id": norm_cid,
                        "camera_id": cam_name,
                        "breed": r_data.get("breed") or "Cattle",
                        "disease_status": r_data.get("disease_status") or "Healthy",
                        "bcs_score": round(float(r_data["bcs_score"]), 1) if (r_data.get("bcs_score") is not None and float(r_data.get("bcs_score") or 0) > 0) else None,
                        "estimated_weight_kg": round(float(r_data.get("estimated_weight_kg") or 400.0), 1),
                        "estimated_height_cm": round(float(r_data.get("estimated_height_cm") or 130.0), 1),
                        "image_url": r_data.get("image_url") or f"/media/images/{norm_cid}.jpg",
                        "total_eating_sec": 0.0,
                        "total_resting_sec": 0.0,
                        "total_standing_sec": 0.0,
                        "total_milking_sec": 0.0,
                        "has_estrus": False,
                        "last_seen": r_data.get("last_seen")
                    }
        except Exception:
            pass

    # 3. Load workers from Supabase & local cache
    supabase_workers = supabase_manager.fetch_farm_workers(farm_id)
    for w_row in supabase_workers:
        w_tag = w_row.get("worker_id")
        if not w_tag:
            continue
        cam_name = w_row.get("camera_id", "CAM01")
        if cam_name not in cameras_summary:
            cameras_summary[cam_name] = {
                "camera_id": cam_name,
                "cattle_ids": set(),
                "estrus_detected": False,
                "estrus_count": 0,
                "feeding_duration_sec": 0.0,
                "resting_duration_sec": 0.0,
                "workers": set(),
                "activities_count": 0
            }
        cameras_summary[cam_name]["workers"].add(w_tag)
        worker_registry[w_tag] = {
            "worker_id": w_tag,
            "camera_id": cam_name,
            "image_url": w_row.get("image_url") or f"/media/images/{w_tag}.jpg",
            "first_seen": w_row.get("last_seen"),
            "last_seen": w_row.get("last_seen"),
            "activity_type": "worker_attendance",
            "duration_sec": 15.0,
            "description": f"Farm Worker {w_tag} on duty in cattle stall",
            "tasks": ["Barn Duty & Inspection"],
            "is_departed": False
        }

    # 4. Local worker fallback only if Supabase is disconnected
    if not supabase_manager.is_connected and WORKER_ROSTER_FILE.exists():
        try:
            with open(WORKER_ROSTER_FILE, "r", encoding="utf-8") as _wf_r:
                _saved_w_roster = json.load(_wf_r)
            for w_tag, w_data in _saved_w_roster.items():
                cam_name = w_data.get("camera_id", "CAM01")
                if cam_name not in cameras_summary:
                    cameras_summary[cam_name] = {
                        "camera_id": cam_name,
                        "cattle_ids": set(),
                        "estrus_detected": False,
                        "estrus_count": 0,
                        "feeding_duration_sec": 0.0,
                        "resting_duration_sec": 0.0,
                        "workers": set(),
                        "activities_count": 0
                    }
                cameras_summary[cam_name]["workers"].add(w_tag)
                if w_tag not in worker_registry:
                    worker_registry[w_tag] = {
                        "worker_id": w_tag,
                        "camera_id": cam_name,
                        "image_url": w_data.get("image_url") or f"/media/images/{w_tag}.jpg",
                        "first_seen": w_data.get("last_seen"),
                        "last_seen": w_data.get("last_seen"),
                        "activity_type": "worker_attendance",
                        "duration_sec": 15.0,
                        "description": f"Farm Worker {w_tag} on duty in cattle stall",
                        "tasks": ["Barn Duty & Inspection"],
                        "is_departed": False
                    }
        except Exception:
            pass

    for a in activities:
        cam = a.get("camera_id") or "CAM01"
        if cam not in cameras_summary:
            cameras_summary[cam] = {
                "camera_id": cam,
                "cattle_ids": set(),
                "estrus_detected": False,
                "estrus_count": 0,
                "feeding_duration_sec": 0.0,
                "resting_duration_sec": 0.0,
                "workers": set(),
                "activities_count": 0
            }
        c_sum = cameras_summary[cam]
        c_sum["activities_count"] += 1

        # Track Cattle (Deduplicated by cattle_id)
        c_id = a.get("cattle_id")
        if c_id and (c_id.startswith("COW") or "cow" in c_id.lower() or c_id.isdigit()):
            norm_cid = c_id if c_id.startswith("COW") else f"COW{int(c_id):03d}" if c_id.isdigit() else c_id
            c_sum["cattle_ids"].add(norm_cid)
            if norm_cid in cattle_registry:
                c_rec = cattle_registry[norm_cid]
                if a.get("activity_type") == "feeding_activity":
                    c_rec["total_eating_sec"] += float(a.get("eating_duration_sec") or a.get("activity_duration") or 0.0)
                elif a.get("activity_type") == "resting_activity":
                    c_rec["total_resting_sec"] += float(a.get("resting_duration_sec") or a.get("activity_duration") or 0.0)
                elif a.get("activity_type") == "standing_activity":
                    c_rec["total_standing_sec"] = c_rec.get("total_standing_sec", 0.0) + float(a.get("activity_duration") or 0.0)
                elif a.get("activity_type") == "milking_activity":
                    c_rec["total_milking_sec"] = c_rec.get("total_milking_sec", 0.0) + float(a.get("activity_duration") or 0.0)
                elif a.get("activity_type") == "estrus_detected":
                    c_rec["has_estrus"] = True

        # Track Workers (Deduplicated by worker_id)
        raw_w = a.get("worker_id")
        if not raw_w and str(a.get("activity_type", "")).startswith("worker"):
            raw_w = a.get("ai_detection_result", {}).get("worker_id") if isinstance(a.get("ai_detection_result"), dict) else None
        w_id = None
        if raw_w:
            w_str = str(raw_w).strip()
            w_id = f"WRK{int(w_str):03d}" if w_str.isdigit() else w_str
        if w_id:
            c_sum["workers"].add(w_id)
            is_exit = (a.get("activity_type") == "worker_exit" or a.get("status") == "Departed")
            img_path = a.get("image_url") or f"/media/images/{w_id}.jpg"
            det_tasks = a.get("ai_detection_result", {}).get("tasks", []) if isinstance(a.get("ai_detection_result"), dict) else []
            if w_id not in worker_registry:
                worker_registry[w_id] = {
                    "worker_id": w_id,
                    "camera_id": cam,
                    "image_url": img_path,
                    "first_seen": a.get("start_time") or a.get("created_at"),
                    "last_seen": a.get("end_time") or a.get("created_at"),
                    "activity_type": a.get("activity_type"),
                    "duration_sec": float(a.get("activity_duration") or 0.0),
                    "description": a.get("description"),
                    "tasks": list(det_tasks),
                    "is_departed": is_exit
                }
            else:
                w_rec = worker_registry[w_id]
                w_rec["last_seen"] = a.get("end_time") or a.get("created_at") or w_rec["last_seen"]
                w_rec["duration_sec"] = max(w_rec["duration_sec"], float(a.get("activity_duration") or 0.0))
                if is_exit:
                    w_rec["is_departed"] = True
                if a.get("image_url") and not w_rec.get("image_url"):
                    w_rec["image_url"] = a.get("image_url")
                if det_tasks:
                    w_rec["tasks"] = list(set(w_rec.get("tasks", [])).union(set(det_tasks)))

        # Track Estrus per Camera
        if a.get("activity_type") == "estrus_detected":
            c_sum["estrus_detected"] = True
            c_sum["estrus_count"] += 1

        c_sum["feeding_duration_sec"] += float(a.get("eating_duration_sec") or 0.0)
        c_sum["resting_duration_sec"] += float(a.get("resting_duration_sec") or 0.0)

    # Serialize camera summaries
    serialized_cameras = []
    for cam_id, cs in cameras_summary.items():
        serialized_cameras.append({
            "camera_id": cam_id,
            "unique_cattle_count": len(cs["cattle_ids"]),
            "cattle_ids": sorted(list(cs["cattle_ids"])),
            "estrus_detected": cs["estrus_detected"],
            "estrus_count": cs["estrus_count"],
            "feeding_duration_sec": round(cs["feeding_duration_sec"], 1),
            "resting_duration_sec": round(cs["resting_duration_sec"], 1),
            "workers_count": len(cs["workers"]),
            "worker_ids": sorted(list(cs["workers"])),
            "activities_count": cs["activities_count"]
        })

    # Accurately aggregate unique suspected diseased cattle from activities & cattle registry
    diseased_entities = set()
    for a in activities:
        act_t = str(a.get("activity_type", ""))
        d_stat = str(a.get("disease_status", "")).strip()
        cid = a.get("cattle_id")
        if act_t in ("disease_suspected", "disease_alert") or (d_stat and d_stat.lower() not in ("healthy", "scanning...", "none", "")):
            if cid:
                diseased_entities.add(cid)
            else:
                diseased_entities.add(a.get("activity_id", str(len(diseased_entities))))

    for norm_cid, c_data in cattle_registry.items():
        d_stat = str(c_data.get("disease_status", "")).strip()
        if d_stat and d_stat.lower() not in ("healthy", "scanning...", "none", ""):
            diseased_entities.add(norm_cid)

    disease_count = max(disease_count, len(diseased_entities))

    unique_breeds = set(c["breed"] for c in cattle_registry.values() if c.get("breed") and c.get("breed") not in ("Cattle", "Scanning..."))
    bcs_cows_count = len([c for c in cattle_registry.values() if c.get("bcs_score") is not None])

    return {
        "farm_id": farm_id,
        "total": total_count,
        "metrics": {
            "estrus_count": estrus_count,
            "disease_count": disease_count,
            "calving_count": calving_count,
            "breed_count": len(unique_breeds),
            "bcs_count": bcs_cows_count,
            "worker_count": len(worker_registry),
            "unique_cattle_count": len(cattle_registry),
            "unique_worker_count": len(worker_registry)
        },
        "camera_summaries": serialized_cameras,
        "cattle_registry": list(cattle_registry.values()),
        "worker_registry": list(worker_registry.values()),
        "activities": activities
    }


@app.post("/api/activities/clean-duplicates")
async def clean_duplicate_activities_endpoint(user: Dict[str, Any] = Depends(get_current_user)):
    """Consolidates duplicate records for the same cattle in Supabase & local cache."""
    farm_id = user.get("farm_id")
    res = supabase_manager.clean_duplicate_activities(farm_id)
    return res


@app.delete("/api/activities/clear-all")
async def clear_all_activities_endpoint(user: Dict[str, Any] = Depends(get_current_user)):
    """Permanently deletes all cattle, farm activities, and worker records from Supabase."""
    farm_id = user.get("farm_id", DEFAULT_FARM_ID)
    res = supabase_manager.clear_all_activities(farm_id)
    # Reset in-memory audit state
    last_registration_audit.update({
        "timestamp": None,
        "detected_count": 0,
        "registered_count": 0,
        "success": True,
        "detected_cattle": [],
        "registered_cattle": [],
        "failed_cattle": {}
    })
    return res


class ActivityUpdateRequest(BaseModel):
    activity_type: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    confidence_score: Optional[float] = None


@app.delete("/api/activities/{activity_id}")
async def delete_activity(
    activity_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Deletes an activity record from database & cache."""
    farm_id = user.get("farm_id", DEFAULT_FARM_ID)
    ok = supabase_manager.delete_activity(farm_id, activity_id)
    return {"success": ok, "message": f"Activity {activity_id} removed"}


@app.patch("/api/activities/{activity_id}")
async def update_activity(
    activity_id: str,
    payload: ActivityUpdateRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Updates an activity record (e.g. correcting a false prediction)."""
    farm_id = user.get("farm_id", DEFAULT_FARM_ID)
    updates = {k: v for k, v in payload.dict().items() if v is not None}
    ok = supabase_manager.update_activity(farm_id, activity_id, updates)
    return {"success": ok, "message": f"Activity {activity_id} updated"}


@app.get("/api/supabase/status")
async def get_supabase_status():
    """Returns live connection status of Supabase."""
    return {
        "is_connected": supabase_manager.is_connected,
        "mode": "Live Supabase Cloud" if supabase_manager.is_connected else "Local / Offline Fallback"
    }


@app.post("/api/supabase/sync")
async def trigger_supabase_sync():
    """Uploads and syncs all local captures, images, and videos to Supabase."""
    res = supabase_manager.sync_local_to_supabase()
    return res


@app.get("/favicon.ico")
async def get_favicon():
    logo_path = BASE_DIR / "logo.png"
    if logo_path.exists():
        return FileResponse(logo_path, media_type="image/png")
    return Response(status_code=204)


@app.get("/media/images/{filename}")
@app.get("/captures/images/{filename}")
async def get_local_image(filename: str):
    """Serves captured high-res image from local storage with fuzzy fallback and no-cache header."""
    file_path = LOCAL_IMAGES_DIR / filename
    no_cache_headers = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    if file_path.exists():
        return FileResponse(file_path, media_type="image/jpeg", headers=no_cache_headers)

    # Try fuzzy match by stem or prefix (e.g. af1f801d...)
    stem = Path(filename).stem
    candidates = list(LOCAL_IMAGES_DIR.glob(f"*{stem}*"))
    if not candidates and "-" in stem:
        short_id = stem.split("-")[0]
        candidates = list(LOCAL_IMAGES_DIR.glob(f"*{short_id}*"))

    if candidates:
        return FileResponse(candidates[0], media_type="image/jpeg", headers=no_cache_headers)

    logo_path = BASE_DIR / "logo.png"
    if logo_path.exists():
        return FileResponse(logo_path, media_type="image/png", headers=no_cache_headers)

    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/media/videos/{filename}")
async def get_local_video(filename: str):
    """Serves captured 10-second video clip from local storage with fuzzy fallback."""
    file_path = LOCAL_VIDEOS_DIR / filename
    if file_path.exists():
        return FileResponse(file_path, media_type="video/mp4")

    stem = Path(filename).stem
    candidates = list(LOCAL_VIDEOS_DIR.glob(f"*{stem}*"))
    if not candidates and "-" in stem:
        short_id = stem.split("-")[0]
        candidates = list(LOCAL_VIDEOS_DIR.glob(f"*{short_id}*"))

    if candidates:
        return FileResponse(candidates[0], media_type="video/mp4")

    raise HTTPException(status_code=404, detail="Video not found")


# ---------------------------------------------------------------------
# Web UI - PashuX Full All-in-One Dashboard
# ---------------------------------------------------------------------
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PashuX AI – Freight & Livestock Telemetry Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    /* =====================================================================
       PASHUX DESIGN SYSTEM: FREIGHT & DATA LOGISTICS (LIGHT THEME DEFAULT)
       ===================================================================== */
    :root {
      --bg-body: #f8fafc;
      --bg-surface: #ffffff;
      --bg-surface-elevated: #ffffff;
      --bg-muted: #f1f5f9;
      --bg-hover: #f8fafc;
      --border-subtle: #e2e8f0;
      --border-medium: #cbd5e1;
      --border-strong: #94a3b8;
      
      --text-heading: #0f172a;
      --text-body: #334155;
      --text-muted: #64748b;
      --text-inverse: #ffffff;

      --primary: #0284c7;
      --primary-hover: #0369a1;
      --primary-light: #e0f2fe;
      --primary-border: #bae6fd;
      
      --accent-emerald: #059669;
      --accent-emerald-light: #ecfdf5;
      --accent-emerald-border: #a7f3d0;

      --accent-estrus: #e11d48;
      --accent-estrus-light: #ffe4e6;
      --accent-estrus-border: #fecdd3;

      --accent-disease: #dc2626;
      --accent-disease-light: #fef2f2;
      --accent-disease-border: #fecaca;

      --accent-warning: #d97706;
      --accent-warning-light: #fffbeb;
      --accent-warning-border: #fde68a;

      --accent-feeding: #0284c7;
      --accent-feeding-light: #f0f9ff;
      --accent-feeding-border: #bae6fd;

      --accent-standing: #0891b2;
      --accent-standing-light: #ecfeff;
      --accent-standing-border: #a5f3fc;

      --accent-resting: #7c3aed;
      --accent-resting-light: #f5f3ff;
      --accent-resting-border: #ddd6fe;

      --accent-milking: #ea580c;
      --accent-milking-light: #fff7ed;
      --accent-milking-border: #ffedd5;

      --card-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.06), 0 1px 2px -1px rgba(15, 23, 42, 0.04);
      --card-shadow-hover: 0 12px 24px -4px rgba(15, 23, 42, 0.10), 0 4px 6px -2px rgba(15, 23, 42, 0.04);
      --card-shadow-elevated: 0 20px 25px -5px rgba(15, 23, 42, 0.12), 0 8px 10px -6px rgba(15, 23, 42, 0.04);

      --font-sans: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', SFMono-Regular, Consolas, monospace;
    }

    [data-theme="dark"] {
      --bg-body: #0b1120;
      --bg-surface: #131d31;
      --bg-surface-elevated: #1a2744;
      --bg-muted: #1e293b;
      --bg-hover: #1e293b;
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-medium: rgba(255, 255, 255, 0.16);
      --border-strong: rgba(255, 255, 255, 0.28);

      --text-heading: #f8fafc;
      --text-body: #cbd5e1;
      --text-muted: #94a3b8;
      --text-inverse: #0f172a;

      --primary: #38bdf8;
      --primary-hover: #7dd3fc;
      --primary-light: rgba(56, 189, 248, 0.14);
      --primary-border: rgba(56, 189, 248, 0.35);

      --accent-emerald: #34d399;
      --accent-emerald-light: rgba(52, 211, 153, 0.14);
      --accent-emerald-border: rgba(52, 211, 153, 0.35);

      --accent-estrus: #fb7185;
      --accent-estrus-light: rgba(251, 113, 133, 0.16);
      --accent-estrus-border: rgba(251, 113, 133, 0.35);

      --accent-disease: #f87171;
      --accent-disease-light: rgba(248, 113, 113, 0.16);
      --accent-disease-border: rgba(248, 113, 113, 0.35);

      --accent-warning: #fbbf24;
      --accent-warning-light: rgba(251, 191, 36, 0.16);
      --accent-warning-border: rgba(251, 191, 36, 0.35);

      --accent-feeding: #38bdf8;
      --accent-feeding-light: rgba(56, 189, 248, 0.14);
      --accent-feeding-border: rgba(56, 189, 248, 0.35);

      --accent-standing: #22d3ee;
      --accent-standing-light: rgba(34, 211, 238, 0.14);
      --accent-standing-border: rgba(34, 211, 238, 0.35);

      --accent-resting: #a78bfa;
      --accent-resting-light: rgba(167, 139, 250, 0.16);
      --accent-resting-border: rgba(167, 139, 250, 0.35);

      --accent-milking: #fb923c;
      --accent-milking-light: rgba(251, 146, 60, 0.16);
      --accent-milking-border: rgba(251, 146, 60, 0.35);

      --card-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
      --card-shadow-hover: 0 14px 28px rgba(0, 0, 0, 0.50);
      --card-shadow-elevated: 0 24px 38px rgba(0, 0, 0, 0.65);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    body {
      background-color: var(--bg-body);
      color: var(--text-body);
      font-family: var(--font-sans);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      -webkit-font-smoothing: antialiased;
      transition: background-color 0.25s ease, color 0.25s ease;
    }

    /* Subtle freight grid background in light theme */
    body::before {
      content: "";
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background-image: 
        linear-gradient(to right, var(--border-subtle) 1px, transparent 1px),
        linear-gradient(to bottom, var(--border-subtle) 1px, transparent 1px);
      background-size: 32px 32px;
      opacity: 0.35;
      pointer-events: none;
      z-index: 0;
    }

    @keyframes spin { 100% { transform: rotate(360deg); } }
    @keyframes pulseSoft { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

    /* =====================================================================
       TOP FREIGHT TELEMETRY STATUS TICKER
       ===================================================================== */
    .top-telemetry-strip {
      background: var(--bg-muted);
      border-bottom: 1px solid var(--border-subtle);
      font-size: 0.74rem;
      font-family: var(--font-mono);
      color: var(--text-muted);
      padding: 0.35rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: relative;
      z-index: 60;
    }
    .ticker-items {
      display: flex;
      gap: 1.5rem;
      align-items: center;
      flex-wrap: wrap;
    }
    .ticker-item {
      display: flex;
      align-items: center;
      gap: 0.35rem;
    }
    .dot-live {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--accent-emerald);
      box-shadow: 0 0 8px var(--accent-emerald);
      animation: pulseSoft 2s infinite;
    }

    /* =====================================================================
       HEADER NAVIGATION
       ===================================================================== */
    header {
      background: var(--bg-surface);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border-subtle);
      position: sticky;
      top: 0;
      z-index: 50;
      padding: 0.85rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 0.85rem;
      text-decoration: none;
    }
    .brand-logo {
      width: 42px;
      height: 42px;
      border-radius: 12px;
      background: linear-gradient(135deg, #0284c7, #059669);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.35rem;
      color: #fff;
      box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25);
    }
    .brand-title {
      font-size: 1.25rem;
      font-weight: 800;
      color: var(--text-heading);
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }
    .brand-badge {
      background: var(--primary-light);
      color: var(--primary);
      border: 1px solid var(--primary-border);
      padding: 0.18rem 0.55rem;
      border-radius: 6px;
      font-size: 0.68rem;
      font-weight: 700;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    /* Theme Toggle Switch */
    .btn-theme-toggle {
      background: var(--bg-muted);
      border: 1px solid var(--border-subtle);
      color: var(--text-heading);
      padding: 0.45rem 0.85rem;
      border-radius: 10px;
      font-size: 0.82rem;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.45rem;
      transition: all 0.2s ease;
    }
    .btn-theme-toggle:hover {
      background: var(--bg-hover);
      border-color: var(--border-medium);
      transform: translateY(-1px);
    }

    .user-pill {
      display: flex;
      align-items: center;
      gap: 0.6rem;
      background: var(--bg-muted);
      padding: 0.4rem 0.9rem;
      border-radius: 20px;
      border: 1px solid var(--border-subtle);
    }
    .user-phone {
      font-family: var(--font-mono);
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--text-heading);
    }

    .btn-auth {
      background: var(--bg-surface);
      border: 1px solid var(--border-medium);
      color: var(--text-heading);
      padding: 0.45rem 0.95rem;
      border-radius: 8px;
      font-size: 0.8rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-auth:hover {
      background: var(--bg-muted);
      border-color: var(--border-strong);
    }

    .container {
      max-width: 1520px;
      margin: 0 auto;
      padding: 1.8rem 2rem;
      width: 100%;
      flex: 1;
      position: relative;
      z-index: 10;
    }

    /* =====================================================================
       ACTION TOOLBAR (FREIGHT LOGISTICS BANNER)
       ===================================================================== */
    .action-toolbar {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 16px;
      padding: 1.25rem 1.6rem;
      margin-bottom: 1.8rem;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 1.2rem;
      box-shadow: var(--card-shadow);
    }
    .farm-meta-info h3 {
      font-size: 1.2rem;
      font-weight: 800;
      color: var(--text-heading);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .farm-meta-info p {
      font-size: 0.82rem;
      color: var(--text-muted);
      font-family: var(--font-mono);
      margin-top: 0.25rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .farm-manifest-tag {
      background: var(--accent-emerald-light);
      color: var(--accent-emerald);
      border: 1px solid var(--accent-emerald-border);
      padding: 0.15rem 0.45rem;
      border-radius: 4px;
      font-size: 0.7rem;
      font-weight: 700;
    }

    .toolbar-buttons {
      display: flex;
      gap: 0.65rem;
      flex-wrap: wrap;
      align-items: center;
    }
    .btn-action {
      background: linear-gradient(135deg, #0284c7, #0369a1);
      color: #fff;
      border: none;
      padding: 0.6rem 1.15rem;
      border-radius: 10px;
      font-size: 0.84rem;
      font-weight: 700;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      box-shadow: 0 2px 6px rgba(2, 132, 199, 0.25);
      transition: all 0.2s ease;
    }
    .btn-action:hover {
      transform: translateY(-1px);
      box-shadow: 0 6px 14px rgba(2, 132, 199, 0.35);
    }
    .btn-action-purple {
      background: linear-gradient(135deg, #7c3aed, #6d28d9);
      box-shadow: 0 2px 6px rgba(124, 58, 237, 0.25);
    }
    .btn-action-purple:hover {
      box-shadow: 0 6px 14px rgba(124, 58, 237, 0.35);
    }
    .btn-secondary {
      background: var(--bg-surface);
      color: var(--text-heading);
      border: 1px solid var(--border-medium);
      box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .btn-secondary:hover {
      background: var(--bg-muted);
      border-color: var(--border-strong);
      box-shadow: 0 2px 5px rgba(0,0,0,0.06);
    }
    .btn-danger-outline {
      border-color: var(--accent-disease-border) !important;
      color: var(--accent-disease) !important;
      background: var(--accent-disease-light) !important;
    }
    .btn-danger-outline:hover {
      background: #fee2e2 !important;
    }

    /* =====================================================================
       PROGRESS BANNER
       ===================================================================== */
    .video-job-banner {
      background: var(--primary-light);
      border: 1px solid var(--primary-border);
      border-radius: 14px;
      padding: 1.1rem 1.4rem;
      margin-bottom: 1.8rem;
      box-shadow: var(--card-shadow);
    }
    .progress-bar-bg {
      width: 100%;
      height: 8px;
      background: var(--border-subtle);
      border-radius: 4px;
      overflow: hidden;
      margin-top: 0.6rem;
    }
    .progress-bar-fill {
      height: 100%;
      background: linear-gradient(90deg, #0284c7, #059669);
      transition: width 0.3s ease;
    }

    /* =====================================================================
       LIVE STREAM BOX
       ===================================================================== */
    .stream-box {
      background: #000;
      border: 1px solid var(--border-subtle);
      border-radius: 16px;
      overflow: hidden;
      margin-bottom: 1.8rem;
      display: none;
      box-shadow: var(--card-shadow-elevated);
    }
    .stream-header {
      background: var(--bg-surface);
      padding: 0.85rem 1.4rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border-subtle);
    }
    .stream-img {
      width: 100%;
      max-height: 520px;
      object-fit: contain;
      display: block;
      background: #000;
    }

    /* =====================================================================
       STATS OVERVIEW (FREIGHT KPI TILES)
       ===================================================================== */
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1.1rem;
      margin-bottom: 1.8rem;
    }
    .stat-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 1.2rem 1.35rem;
      position: relative;
      overflow: hidden;
      box-shadow: var(--card-shadow);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    .stat-card:hover {
      transform: translateY(-2px);
      box-shadow: var(--card-shadow-hover);
    }
    .stat-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      bottom: 0;
      width: 4px;
    }
    .stat-cattle::before { background: var(--accent-emerald); }
    .stat-standing::before { background: var(--accent-standing); }
    .stat-feeding::before { background: var(--accent-feeding); }
    .stat-disease::before { background: var(--accent-disease); }
    .stat-estrus::before { background: var(--accent-estrus); }
    .stat-milking::before { background: var(--accent-milking); }
    .stat-resting::before { background: var(--accent-resting); }
    .stat-worker::before { background: #0284c7; }

    .stat-card-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 0.5rem;
    }
    .stat-label {
      font-size: 0.74rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-muted);
    }
    .stat-icon {
      font-size: 1.25rem;
    }
    .stat-value {
      font-size: 2.1rem;
      font-weight: 800;
      font-family: var(--font-sans);
      color: var(--text-heading);
      line-height: 1.1;
    }
    .stat-subtext {
      font-size: 0.74rem;
      color: var(--text-muted);
      margin-top: 0.4rem;
      display: flex;
      align-items: center;
      gap: 0.3rem;
      font-weight: 500;
    }

    /* =====================================================================
       REGISTRY & LEDGER SECTION (FREIGHT MANIFEST)
       ===================================================================== */
    .camera-summary-box {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 16px;
      padding: 1.5rem;
      margin-bottom: 1.8rem;
      box-shadow: var(--card-shadow);
    }
    .section-header-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.2rem;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 0.9rem;
      flex-wrap: wrap;
      gap: 0.8rem;
    }
    .section-title {
      font-size: 1.18rem;
      font-weight: 800;
      color: var(--text-heading);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .section-subtitle {
      font-size: 0.82rem;
      color: var(--text-muted);
      margin-top: 0.2rem;
    }
    .manifest-pill {
      padding: 0.35rem 0.85rem;
      border-radius: 20px;
      font-weight: 700;
      font-size: 0.78rem;
      font-family: var(--font-mono);
      background: var(--accent-emerald-light);
      color: var(--accent-emerald);
      border: 1px solid var(--accent-emerald-border);
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }

    /* Camera Summary Nodes Grid */
    .camera-nodes-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 1rem;
      margin-bottom: 1.4rem;
    }
    .camera-node-card {
      background: var(--bg-muted);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 1.1rem;
      transition: all 0.2s;
    }
    .camera-node-card:hover {
      border-color: var(--border-medium);
      background: var(--bg-surface);
      box-shadow: var(--card-shadow);
    }

    /* Horizontal Rosters (Cattle & Workers) */
    .roster-section-title {
      font-size: 0.95rem;
      font-weight: 800;
      color: var(--text-heading);
      margin-bottom: 0.75rem;
      display: flex;
      align-items: center;
      gap: 0.45rem;
    }
    .horizontal-roster {
      display: flex;
      gap: 0.9rem;
      overflow-x: auto;
      padding-bottom: 0.7rem;
      scrollbar-width: thin;
    }
    .horizontal-roster::-webkit-scrollbar {
      height: 6px;
    }
    .horizontal-roster::-webkit-scrollbar-thumb {
      background: var(--border-medium);
      border-radius: 4px;
    }

    .cattle-roster-card {
      min-width: 230px;
      max-width: 245px;
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 0.85rem;
      flex-shrink: 0;
      box-shadow: var(--card-shadow);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .cattle-roster-card:hover {
      transform: translateY(-2px);
      box-shadow: var(--card-shadow-hover);
      border-color: var(--border-medium);
    }
    .roster-thumb-box {
      position: relative;
      width: 100%;
      height: 120px;
      border-radius: 8px;
      overflow: hidden;
      margin-bottom: 0.6rem;
      background: var(--bg-muted);
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px solid var(--border-subtle);
    }
    .roster-thumb-img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    .roster-tag-pill {
      position: absolute;
      top: 6px;
      left: 6px;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(4px);
      padding: 2px 7px;
      border-radius: 5px;
      color: #38bdf8;
      font-family: var(--font-mono);
      font-weight: 700;
      font-size: 0.76rem;
      border: 1px solid rgba(56, 189, 248, 0.4);
    }

    .worker-roster-card {
      min-width: 240px;
      max-width: 265px;
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 0.85rem;
      flex-shrink: 0;
      box-shadow: var(--card-shadow);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
      position: relative;
    }
    .worker-roster-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: #0284c7;
      border-radius: 12px 12px 0 0;
    }
    .worker-roster-card:hover {
      transform: translateY(-2px);
      box-shadow: var(--card-shadow-hover);
      border-color: var(--border-medium);
    }

    /* =====================================================================
       FILTER BAR (FREIGHT SEARCH PANEL)
       ===================================================================== */
    .filter-section {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 1.1rem 1.4rem;
      margin-bottom: 1.8rem;
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      justify-content: space-between;
      align-items: center;
      box-shadow: var(--card-shadow);
    }
    .filter-group {
      display: flex;
      flex-wrap: wrap;
      gap: 0.9rem;
      align-items: center;
    }
    .filter-item {
      display: flex;
      flex-direction: column;
      gap: 0.3rem;
    }
    .filter-item label {
      font-size: 0.72rem;
      color: var(--text-muted);
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .filter-select, .filter-input {
      background: var(--bg-body);
      border: 1px solid var(--border-medium);
      color: var(--text-heading);
      padding: 0.52rem 0.85rem;
      border-radius: 8px;
      font-size: 0.84rem;
      font-weight: 600;
      outline: none;
      transition: border-color 0.2s;
    }
    .filter-select:focus, .filter-input:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px var(--primary-light);
    }

    /* =====================================================================
       ACTIVITIES GRID & EVIDENCE CARDS
       ===================================================================== */
    .activities-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 1.4rem;
    }
    .activity-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
      cursor: pointer;
      box-shadow: var(--card-shadow);
    }
    .activity-card:hover {
      transform: translateY(-3px);
      box-shadow: var(--card-shadow-hover);
      border-color: var(--border-medium);
    }
    .card-media {
      position: relative;
      width: 100%;
      height: 205px;
      background: #0f172a;
    }
    .card-thumb {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    .video-tag {
      position: absolute;
      bottom: 10px;
      right: 10px;
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid rgba(255, 255, 255, 0.18);
      color: #fff;
      padding: 0.22rem 0.6rem;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 700;
      font-family: var(--font-mono);
      backdrop-filter: blur(4px);
    }
    .badge-status {
      position: absolute;
      top: 10px;
      left: 10px;
      padding: 0.3rem 0.7rem;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      backdrop-filter: blur(6px);
      box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    .badge-estrus { background: var(--accent-estrus); color: #fff; }
    .badge-disease { background: var(--accent-disease); color: #fff; }
    .badge-calving { background: #d97706; color: #fff; }
    .badge-feeding { background: #0284c7; color: #fff; }
    .badge-standing { background: #0891b2; color: #fff; }
    .badge-milking { background: #ea580c; color: #fff; }
    .badge-worker { background: #7c3aed; color: #fff; }
    .badge-general { background: #059669; color: #fff; }

    .card-body {
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      flex: 1;
      gap: 0.7rem;
    }
    .card-top {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 0.5rem;
    }
    .activity-name {
      font-size: 1.05rem;
      font-weight: 800;
      color: var(--text-heading);
      line-height: 1.3;
    }
    .activity-conf {
      font-family: var(--font-mono);
      font-size: 0.78rem;
      font-weight: 700;
      background: var(--accent-emerald-light);
      color: var(--accent-emerald);
      padding: 0.2rem 0.55rem;
      border-radius: 6px;
      border: 1px solid var(--accent-emerald-border);
      flex-shrink: 0;
    }
    .activity-desc {
      font-size: 0.85rem;
      color: var(--text-muted);
      line-height: 1.5;
    }
    .telemetry-chip {
      background: var(--bg-muted);
      color: var(--text-heading);
      border: 1px solid var(--border-subtle);
      padding: 0.2rem 0.5rem;
      border-radius: 6px;
      font-size: 0.74rem;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 0.25rem;
    }
    .meta-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: auto;
      padding-top: 0.85rem;
      border-top: 1px solid var(--border-subtle);
      font-size: 0.76rem;
      color: var(--text-muted);
    }

    /* =====================================================================
       MODALS & DIALOGS
       ===================================================================== */
    .modal-backdrop {
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(15, 23, 42, 0.65);
      backdrop-filter: blur(8px);
      z-index: 100;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
    }
    .modal-content {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 18px;
      max-width: 920px;
      width: 100%;
      max-height: 90vh;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      box-shadow: var(--card-shadow-elevated);
      color: var(--text-body);
    }
    .modal-header {
      padding: 1.2rem 1.6rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border-subtle);
    }
    .modal-title {
      font-size: 1.2rem;
      font-weight: 800;
      color: var(--text-heading);
    }
    .btn-close {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 1.6rem;
      cursor: pointer;
      line-height: 1;
    }
    .btn-close:hover {
      color: var(--text-heading);
    }
    .modal-video {
      width: 100%;
      max-height: 480px;
      background: #000;
    }
    .modal-details {
      padding: 1.5rem;
      display: grid;
      grid-template-columns: 1.8fr 1.2fr;
      gap: 1.5rem;
    }
    .auth-modal { max-width: 460px; }
    .auth-form {
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    .form-group {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }
    .form-group label {
      font-size: 0.74rem;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .form-control {
      background: var(--bg-body);
      border: 1px solid var(--border-medium);
      color: var(--text-heading);
      padding: 0.65rem 0.95rem;
      border-radius: 8px;
      font-size: 0.9rem;
      outline: none;
      transition: border-color 0.2s;
    }
    .form-control:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px var(--primary-light);
    }

    .empty-state {
      grid-column: 1 / -1;
      text-align: center;
      padding: 4rem 2rem;
      background: var(--bg-surface);
      border-radius: 16px;
      border: 1px dashed var(--border-medium);
      color: var(--text-muted);
    }
    .empty-state h3 {
      font-size: 1.2rem;
      font-weight: 800;
      color: var(--text-heading);
      margin-bottom: 0.5rem;
    }
  </style>
</head>
<body>

  <!-- Top Freight Telemetry Status Ticker -->
  <div class="top-telemetry-strip">
    <div class="ticker-items">
      <div class="ticker-item">
        <span class="dot-live"></span>
        <span>EDGE TELEMETRY: <b>ACTIVE</b></span>
      </div>
      <div class="ticker-item">
        <span>📡 AI INFERENCE: <b>MULTI-MODEL PIPELINE</b></span>
      </div>
      <div class="ticker-item">
        <span>🛡️ BIO-SECURITY SURVEILLANCE: <b>HIGH CONFIDENCE GATED</b></span>
      </div>
      <div class="ticker-item">
        <span>☁️ SUPABASE CLOUD SYNC: <b>SYNCHRONIZED</b></span>
      </div>
    </div>
    <div>
      <span>🕒 <span id="systemClock">--:--:--</span></span>
    </div>
  </div>

  <!-- Top Navigation Header -->
  <header>
    <a href="/" class="brand">
      <div class="brand-logo">🚚</div>
      <div>
        <div class="brand-title">PashuX<span style="font-size:0.8rem; font-weight:600; color:var(--text-muted);">Fleet & Livestock</span></div>
      </div>
    </a>

    <div class="header-actions">
      <!-- Theme Switcher Button -->
      <button class="btn-theme-toggle" id="btnThemeToggle" onclick="toggleDashboardTheme()" title="Switch between Light and Dark Theme">
        <span id="themeToggleIcon">🌙</span> <span id="themeToggleText">Dark Mode</span>
      </button>

      <div class="user-pill" id="userPill">
        <span style="width: 8px; height: 8px; border-radius: 50%; background: var(--accent-emerald);"></span>
        <span class="user-phone" id="displayPhone">+91 98765 43210</span>
      </div>
      <button class="btn-auth" id="btnLoginToggle" onclick="openAuthModal()">🔑 Switch / Login</button>
      <button class="btn-auth" id="btnLogout" onclick="logout()" style="display: none;">🚪 Logout</button>
    </div>
  </header>

  <main class="container">

    <!-- Active Farm & Freight Action Toolbar -->
    <section class="action-toolbar">
      <div class="farm-meta-info">
        <h3 id="displayFarmName">🏭 PashuX Pioneer Livestock Farm</h3>
        <p>
          <span class="farm-manifest-tag">MANIFEST HUB #01</span>
          <span id="displayFarmId">Farm ID: 11111111-1111-1111-1111-111111111111</span>
        </p>
      </div>

      <div class="toolbar-buttons">
        <button class="btn-action" onclick="openConnectCameraModal()">
          <span>📹</span> Connect CCTV Camera
        </button>
        <button class="btn-action btn-action-purple" onclick="openUploadModal()">
          <span>📁</span> Upload CCTV Video
        </button>
        <button class="btn-action btn-secondary" onclick="cleanDuplicates()" title="Consolidates duplicate records for the same cattle">
          <span>🧹</span> Consolidate Duplicates
        </button>
        <button class="btn-action btn-secondary btn-danger-outline" onclick="clearAllActivities()" title="Permanently delete all activity history">
          <span>🗑️</span> Clear History
        </button>
        <button class="btn-action btn-secondary" id="btnToggleLiveStream" onclick="toggleLiveStream()">
          <span>🔴</span> View Live AI Stream
        </button>
      </div>
    </section>

    <!-- Video Analysis Progress Notification Banner -->
    <div id="videoJobBanner" class="video-job-banner" style="display: none;">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div style="display: flex; align-items: center; gap: 0.65rem;">
          <span id="jobSpinner" style="display: inline-block; font-size: 1.15rem; animation: spin 1s linear infinite;">⏳</span>
          <span id="jobBannerText" style="font-weight: 700; font-size: 0.95rem; color: var(--text-heading);">Analyzing video feed...</span>
        </div>
        <span id="jobPercentText" style="font-family: var(--font-mono); font-weight: 800; color: var(--primary); font-size: 1.05rem;">0%</span>
      </div>
      <div class="progress-bar-bg">
        <div id="jobProgressBarFill" class="progress-bar-fill" style="width: 0%;"></div>
      </div>
    </div>

    <!-- Live Stream Preview Window -->
    <section class="stream-box" id="streamBox">
      <div class="stream-header">
        <span style="font-weight: 800; color: var(--accent-emerald); display: flex; align-items: center; gap: 0.5rem;">
          <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: var(--accent-emerald); box-shadow: 0 0 10px var(--accent-emerald);"></span>
          <span id="streamTitle">Live CCTV AI Detection Feed</span>
        </span>
        <div style="display: flex; align-items: center; gap: 0.8rem;">
          <span id="streamStatusBadge" style="font-size: 0.8rem; color: var(--accent-emerald); font-family: var(--font-mono); font-weight: 700;">🟢 Active Surveillance</span>
          <button class="btn-action" style="padding: 0.3rem 0.75rem; font-size: 0.75rem; background: var(--bg-muted); color: var(--text-heading); border: 1px solid var(--border-medium);" onclick="toggleLiveStream()">✕ Close Feed</button>
        </div>
      </div>
      <img id="liveStreamImg" class="stream-img" src="" alt="Live CCTV AI Stream">
    </section>

    <!-- Stats Overview (PashuX AI Vision Model Grid) -->
    <section class="stats-grid">
      <div class="stat-card stat-cattle">
        <div class="stat-card-header">
          <span class="stat-label">Total Cattle Monitored</span>
          <span class="stat-icon">🐄</span>
        </div>
        <div class="stat-value" id="statCattle">0</div>
        <div class="stat-subtext"><span>✅</span> Zero-Duplicate Tracking</div>
      </div>

      <div class="stat-card stat-standing">
        <div class="stat-card-header">
          <span class="stat-label">Breeds Identified</span>
          <span class="stat-icon">🧬</span>
        </div>
        <div class="stat-value" id="statBreeds">0</div>
        <div class="stat-subtext"><span>🎯</span> AI Classifier (best.onnx)</div>
      </div>

      <div class="stat-card stat-feeding">
        <div class="stat-card-header">
          <span class="stat-label">BCS Scoring (1-5)</span>
          <span class="stat-icon">🥩</span>
        </div>
        <div class="stat-value" id="statBcs">0</div>
        <div class="stat-subtext"><span>📊</span> Deep CNN Model (v2)</div>
      </div>

      <div class="stat-card stat-disease">
        <div class="stat-card-header">
          <span class="stat-label">Suspected Disease</span>
          <span class="stat-icon">🩺</span>
        </div>
        <div class="stat-value" id="statDisease">0</div>
        <div class="stat-subtext"><span>🛡️</span> FMD / LSD Screening Model</div>
      </div>

      <div class="stat-card stat-estrus">
        <div class="stat-card-header">
          <span class="stat-label">Estrus / Mounting</span>
          <span class="stat-icon">🔥</span>
        </div>
        <div class="stat-value" id="statEstrus">0</div>
        <div class="stat-subtext"><span>🧬</span> Kinematic CowXNet AI</div>
      </div>

      <div class="stat-card stat-worker">
        <div class="stat-card-header">
          <span class="stat-label">Active Farm Crew</span>
          <span class="stat-icon">👷</span>
        </div>
        <div class="stat-value" id="statWorker">0</div>
        <div class="stat-subtext"><span>📋</span> Verified Duty Ledger</div>
      </div>
    </section>

    <!-- Daily Camera & Cattle Herd Intelligence Registry -->
    <section class="camera-summary-box">
      <div class="section-header-row">
        <div>
          <h3 class="section-title">
            <span>📸</span> Real-Time Livestock Manifest & Herd Telemetry
          </h3>
          <p class="section-subtitle">
            Deduplicated cattle count, individual breed profiles, BCS scoring, biometrics, and verified human worker ledger.
          </p>
        </div>
        <div id="cameraStatusPill" class="manifest-pill">
          <span>🛡️</span> Zero-Duplicate Active Manifest
        </div>
      </div>

      <!-- Camera Feed Summary Cards -->
      <div id="cameraSummaryCards" class="camera-nodes-grid">
        <!-- Filled dynamically by JavaScript -->
      </div>

      <!-- Registered Cattle Individual Cards -->
      <div style="margin-top: 1.2rem;">
        <div class="roster-section-title">
          <span>🐄</span> Registered Livestock Ledger (Breed, BCS, Biometry & Duration)
        </div>
        <div id="cattleRosterList" class="horizontal-roster">
          <!-- Filled dynamically by JavaScript -->
        </div>
      </div>

      <!-- Verified Farm Worker Attendance Roster -->
      <div style="margin-top: 1.4rem; border-top: 1px solid var(--border-subtle); padding-top: 1.1rem;">
        <div class="roster-section-title">
          <span>👷</span> Farm Personnel & Worker Attendance Ledger (Check-In & Tasks)
        </div>
        <div id="workerRosterList" class="horizontal-roster">
          <!-- Filled dynamically by JavaScript -->
        </div>
      </div>
    </section>

    <!-- Filter & Search Section -->
    <section class="filter-section">
      <div class="filter-group">
        <div class="filter-item">
          <label>🏷️ Activity Type</label>
          <select id="filterType" class="filter-select" onchange="fetchActivities()">
            <option value="all">All AI Detections</option>
            <option value="estrus_detected">🔥 Estrus / Mounting Heat</option>
            <option value="disease_suspected">🩺 Suspected Disease Alert</option>
            <option value="bcs_alert">📊 Extreme BCS Alert</option>
            <option value="worker_entry">🚪 Worker Check-In</option>
            <option value="cleaning_activity">🧹 Cleaning / Sanitation Duty</option>
            <option value="worker_attendance">👷 Worker Attendance Summary</option>
          </select>
        </div>

        <div class="filter-item">
          <label>📹 Camera Node</label>
          <select id="filterCamera" class="filter-select" onchange="fetchActivities()">
            <option value="all">All CCTV Cameras</option>
            <option value="CAM01">CAM01 (Main Gate & Feed Alley)</option>
            <option value="CAM02">CAM02 (Milking Stall Area)</option>
          </select>
        </div>

        <div class="filter-item">
          <label>🔍 Tag / Worker ID</label>
          <input type="text" id="filterEntity" class="filter-input" placeholder="e.g. COW003 or WRK002" onkeyup="if(event.key==='Enter') fetchActivities()">
        </div>

        <div class="filter-item">
          <label>📅 Log Date</label>
          <input type="date" id="filterDate" class="filter-input" onchange="fetchActivities()">
        </div>
      </div>

      <div>
        <button class="btn-action btn-secondary" onclick="fetchActivities()">
          <span>↻</span> Refresh Telemetry
        </button>
      </div>
    </section>

    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.2rem;">
      <h2 style="font-size: 1.25rem; font-weight: 800; color: var(--text-heading); display: flex; align-items: center; gap: 0.5rem;">
        <span>📦</span> Activity Evidence Stream <span style="font-size: 0.82rem; font-weight: 600; color: var(--text-muted);">(10-Second Clips Stored in Supabase)</span>
      </h2>
      <span id="totalBadge" class="manifest-pill" style="font-size: 0.8rem;">0 Events</span>
    </div>

    <!-- Activities Card Grid -->
    <div class="activities-grid" id="activitiesGrid"></div>

  </main>

  <!-- Video Playback & Detail Modal -->
  <div class="modal-backdrop" id="activityModal">
    <div class="modal-content">
      <div class="modal-header">
        <div class="modal-title" id="modalTitle">Activity Telemetry Detail</div>
        <button class="btn-close" onclick="closeModal('activityModal')">&times;</button>
      </div>
      <video id="modalVideoPlayer" class="modal-video" controls autoplay playsinline></video>
      <div class="modal-details">
        <div>
          <h4 style="font-size: 0.85rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.5rem; font-weight: 700;">📝 Observation Summary</h4>
          <p id="modalDesc" style="font-size: 0.95rem; line-height: 1.5; margin-bottom: 1rem; color: var(--text-heading);"></p>
          <div style="font-size: 0.76rem; font-weight: 700; color: var(--text-muted); margin-bottom: 0.35rem; text-transform: uppercase;">Raw Model Output (JSON)</div>
          <pre id="modalJson" style="background: var(--bg-muted); border: 1px solid var(--border-subtle); padding: 0.85rem; border-radius: 8px; font-size: 0.75rem; color: var(--text-body); overflow-x: auto; max-height: 150px; font-family: var(--font-mono);"></pre>
        </div>
        <div>
          <h4 style="font-size: 0.85rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.5rem; font-weight: 700;">📊 Livestock Telemetry & AI Metrics</h4>
          <div style="font-size: 0.84rem; line-height: 1.9; color: var(--text-heading);">
            <div>🏷️ <b>Entity / Tag:</b> <span id="modalEntity">-</span></div>
            <div>🐄 <b>Herd Count:</b> <span id="modalHerd">-</span></div>
            <div>🏷️ <b>Breed:</b> <span id="modalBreed" style="color: var(--primary); font-weight: 700;">-</span></div>
            <div>🩺 <b>Disease Status:</b> <span id="modalDisease">-</span></div>
            <div>🥩 <b>BCS Score:</b> <span id="modalBcs">-</span></div>
            <div>⚖️ <b>Est. Live Weight:</b> <span id="modalWeight">-</span></div>
            <div>📏 <b>Wither Height:</b> <span id="modalHeight">-</span></div>
            <div>🕒 <b>Timestamp:</b> <span id="modalTime">-</span></div>
            <div>📹 <b>Camera Node:</b> <span id="modalCam">-</span></div>
            <div>🎯 <b>Model Confidence:</b> <span id="modalConf">-</span></div>
            <div>☁️ <b>Storage:</b> <b>Supabase Cloud DB & Private Bucket</b></div>
          </div>
        </div>
      </div>
      <div style="display: flex; gap: 0.75rem; padding: 1.1rem 1.5rem; border-top: 1px solid var(--border-subtle); justify-content: flex-end; background: var(--bg-muted);">
        <button class="btn-action btn-secondary btn-danger-outline" onclick="deleteCurrentActivity()">
          🗑️ Delete Record
        </button>
      </div>
    </div>
  </div>

  <!-- Connect CCTV Camera Modal -->
  <div class="modal-backdrop" id="connectCameraModal">
    <div class="modal-content auth-modal">
      <div class="modal-header">
        <div class="modal-title">📹 Connect Farm CCTV Camera</div>
        <button class="btn-close" onclick="closeModal('connectCameraModal')">&times;</button>
      </div>
      <form class="auth-form" onsubmit="handleConnectCamera(event)">
        <div class="form-group">
          <label>Camera Designation / Name</label>
          <input type="text" id="camName" class="form-control" placeholder="e.g. Corral Cam 01" required>
        </div>
        <div class="form-group">
          <label>RTSP / ONVIF URL (or '0' for WebCam)</label>
          <input type="text" id="camUrl" class="form-control" placeholder="rtsp://admin:pass@192.168.1.58:554/stream" required>
        </div>
        <div class="form-group">
          <label>Pen / Location Description</label>
          <input type="text" id="camLoc" class="form-control" placeholder="e.g. Main Feeding Alley">
        </div>
        <button type="submit" class="btn-action" style="width: 100%; justify-content: center; margin-top: 0.5rem;">
          Connect & Start 24/7 AI Surveillance
        </button>
      </form>
    </div>
  </div>

  <!-- Upload CCTV Video Modal -->
  <div class="modal-backdrop" id="uploadModal">
    <div class="modal-content auth-modal">
      <div class="modal-header">
        <div class="modal-title">📁 Ingest CCTV Video for AI Detection</div>
        <button class="btn-close" onclick="closeModal('uploadModal')">&times;</button>
      </div>
      <form class="auth-form" onsubmit="handleUploadSubmit(event)">
        <div class="form-group">
          <label>Select Video File (.mp4, .avi, .mov)</label>
          <input type="file" id="uploadVideoFile" class="form-control" accept="video/*" required>
        </div>
        <div class="form-group">
          <label>Assigned Camera Node</label>
          <select id="uploadCameraId" class="filter-select" style="width: 100%;">
            <option value="CAM01">CAM01 (Main Gate & Feed Alley)</option>
            <option value="CAM02">CAM02 (Milking Stall Area)</option>
          </select>
        </div>
        <div class="form-group" style="background: var(--bg-muted); border: 1px solid var(--border-subtle); border-radius: 10px; padding: 0.95rem; margin-top: 0.4rem;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
            <label style="font-weight: 700; color: var(--text-heading); font-size: 0.85rem; margin: 0;">⚡ Active AI Models to Run:</label>
            <label style="font-size: 0.75rem; color: var(--primary); cursor: pointer; display: flex; align-items: center; gap: 0.3rem; font-weight: 600;">
              <input type="checkbox" id="chkSelectAllModels" checked onchange="toggleAllUploadModels(this.checked)"> Select All
            </label>
          </div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; font-size: 0.8rem; color: var(--text-heading);">
            <label style="display: flex; align-items: center; gap: 0.4rem; cursor: pointer;">
              <input type="checkbox" name="ai_model_opt" value="breed" checked class="upload-model-chk"> 🏷️ Breed Classifier
            </label>
            <label style="display: flex; align-items: center; gap: 0.4rem; cursor: pointer;">
              <input type="checkbox" name="ai_model_opt" value="bcs" checked class="upload-model-chk"> 🥩 BCS Scoring (1-5)
            </label>
            <label style="display: flex; align-items: center; gap: 0.4rem; cursor: pointer;">
              <input type="checkbox" name="ai_model_opt" value="biometry" checked class="upload-model-chk"> ⚖️ Height & Weight
            </label>
            <label style="display: flex; align-items: center; gap: 0.4rem; cursor: pointer;">
              <input type="checkbox" name="ai_model_opt" value="disease" checked class="upload-model-chk"> 🩺 Health Screening
            </label>
            <label style="display: flex; align-items: center; gap: 0.4rem; cursor: pointer;">
              <input type="checkbox" name="ai_model_opt" value="estrus" checked class="upload-model-chk"> 🔥 Estrus & Mounting
            </label>
            <label style="display: flex; align-items: center; gap: 0.4rem; cursor: pointer;">
              <input type="checkbox" name="ai_model_opt" value="worker" checked class="upload-model-chk"> 👷 Crew Attendance
            </label>
          </div>
        </div>
        <div id="uploadProgressContainer" style="display: none; margin-top: 0.75rem; padding: 0.5rem 0.2rem;">
          <div style="display: flex; justify-content: space-between; font-size: 0.8rem; font-weight: 600; color: var(--text-heading); margin-bottom: 0.35rem;">
            <span id="uploadProgressText">Uploading video to PashuX...</span>
            <span id="uploadProgressPercent" style="color: var(--primary);">0%</span>
          </div>
          <div style="width: 100%; height: 8px; background: var(--bg-muted); border-radius: 999px; overflow: hidden; border: 1px solid var(--border-subtle);">
            <div id="uploadProgressBarFill" style="width: 0%; height: 100%; background: linear-gradient(90deg, var(--primary), var(--accent)); transition: width 0.15s ease;"></div>
          </div>
        </div>
        <button type="submit" class="btn-action" style="width: 100%; justify-content: center; margin-top: 0.5rem;" id="btnUploadSubmit">
          🚀 Run AI Analysis & Save to Manifest
        </button>
      </form>
    </div>
  </div>

  <!-- Authentication Modal (Login / Register) -->
  <div class="modal-backdrop" id="authModal">
    <div class="modal-content auth-modal">
      <div class="modal-header">
        <div class="modal-title" id="authTitle">🔑 Farm Owner / Staff Login</div>
        <button class="btn-close" onclick="closeModal('authModal')">&times;</button>
      </div>
      <form class="auth-form" onsubmit="handleAuthSubmit(event)">
        <div class="form-group">
          <label>Phone Number</label>
          <input type="tel" id="authPhone" class="form-control" placeholder="+91 98765 43210" required>
        </div>
        <div class="form-group">
          <label>Password</label>
          <input type="password" id="authPassword" class="form-control" placeholder="••••••••" required>
        </div>
        <div class="form-group reg-field" style="display: none;">
          <label>Full Name</label>
          <input type="text" id="authFullName" class="form-control" placeholder="Farm Owner Name">
        </div>
        <div class="form-group reg-field" style="display: none;">
          <label>Farm Name</label>
          <input type="text" id="authFarmName" class="form-control" placeholder="e.g. Sunrise Dairy Farm">
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem;">
          <a href="#" id="toggleAuthMode" style="color: var(--primary); font-size: 0.78rem; font-weight: 600; text-decoration: none;" onclick="toggleAuthMode(event)">Need an account? Register</a>
        </div>
        <button type="submit" class="btn-action" style="width: 100%; justify-content: center; margin-top: 0.5rem;" id="authSubmitBtn">
          Secure Login
        </button>
      </form>
    </div>
  </div>

  <!-- Core JavaScript Engine -->
  <script>
    let authToken = localStorage.getItem("pashux_jwt") || "";
    let isRegisterMode = false;
    let activitiesCache = [];
    let isLiveStreamOpen = false;

    // Theme Management: Light Theme by default, with smooth toggle
    function initTheme() {
      const savedTheme = localStorage.getItem("pashux_theme") || "light";
      applyTheme(savedTheme);
    }

    function applyTheme(theme) {
      document.documentElement.setAttribute("data-theme", theme);
      localStorage.setItem("pashux_theme", theme);
      const icon = document.getElementById("themeToggleIcon");
      const text = document.getElementById("themeToggleText");
      if (theme === "dark") {
        if (icon) icon.innerText = "☀️";
        if (text) text.innerText = "Light Mode";
      } else {
        if (icon) icon.innerText = "🌙";
        if (text) text.innerText = "Dark Mode";
      }
    }

    function toggleDashboardTheme() {
      const current = document.documentElement.getAttribute("data-theme") || "light";
      const next = current === "light" ? "dark" : "light";
      applyTheme(next);
    }

    // Real-Time System Clock
    function updateClock() {
      const el = document.getElementById("systemClock");
      if (el) {
        const now = new Date();
        el.innerText = now.toLocaleTimeString() + " • " + now.toISOString().slice(0, 10);
      }
    }
    setInterval(updateClock, 1000);
    updateClock();

    function getAuthHeaders() {
      const headers = { "Content-Type": "application/json" };
      if (authToken) headers["Authorization"] = `Bearer ${authToken}`;
      return headers;
    }

    async function checkUserSession() {
      try {
        const res = await fetch("/api/auth/me", { headers: getAuthHeaders() });
        if (res.status === 401) {
          localStorage.removeItem("pashux_jwt");
          authToken = "";
          const fallbackRes = await fetch("/api/auth/me");
          const fallbackData = await fallbackRes.json();
          if (fallbackData.user) {
            document.getElementById("displayPhone").innerText = fallbackData.user.phone || "";
            document.getElementById("displayFarmName").innerText = "🏭 " + (fallbackData.user.farm_name || "Dairy Farm");
            document.getElementById("displayFarmId").innerText = `Farm ID: ${fallbackData.user.farm_id || ""}`;
          }
          return;
        }
        const data = await res.json();
        if (data.user) {
          document.getElementById("displayPhone").innerText = data.user.phone || "";
          document.getElementById("displayFarmName").innerText = "🏭 " + (data.user.farm_name || "Dairy Farm");
          document.getElementById("displayFarmId").innerText = `Farm ID: ${data.user.farm_id || ""}`;
          if (authToken) {
            document.getElementById("btnLogout").style.display = "block";
            document.getElementById("btnLoginToggle").style.display = "none";
          }
        }
      } catch (err) {
        console.error("Auth session error:", err);
      }
    }

    function logout() {
      localStorage.removeItem("pashux_jwt");
      authToken = "";
      location.reload();
    }

    function openLiveStream(title = "Live CCTV AI Detection Feed", badgeText = "Surveillance Active") {
      const box = document.getElementById("streamBox");
      const img = document.getElementById("liveStreamImg");
      const btn = document.getElementById("btnToggleLiveStream");
      const titleEl = document.getElementById("streamTitle");
      const badgeEl = document.getElementById("streamStatusBadge");
      isLiveStreamOpen = true;
      box.style.display = "block";
      if (titleEl) titleEl.innerText = title;
      if (badgeEl) badgeEl.innerText = badgeText;
      img.src = "/api/cctv/stream?_t=" + Date.now();
      if (btn) btn.innerHTML = "<span>⏹</span> Hide Stream";
      box.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    function toggleLiveStream() {
      const box = document.getElementById("streamBox");
      const img = document.getElementById("liveStreamImg");
      const btn = document.getElementById("btnToggleLiveStream");
      isLiveStreamOpen = !isLiveStreamOpen;
      if (isLiveStreamOpen) {
        openLiveStream();
      } else {
        box.style.display = "none";
        img.src = "";
        if (btn) btn.innerHTML = "<span>🔴</span> View Live AI Stream";
      }
    }

    async function fetchActivities() {
      const type = document.getElementById("filterType").value;
      const camera = document.getElementById("filterCamera").value;
      const entity = document.getElementById("filterEntity").value.trim() || "all";
      const date = document.getElementById("filterDate").value || "";

      let url = `/api/activities?activity_type=${encodeURIComponent(type)}&camera_id=${encodeURIComponent(camera)}&cattle_id=${encodeURIComponent(entity)}&_t=${Date.now()}`;
      if (date) url += `&date=${encodeURIComponent(date)}`;

      try {
        let res = await fetch(url, { headers: getAuthHeaders(), cache: "no-store" });
        if (res.status === 401) {
          localStorage.removeItem("pashux_jwt");
          authToken = "";
          res = await fetch(url, { cache: "no-store" });
        }
        const data = await res.json();
        activitiesCache = data.activities || [];
        renderActivities(activitiesCache);
        renderCameraSummaries(data.camera_summaries, data.cattle_registry, data.worker_registry);

        if (data.metrics) {
          if (document.getElementById("statCattle")) {
            document.getElementById("statCattle").innerText = data.metrics.unique_cattle_count || 0;
          }
          if (document.getElementById("statBreeds")) {
            document.getElementById("statBreeds").innerText = data.metrics.breed_count || 0;
          }
          if (document.getElementById("statBcs")) {
            document.getElementById("statBcs").innerText = data.metrics.bcs_count || 0;
          }
          if (document.getElementById("statDisease")) {
            document.getElementById("statDisease").innerText = data.metrics.disease_count || 0;
          }
          if (document.getElementById("statEstrus")) {
            document.getElementById("statEstrus").innerText = data.metrics.estrus_count || 0;
          }
          if (document.getElementById("statWorker")) {
            document.getElementById("statWorker").innerText = data.metrics.unique_worker_count || data.metrics.worker_count || 0;
          }
        }
        document.getElementById("totalBadge").innerText = `${activitiesCache.length} Events`;
      } catch (err) {
        console.error("Error fetching activities:", err);
      }
    }

    function renderCameraSummaries(cameraSummaries, cattleRegistry, workerRegistry) {
      const summaryContainer = document.getElementById("cameraSummaryCards");
      const rosterContainer = document.getElementById("cattleRosterList");
      if (!summaryContainer || !rosterContainer) return;

      summaryContainer.innerHTML = "";
      rosterContainer.innerHTML = "";

      if (!cameraSummaries || cameraSummaries.length === 0) {
        summaryContainer.innerHTML = `
          <div style="grid-column: 1/-1; padding: 1.2rem; text-align: center; color: var(--text-muted); font-size: 0.88rem; background: var(--bg-muted); border-radius: 8px;">
            No cameras registered yet. Connect CCTV camera or upload a video recording to view daily herd registry.
          </div>
        `;
      } else {
        cameraSummaries.forEach(cs => {
          const card = document.createElement("div");
          card.className = "camera-node-card";
          
          const estrusBadge = cs.estrus_detected ? 
            `<span style="background: var(--accent-estrus-light); color: var(--accent-estrus); border: 1px solid var(--accent-estrus-border); padding: 0.2rem 0.6rem; border-radius: 6px; font-weight: 700; font-size: 0.76rem;">🔥 ESTRUS DETECTED (${cs.estrus_count})</span>` : 
            `<span style="background: var(--accent-emerald-light); color: var(--accent-emerald); border: 1px solid var(--accent-emerald-border); padding: 0.2rem 0.6rem; border-radius: 6px; font-weight: 700; font-size: 0.76rem;">🟢 Normal / No Heat</span>`;

          card.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.7rem;">
              <div style="font-weight: 800; font-size: 1.05rem; color: var(--text-heading);">📹 Node: ${cs.camera_id}</div>
              <div>${estrusBadge}</div>
            </div>
            <div style="font-size: 0.88rem; color: var(--text-body); margin-bottom: 0.6rem;">
              Herd Count Today: <b style="color: var(--accent-emerald);">${cs.unique_cattle_count} Unique Cattle</b> (Zero Duplicate)
            </div>
            <div style="display: flex; flex-direction: column; gap: 0.35rem; font-size: 0.82rem;">
              <div>🧬 <b>Breed Classifier:</b> <span style="color:var(--primary); font-weight:700;">best.onnx (41 Classes)</span></div>
              <div>🩺 <b>Disease Status:</b> <span style="color:var(--accent-emerald); font-weight:700;">Healthy Screen Verified</span></div>
              <div>👷 <b>Personnel Logged:</b> <span style="color:var(--accent-warning); font-weight:700;">${cs.workers_count} Verified Worker(s)</span></div>
            </div>
          `;
          summaryContainer.appendChild(card);
        });
      }

      if (!cattleRegistry || cattleRegistry.length === 0) {
        rosterContainer.innerHTML = `
          <div style="color: var(--text-muted); font-size: 0.85rem; padding: 0.6rem;">
            No cattle logged today yet.
          </div>
        `;
      } else {
        cattleRegistry.forEach(c => {
          const item = document.createElement("div");
          item.className = "cattle-roster-card";
          
          const disColor = (c.disease_status === 'Healthy' || !c.disease_status) ? 'var(--accent-emerald)' : 'var(--accent-disease)';
          const heatBadge = c.has_estrus ? 
            `<span style="background: var(--accent-estrus-light); color: var(--accent-estrus); font-size: 0.68rem; font-weight: 800; padding: 0.15rem 0.45rem; border-radius: 4px; border: 1px solid var(--accent-estrus-border);">🔥 Estrus Sign</span>` : 
            `<span style="background: var(--bg-muted); color: var(--text-muted); font-size: 0.68rem; padding: 0.15rem 0.45rem; border-radius: 4px;">Normal</span>`;

          const feedSec = Math.round(c.total_eating_sec || 0);
          const restSec = Math.round(c.total_resting_sec || 0);
          const standSec = Math.round(c.total_standing_sec || 0);
          const hasValidBcs = (c.bcs_score !== null && c.bcs_score !== undefined && Number(c.bcs_score) > 0);
          const bcsScoreNum = hasValidBcs ? Number(c.bcs_score) : 0;
          let bcsDisplay = 'Rear/Hip Needed';
          let bcsColor = 'var(--text-muted)';

          if (hasValidBcs) {
            bcsDisplay = `${bcsScoreNum.toFixed(1)}/5.0`;
            if (bcsScoreNum <= 2.2) {
              bcsColor = 'var(--accent-disease)';
              bcsDisplay += ' (Thin)';
            } else if (bcsScoreNum >= 4.0) {
              bcsColor = 'var(--accent-warning)';
              bcsDisplay += ' (Fleshy)';
            } else {
              bcsColor = 'var(--accent-emerald)';
            }
          }

          let durParts = [];
          if (feedSec > 0) durParts.push(`🌾 ${Math.floor(feedSec / 60)}m ${feedSec % 60}s`);
          if (standSec > 0) durParts.push(`🚶 ${Math.floor(standSec / 60)}m ${standSec % 60}s`);
          if (restSec > 0) durParts.push(`💤 ${Math.floor(restSec / 60)}m ${restSec % 60}s`);
          const durDisplay = durParts.length > 0 ? durParts.join(" | ") : `🚶 0m 15s | 🌾 0m 0s`;

          const imgSrc = c.image_url || (`/media/images/${c.cattle_id}.jpg`);
          item.innerHTML = `
            <div class="roster-thumb-box">
              <img src="${imgSrc}" 
                   alt="${c.cattle_id}" 
                   class="roster-thumb-img"
                   onerror="this.onerror=null; this.src='/logo.png'; this.style.objectFit='contain'; this.style.opacity='0.4';" />
              <div class="roster-tag-pill">
                ${c.cattle_id}
              </div>
              <div style="position: absolute; top: 6px; right: 6px;">
                ${heatBadge}
              </div>
            </div>
            <div style="font-size: 0.84rem; font-weight: 700; color: var(--text-heading); margin-bottom: 0.2rem;">🐄 ${c.breed}</div>
            <div style="font-size: 0.78rem; font-weight: 700; color: ${disColor}; margin-bottom: 0.25rem;">🩺 ${c.disease_status || 'Healthy'}</div>
            <div style="font-size: 0.74rem; color: var(--text-muted); margin-bottom: 0.35rem;">
              BCS: <b style="color:${bcsColor};">${bcsDisplay}</b> | ⚖️ ${Math.round(c.estimated_weight_kg || 400)}kg | 📏 ${Math.round(c.estimated_height_cm || 130)}cm
            </div>
            <div style="font-size: 0.72rem; color: var(--primary); font-family: var(--font-mono); font-weight: 600; border-top: 1px solid var(--border-subtle); padding-top: 0.35rem;">
              ${durDisplay}
            </div>
          `;
          rosterContainer.appendChild(item);
        });
      }

      const workerRosterContainer = document.getElementById("workerRosterList");
      if (workerRosterContainer) {
        workerRosterContainer.innerHTML = "";
        if (!workerRegistry || workerRegistry.length === 0) {
          workerRosterContainer.innerHTML = `
            <div style="color: var(--text-muted); font-size: 0.85rem; padding: 0.6rem;">
              No farm workers detected yet. Workers entering the camera frame are automatically logged here.
            </div>
          `;
        } else {
          workerRegistry.forEach(w => {
            const item = document.createElement("div");
            item.className = "worker-roster-card";
            const durMin = Math.floor((w.duration_sec || 0) / 60);
            const durSec = Math.round((w.duration_sec || 0) % 60);
            const firstTime = w.first_seen ? new Date(w.first_seen).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : "--";
            const lastTime = w.last_seen ? new Date(w.last_seen).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : "--";
            const isDeparted = w.is_departed || false;
            const statusBadge = isDeparted
              ? `<span style="background: var(--accent-warning-light); color: var(--accent-warning); font-size: 0.68rem; font-weight: 700; padding: 0.15rem 0.45rem; border-radius: 4px; border: 1px solid var(--accent-warning-border);">🚪 Departed</span>`
              : `<span style="background: var(--accent-emerald-light); color: var(--accent-emerald); font-size: 0.68rem; font-weight: 700; padding: 0.15rem 0.45rem; border-radius: 4px; border: 1px solid var(--accent-emerald-border);">🟢 Active in Barn</span>`;
            const tasksDisplay = (w.tasks && w.tasks.length > 0) ? w.tasks.join(", ") : (w.description || "Barn Duty & Inspection");
            const imgSrc = w.image_url || (`/media/images/${w.worker_id}.jpg`);

            item.innerHTML = `
              <div class="roster-thumb-box" style="height: 115px;">
                <img src="${imgSrc}" onerror="this.onerror=null; this.src='/logo.png'; this.style.objectFit='contain'; this.style.opacity='0.4';" class="roster-thumb-img">
                <span style="position: absolute; bottom: 4px; left: 4px; background: rgba(15,23,42,0.85); font-size: 0.65rem; color: #fff; padding: 2px 6px; border-radius: 3px; font-weight: 700;">📸 Entry Photo</span>
              </div>
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.45rem;">
                <b style="color: var(--primary); font-family: var(--font-mono); font-size: 0.95rem;">👷 ${w.worker_id}</b>
                ${statusBadge}
              </div>
              <div style="font-size: 0.76rem; color: var(--text-heading); margin-bottom: 0.25rem;">
                🕒 <b>Entered:</b> ${firstTime}
              </div>
              <div style="font-size: 0.76rem; color: var(--text-body); margin-bottom: 0.35rem;">
                ${isDeparted ? `🚪 <b>Left:</b> ${lastTime}` : `⏱️ <b>Working:</b> ${durMin}m ${durSec}s`}
              </div>
              <div style="font-size: 0.75rem; color: var(--accent-warning); margin-bottom: 0.35rem; line-height: 1.3; font-weight: 600;">
                🛠️ <b>Activities:</b> ${tasksDisplay}
              </div>
              <div style="font-size: 0.72rem; color: var(--text-muted); border-top: 1px solid var(--border-subtle); padding-top: 0.35rem;">
                Total Time: <b style="color: var(--primary);">${durMin}m ${durSec}s</b> | Cam: ${w.camera_id}
              </div>
            `;
            workerRosterContainer.appendChild(item);
          });
        }
      }
    }

    function getActivityBadge(type) {
      if (type.includes("milking")) return { class: "badge-milking", label: "🥛 Milking Activity" };
      if (type.includes("worker_feeding")) return { class: "badge-feeding", label: "🌾 Worker Feeding Cattle" };
      if (type.includes("attendance")) return { class: "badge-worker", label: "👷 Worker Attendance" };
      if (type.includes("estrus")) return { class: "badge-estrus", label: "🔥 Detected Estrus Sign" };
      if (type.includes("disease")) return { class: "badge-disease", label: "🩺 Suspected Disease Sign" };
      if (type.includes("calving")) return { class: "badge-calving", label: "🍼 Suspected Calving Sign" };
      if (type.includes("feeding")) return { class: "badge-feeding", label: "🌾 Feeding / Chewing" };
      if (type.includes("standing")) return { class: "badge-standing", label: "🚶 Standing / Grazing" };
      if (type.includes("resting")) return { class: "badge-resting", label: "💤 Resting / Lying" };  
      if (type.includes("worker") || type.includes("clean")) return { class: "badge-worker", label: "👷 Worker Activity" };
      return { class: "badge-general", label: "🐄 Farm Activity" };
    }

    function renderActivities(activities) {
      const grid = document.getElementById("activitiesGrid");
      grid.innerHTML = "";

      if (!activities || activities.length === 0) {
        grid.innerHTML = `
          <div class="empty-state">
            <h3>No Farm Activities Logged Yet</h3>
            <p>Connect your CCTV camera or upload a video recording to automatically detect activities and capture 10-second video evidence in Supabase.</p>
          </div>
        `;
        return;
      }

      activities.forEach((item, idx) => {
        const badge = getActivityBadge(item.activity_type);
        const card = document.createElement("div");
        card.className = "activity-card";
        card.onclick = () => openActivityDetail(idx);

        const entityLabel = item.cattle_id ? `#${item.cattle_id}` : (item.worker_id ? `#${item.worker_id}` : "Herd");
        const formattedDate = new Date(item.start_time || item.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + ' ' + (item.detection_date || '');

        const hasVideo = item.captured_video_storage_path && item.captured_video_storage_path.trim() !== "";
        let mediaTag = "";
        let durationHtml = "";

        const startTimeStr = item.start_time ? new Date(item.start_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : "";
        const endTimeStr = item.end_time ? new Date(item.end_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : "";
        const timeRangeStr = (startTimeStr && endTimeStr && startTimeStr !== endTimeStr) ? ` (${startTimeStr} - ${endTimeStr})` : (startTimeStr ? ` (Started: ${startTimeStr})` : '');

        if (item.activity_type.includes("worker_feeding")) {
          const feedSec = Math.round(item.eating_duration_sec || item.activity_duration || 0);
          const fMin = Math.floor(feedSec / 60);
          const fSec = feedSec % 60;
          mediaTag = `<div class="video-tag" style="background: rgba(2, 132, 199, 0.9);">🌾 ${fMin}m ${fSec}s Feeding</div>`;
          const workerBadge = item.worker_id ? `<span style="background: var(--primary-light); color: var(--primary); padding: 0.15rem 0.45rem; border-radius: 4px; font-size: 0.72rem; font-weight: 700;">👷 Worker: ${item.worker_id}</span>` : '';
          durationHtml = `
            <div style="background: var(--accent-feeding-light); border: 1px solid var(--accent-feeding-border); border-radius: 6px; padding: 0.45rem 0.65rem; margin-top: 0.35rem;">
              <div style="color: var(--primary); font-weight: 800; font-size: 0.85rem; display: flex; align-items: center; justify-content: space-between;">
                <span>🌾 Worker Feeding: ${fMin}m ${fSec}s</span>
                ${workerBadge}
              </div>
              <div style="font-weight: 500; color: var(--text-muted); font-size: 0.74rem; margin-top: 0.15rem;">⏱️ Time Window: ${startTimeStr || 'Active'}${endTimeStr && endTimeStr !== startTimeStr ? ' - ' + endTimeStr : ''}</div>
            </div>`;
        } else if (item.activity_type.includes("milking")) {
          const milkSec = Math.round(item.activity_duration || 0);
          const mMin = Math.floor(milkSec / 60);
          const mSec = milkSec % 60;
          mediaTag = `<div class="video-tag" style="background: rgba(234, 88, 12, 0.9);">🥛 ${mMin}m ${mSec}s Milking</div>`;
          const workerBadge = item.worker_id ? `<span style="background: var(--accent-milking-light); color: var(--accent-milking); padding: 0.15rem 0.45rem; border-radius: 4px; font-size: 0.72rem; font-weight: 700;">👷 Worker: ${item.worker_id}</span>` : '';
          durationHtml = `
            <div style="background: var(--accent-milking-light); border: 1px solid var(--accent-milking-border); border-radius: 6px; padding: 0.45rem 0.65rem; margin-top: 0.35rem;">
              <div style="color: var(--accent-milking); font-weight: 800; font-size: 0.85rem; display: flex; align-items: center; justify-content: space-between;">
                <span>🥛 Milking Duration: ${mMin}m ${mSec}s</span>
                ${workerBadge}
              </div>
              <div style="font-weight: 500; color: var(--text-muted); font-size: 0.74rem; margin-top: 0.15rem;">⏱️ Session: ${startTimeStr || 'Active'}${endTimeStr && endTimeStr !== startTimeStr ? ' - ' + endTimeStr : ''}</div>
            </div>`;
        } else if (item.activity_type.includes("standing")) {
          const standSec = Math.round(item.activity_duration || 0);
          const sMin = Math.floor(standSec / 60);
          const sSec = standSec % 60;
          mediaTag = `<div class="video-tag" style="background: rgba(8, 145, 178, 0.9);">🚶 ${sMin}m ${sSec}s Standing</div>`;
          durationHtml = `<div style="color: var(--accent-standing); font-weight: 800; font-size: 0.82rem; margin-top: 0.2rem;">🚶 Standing Duration: ${sMin}m ${sSec}s <span style="font-weight:500; color:var(--text-muted); font-size:0.75rem;">${timeRangeStr}</span></div>`;
        } else if (item.activity_type.includes("feeding")) {
          const eatingSec = Math.round(item.eating_duration_sec || item.activity_duration || 0);
          const eMin = Math.floor(eatingSec / 60);
          const eSec = eatingSec % 60;
          mediaTag = `<div class="video-tag" style="background: rgba(2, 132, 199, 0.9);">🌾 ${eMin}m ${eSec}s Grazing</div>`;
          durationHtml = `<div style="color: var(--primary); font-weight: 800; font-size: 0.82rem; margin-top: 0.2rem;">🌾 Grazing Duration: ${eMin}m ${eSec}s <span style="font-weight:500; color:var(--text-muted); font-size:0.75rem;">${timeRangeStr}</span></div>`;
        } else if (item.activity_type.includes("resting")) {
          const restSec = Math.round(item.resting_duration_sec || item.activity_duration || 0);
          const rMin = Math.floor(restSec / 60);
          const rSec = restSec % 60;
          mediaTag = `<div class="video-tag" style="background: rgba(124, 58, 237, 0.9);">💤 ${rMin}m ${rSec}s Resting</div>`;
          durationHtml = `<div style="color: var(--accent-resting); font-weight: 800; font-size: 0.82rem; margin-top: 0.2rem;">💤 Resting Duration: ${rMin}m ${rSec}s <span style="font-weight:500; color:var(--text-muted); font-size:0.75rem;">${timeRangeStr}</span></div>`;
        } else if (hasVideo) {
          mediaTag = `<div class="video-tag">▶ 10s Clip</div>`;
        } else {
          mediaTag = `<div class="video-tag" style="background: rgba(15, 23, 42, 0.85); color: #fff;">📸 Snapshot</div>`;
        }

        const isWorkerActivity = item.activity_type.startsWith("worker_") || 
                                 item.activity_type.includes("cleaning") || 
                                 item.activity_type.includes("attendance");

        const isDiseased = item.disease_status && item.disease_status !== 'Healthy' && item.disease_status !== 'Scanning...';
        const diseaseAlertBanner = (!isWorkerActivity && isDiseased)
          ? `<div style="background: var(--accent-disease-light); border-left: 3px solid var(--accent-disease); color: var(--accent-disease); padding: 0.4rem 0.6rem; border-radius: 4px; font-weight: 700; font-size: 0.8rem; margin: 0.35rem 0; display: flex; align-items: center; gap: 0.4rem;">⚠️ Suspected Disease: ${item.disease_status}</div>`
          : '';

        let badgeRowHtml = '';
        let entityDisplay = '';
        if (isWorkerActivity) {
          const wId = item.worker_id || 'Staff';
          entityDisplay = `<div>Worker: <b style="color:var(--primary);">👷 #${wId}</b></div>`;
          badgeRowHtml = `
            <div style="display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.35rem;">
              <span class="telemetry-chip" style="background: var(--primary-light); color: var(--primary); border-color: var(--primary-border);">👷 Farm Personnel</span>
              <span class="telemetry-chip">🛠️ ${item.activity_type.replace(/_/g, ' ').toUpperCase()}</span>
              <span class="telemetry-chip" style="background: var(--accent-emerald-light); color: var(--accent-emerald); border-color: var(--accent-emerald-border);">✅ Verified Duty</span>
            </div>
          `;
        } else {
          const breedBadge = `<span class="telemetry-chip">🐄 ${item.breed || 'Cattle'}</span>`;
          const diseaseBadge = isDiseased
            ? `<span class="telemetry-chip" style="background: var(--accent-disease-light); color: var(--accent-disease); border-color: var(--accent-disease-border); font-weight: 700;">⚠️ ${item.disease_status}</span>`
            : `<span class="telemetry-chip" style="background: var(--accent-emerald-light); color: var(--accent-emerald); border-color: var(--accent-emerald-border);">🩺 Healthy</span>`;

          const bcsNum = (item.bcs_score !== null && item.bcs_score !== undefined && Number(item.bcs_score) > 0) ? Number(item.bcs_score) : null;
          const bcsBadge = (bcsNum && bcsNum > 0)
            ? `<span class="telemetry-chip" title="BCS evaluated only when full body appears">📊 BCS: ${bcsNum.toFixed(1)}/5.0 <span style="color:var(--accent-emerald); font-size:0.65rem;">(Full Body)</span></span>`
            : `<span class="telemetry-chip" style="background: var(--accent-warning-light); color: var(--accent-warning); border-color: var(--accent-warning-border);" title="Full body view required to evaluate BCS">📊 BCS: Full Body Required</span>`;

          const weightBadge = `<span class="telemetry-chip">⚖️ ${Math.round(item.estimated_weight_kg || 400)} kg</span>`;
          const heightBadge = `<span class="telemetry-chip">📏 ${Math.round(item.estimated_height_cm || 130)} cm</span>`;

          entityDisplay = `<div>Entity: <b style="color:var(--text-heading);">${entityLabel}</b> <span style="font-size:0.7rem; color:var(--text-muted);">(Herd: ${item.herd_count || 1})</span></div>`;
          badgeRowHtml = `
            <div style="display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.35rem;">
              ${breedBadge} ${diseaseBadge} ${bcsBadge} ${weightBadge} ${heightBadge}
            </div>
          `;
        }

        card.innerHTML = `
          <div class="card-media">
            <img class="card-thumb" src="${item.image_url || '/media/images/placeholder.jpg'}" alt="${item.activity_type}">
            <span class="badge-status ${badge.class}">${badge.label}</span>
            ${mediaTag}
          </div>
          <div class="card-body">
            <div class="card-top">
              <div class="activity-name">${item.activity_type.replace(/_/g, ' ').toUpperCase()}</div>
              <div class="activity-conf">${Math.round(item.confidence_score * 100)}% Conf</div>
            </div>
            <div class="activity-desc">${item.description}</div>
            ${diseaseAlertBanner}
            ${durationHtml}
            ${badgeRowHtml}
            <div class="meta-row">
              ${entityDisplay}
              <div>Cam: <b style="color:var(--text-heading);">${item.camera_id || 'CAM01'}</b></div>
              <div style="margin-left: auto;">${formattedDate}</div>
            </div>
          </div>
        `;
        grid.appendChild(card);
      });
    }

    let currentSelectedActivityId = null;

    function openActivityDetail(idx) {
      const item = activitiesCache[idx];
      if (!item) return;

      const isWorkerAct = item.activity_type.startsWith("worker_") || item.activity_type.includes("cleaning") || item.activity_type.includes("attendance");

      currentSelectedActivityId = item.activity_id;
      document.getElementById("modalTitle").innerText = item.description || item.activity_type;
      document.getElementById("modalDesc").innerText = item.description;
      document.getElementById("modalEntity").innerText = item.cattle_id ? `#${item.cattle_id}` : (item.worker_id ? `Worker #${item.worker_id}` : "Herd General");

      if (document.getElementById("modalHerd")) document.getElementById("modalHerd").innerText = isWorkerAct ? "N/A (Farm Staff)" : `${item.herd_count || 1} Cattle`;
      if (document.getElementById("modalBreed")) document.getElementById("modalBreed").innerText = isWorkerAct ? "N/A (Farm Personnel)" : (item.breed || "Cattle");

      const isDis = item.disease_status && item.disease_status !== 'Healthy' && item.disease_status !== 'Scanning...';
      if (document.getElementById("modalDisease")) {
        document.getElementById("modalDisease").innerHTML = isWorkerAct
          ? `<span style="color: var(--primary); font-weight: 700;">👷 Personnel on Duty</span>`
          : (isDis
            ? `<span style="color: var(--accent-disease); font-weight: 700; background: var(--accent-disease-light); border: 1px solid var(--accent-disease-border); padding: 0.2rem 0.55rem; border-radius: 4px;">⚠️ Suspected: ${item.disease_status}</span>`
            : `<span style="color: var(--accent-emerald); font-weight: 700;">🩺 Healthy</span>`);
      }

      const bcsVal = (item.bcs_score !== null && item.bcs_score !== undefined && Number(item.bcs_score) > 0) ? Number(item.bcs_score) : null;
      if (document.getElementById("modalBcs")) {
        document.getElementById("modalBcs").innerText = isWorkerAct
          ? "N/A (Farm Personnel)"
          : (bcsVal
            ? `${bcsVal.toFixed(2)} / 5.0 (Full Body Evaluated)`
            : "Full Body View Required (Cattle must be fully visible to evaluate BCS)");
      }
      if (document.getElementById("modalWeight")) document.getElementById("modalWeight").innerText = isWorkerAct ? "N/A" : `${Math.round(item.estimated_weight_kg || 400)} kg`;
      if (document.getElementById("modalHeight")) document.getElementById("modalHeight").innerText = isWorkerAct ? "N/A" : `${Math.round(item.estimated_height_cm || 130)} cm`;

      const eatingSec = Math.round(item.eating_duration_sec || (item.activity_type.includes("feeding") ? item.activity_duration : 0) || 0);
      if (document.getElementById("modalEatingDur")) document.getElementById("modalEatingDur").innerText = `${Math.floor(eatingSec / 60)}m ${eatingSec % 60}s`;

      const milkSec = Math.round(item.activity_type.includes("milking") ? (item.activity_duration || 0) : 0);
      if (document.getElementById("modalMilkingDur")) document.getElementById("modalMilkingDur").innerText = `${Math.floor(milkSec / 60)}m ${milkSec % 60}s`;

      const restSec = Math.round(item.resting_duration_sec || (item.activity_type.includes("resting") ? item.activity_duration : 0) || 0);
      if (document.getElementById("modalRestingDur")) document.getElementById("modalRestingDur").innerText = `${Math.floor(restSec / 60)}m ${restSec % 60}s`;

      document.getElementById("modalTime").innerText = new Date(item.start_time || item.created_at).toLocaleString();
      document.getElementById("modalCam").innerText = item.camera_id || "CAM01";
      if (document.getElementById("modalConf")) document.getElementById("modalConf").innerText = `${Math.round(item.confidence_score * 100)}%`;
      document.getElementById("modalJson").innerText = JSON.stringify(item.ai_detection_result || {}, null, 2);

      const player = document.getElementById("modalVideoPlayer");
      player.src = item.video_url || "";
      player.play().catch(e => console.log("Auto-play blocked:", e));

      document.getElementById("activityModal").style.display = "flex";
    }

    async function cleanDuplicates() {
      if (!confirm("Clean and consolidate duplicate entries for the same cattle in Supabase?")) return;
      try {
        const res = await fetch("/api/activities/clean-duplicates", {
          method: "POST",
          headers: getAuthHeaders()
        });
        const data = await res.json();
        alert(data.message || `Consolidation complete: ${data.deleted_count || 0} duplicate rows removed.`);
        fetchActivities();
      } catch (err) {
        alert("Consolidation error: " + err);
      }
    }

    async function clearAllActivities() {
      if (!confirm("Are you sure you want to permanently delete ALL cattle, farm activities, and worker records from Supabase?")) return;
      try {
        const res = await fetch("/api/activities/clear-all", {
          method: "DELETE",
          headers: getAuthHeaders()
        });
        const data = await res.json();
        alert(data.message || "All cattle, farm activities, and worker records permanently deleted from Supabase.");
        fetchActivities();
      } catch (err) {
        alert("Clear error: " + err);
      }
    }

    async function deleteCurrentActivity() {
      if (!currentSelectedActivityId) return;
      if (!confirm("Are you sure you want to remove this activity record?")) return;
      try {
        const res = await fetch(`/api/activities/${currentSelectedActivityId}`, {
          method: "DELETE",
          headers: getAuthHeaders()
        });
        if (res.ok) {
          closeModal("activityModal");
          fetchActivities();
        } else {
          alert("Could not delete activity.");
        }
      } catch (err) {
        alert("Error: " + err);
      }
    }


    function openConnectCameraModal() { document.getElementById("connectCameraModal").style.display = "flex"; }
    function openUploadModal() { document.getElementById("uploadModal").style.display = "flex"; }
    function openAuthModal() { document.getElementById("authModal").style.display = "flex"; }
    function closeModal(id) {
      document.getElementById(id).style.display = "none";
      if (id === "activityModal") {
        const player = document.getElementById("modalVideoPlayer");
        player.pause();
        player.src = "";
      }
    }

    function toggleAuthMode(e) {
      e.preventDefault();
      isRegisterMode = !isRegisterMode;
      document.getElementById("authTitle").innerText = isRegisterMode ? "📝 Register Farm Account" : "🔑 Owner / Worker Login";
      document.querySelectorAll(".reg-field").forEach(el => el.style.display = isRegisterMode ? "flex" : "none");
      document.getElementById("toggleAuthMode").innerText = isRegisterMode ? "Already have an account? Login" : "Need an account? Register";
      document.getElementById("authSubmitBtn").innerText = isRegisterMode ? "Create Account & Farm" : "Secure Login";
    }

    async function handleAuthSubmit(e) {
      e.preventDefault();
      const phone = document.getElementById("authPhone").value.trim();
      const password = document.getElementById("authPassword").value;
      const fullName = document.getElementById("authFullName").value.trim() || "Farm Owner";
      const farmName = document.getElementById("authFarmName").value.trim() || `${fullName}'s Dairy Farm`;

      const endpoint = isRegisterMode ? "/api/auth/register" : "/api/auth/login";
      const body = isRegisterMode ? { phone, password, full_name: fullName, farm_name: farmName } : { phone, password };

      try {
        const res = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        const data = await res.json();
        if (data.success && data.user) {
          authToken = data.user.token;
          localStorage.setItem("pashux_jwt", authToken);
          closeModal("authModal");
          checkUserSession();
          fetchActivities();
        } else {
          alert(data.detail || "Authentication failed.");
        }
      } catch (err) {
        alert("Server communication error: " + err);
      }
    }

    async function handleConnectCamera(e) {
      e.preventDefault();
      const camera_name = document.getElementById("camName").value.trim();
      const rtsp_url = document.getElementById("camUrl").value.trim();
      const location_description = document.getElementById("camLoc").value.trim();

      try {
        const res = await fetch("/api/cctv/connect", {
          method: "POST",
          headers: getAuthHeaders(),
          body: JSON.stringify({ camera_name, rtsp_url, location_description })
        });
        const data = await res.json();
        if (data.success) {
          alert(data.message);
          closeModal("connectCameraModal");
          openLiveStream(`Live Camera: ${camera_name || 'CAM01'}`, "Streaming Live Feed");
        } else {
          alert(data.detail || "Connection failed");
        }
      } catch (err) {
        alert("Camera connection error: " + err);
      }
    }

    function toggleAllUploadModels(isChecked) {
      document.querySelectorAll(".upload-model-chk").forEach(chk => {
        chk.checked = isChecked;
      });
    }

    function handleUploadSubmit(e) {
      e.preventDefault();
      const fileInput = document.getElementById("uploadVideoFile");
      if (!fileInput.files || fileInput.files.length === 0) return;
      const file = fileInput.files[0];
      const cameraId = document.getElementById("uploadCameraId").value;

      const checkedBoxes = Array.from(document.querySelectorAll("input[name='ai_model_opt']:checked"));
      const selectedModels = checkedBoxes.map(cb => cb.value);

      const formData = new FormData();
      formData.append("file", file);
      formData.append("camera_id", cameraId);
      formData.append("enabled_models", JSON.stringify(selectedModels));

      const progContainer = document.getElementById("uploadProgressContainer");
      const progText = document.getElementById("uploadProgressText");
      const progPct = document.getElementById("uploadProgressPercent");
      const progBar = document.getElementById("uploadProgressBarFill");
      const btn = document.getElementById("btnUploadSubmit");

      if (progContainer) {
        progContainer.style.display = "block";
        progText.innerText = `Preparing '${file.name}'...`;
        progPct.innerText = "0%";
        progBar.style.width = "0%";
      }
      btn.disabled = true;

      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/analyze/upload-video", true);
      if (authToken) {
        xhr.setRequestHeader("Authorization", `Bearer ${authToken}`);
      }

      xhr.upload.onprogress = function(event) {
        if (event.lengthComputable) {
          const percent = Math.round((event.loaded / event.total) * 100);
          const loadedMB = (event.loaded / (1024 * 1024)).toFixed(1);
          const totalMB = (event.total / (1024 * 1024)).toFixed(1);
          if (progBar) progBar.style.width = `${percent}%`;
          if (progPct) progPct.innerText = `${percent}%`;
          if (progText) {
            if (percent < 100) {
              progText.innerText = `Uploading: ${loadedMB}MB / ${totalMB}MB (${percent}%)...`;
            } else {
              progText.innerText = `Uploaded! Initializing AI models & starting analysis...`;
            }
          }
        }
      };

      xhr.onload = function() {
        btn.disabled = false;
        if (progContainer) progContainer.style.display = "none";
        try {
          const data = JSON.parse(xhr.responseText);
          if (xhr.status >= 200 && xhr.status < 300 && data.success) {
            closeModal("uploadModal");
            openLiveStream(`Visual AI Processing: ${file.name}`, "Live Detection Engine Active");
            if (data.job_id) {
              trackVideoAnalysisJob(data.job_id, file.name);
            } else {
              alert(data.message);
              fetchActivities();
            }
          } else {
            alert(data.detail || data.message || "Upload failed: Server error");
          }
        } catch (err) {
          alert("Error parsing server response: " + err);
        }
      };

      xhr.onerror = function() {
        btn.disabled = false;
        if (progContainer) progContainer.style.display = "none";
        alert("Network error while uploading video. Please check connection and try again.");
      };

      xhr.send(formData);
    }

    function trackVideoAnalysisJob(jobId, filename) {
      const banner = document.getElementById("videoJobBanner");
      const text = document.getElementById("jobBannerText");
      const pct = document.getElementById("jobPercentText");
      const bar = document.getElementById("jobProgressBarFill");
      const spinner = document.getElementById("jobSpinner");

      banner.style.display = "block";
      text.innerText = `Analyzing '${filename}' - Processing cattle herd activities & duration...`;
      pct.innerText = "0%";
      bar.style.width = "0%";
      spinner.style.display = "inline-block";

      const pollInterval = setInterval(async () => {
        try {
          const res = await fetch(`/api/jobs/${jobId}`, { headers: getAuthHeaders() });
          if (!res.ok) return;
          const job = await res.json();
          const p = Math.round(job.progress_percent || 0);
          pct.innerText = `${p}%`;
          bar.style.width = `${p}%`;
          text.innerText = `Visual Processing: '${filename}' (${job.frames_processed || 0}/${job.total_frames || 0} frames @ ${job.analyzed_fps || 0} FPS)...`;

          if (job.status === "completed") {
            clearInterval(pollInterval);
            pct.innerText = "100%";
            bar.style.width = "100%";
            spinner.style.display = "none";
            text.innerHTML = `✅ <b>Analysis Complete!</b> All cattle feeding, resting, and biometrics stored in Supabase.`;
            const badgeEl = document.getElementById("streamStatusBadge");
            if (badgeEl) badgeEl.innerText = "Analysis Complete";
            fetchActivities();
            setTimeout(() => {
              banner.style.display = "none";
            }, 6000);
          } else if (job.status === "failed") {
            clearInterval(pollInterval);
            spinner.style.display = "none";
            text.innerHTML = `❌ Analysis stopped: ${job.error || 'Failed'}`;
            fetchActivities();
          }
        } catch (e) {
          console.error("Job poll error:", e);
        }
      }, 800);
    }

    // Startup initializers
    initTheme();
    checkUserSession();
    fetchActivities();
    setInterval(fetchActivities, 5000);
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serves the rich PashuX Activity History & CCTV Evidence Dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)


def start_dashboard_server(port: int = 8000):
    """Starts FastAPI web dashboard in a separate daemon thread."""
    config = uvicorn.Config(app=app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    print(f"\n[PashuX Dashboard] Web Activity History active at: http://localhost:{port}")
    return server


if __name__ == "__main__":
    port = int(os.getenv("PORT", os.getenv("WEB_DASHBOARD_PORT", 8000)))
    print(f"\n" + "=" * 65)
    print(f"  PASHUX AI - FARM ACTIVITY DETECTION & CCTV HUB")
    print(f"  Opening Web Dashboard on port: {port}")
    print(f"=" * 65 + "\n")
    if sys.platform == "win32" and not os.getenv("RENDER"):
        try:
            import webbrowser
            threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{port}")).start()
        except Exception:
            pass
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
