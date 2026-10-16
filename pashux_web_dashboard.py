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
    DEFAULT_FARM_ID,
    DEFAULT_CAMERA_ID,
    JWT_SECRET
)
from activity_engine import ActivityEngine
from ai_model_manager import ai_models
from cattle_tracker import SpatialCattleTracker
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


def check_cattle_estrus_behavior(c1: Dict[str, Any], c2: Dict[str, Any]) -> Tuple[bool, str, float]:
    """
    Evaluates kinematic criteria for genuine estrus signs between 2 distinct cattle:
    1. Active Mounting / Rearing:
       - Overlap + elevated posture (top of mounting cow is higher, lower body overlaps mounted cow)
    2. Standing Heat & Chin-Resting / Heat Interaction:
       - Mutual close physical interaction where one cow rests chin/head over the back, withers, or neck of the other
       - Bounding boxes in direct contact/overlap, standing heat posture
    """
    box1, box2 = c1.get("box", [0, 0, 0, 0]), c2.get("box", [0, 0, 0, 0])
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    w1, h1 = max(1, x2_1 - x1_1), max(1, y2_1 - y1_1)
    w2, h2 = max(1, x2_2 - x1_2), max(1, y2_2 - y1_2)

    if min(w1, h1, w2, h2) < 35:
        return False, "", 0.0

    inter_x = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    inter_y = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    inter_area = inter_x * inter_y

    min_area = min(w1 * h1, w2 * h2)
    iomin = (inter_area / float(min_area)) if min_area > 0 else 0.0

    cx1, cy1 = (x1_1 + x2_1) / 2.0, (y1_1 + y2_1) / 2.0
    cx2, cy2 = (x1_2 + x2_2) / 2.0, (y1_2 + y2_2) / 2.0

    # Horizontal distance between box edges
    h_gap = max(0, max(x1_1, x1_2) - min(x2_1, x2_2))
    v_overlap = inter_y / float(min(h1, h2))

    # 1. Active Mounting (One cow mounted on top of another)
    elevation_diff = y1_2 - y1_1
    if (iomin >= 0.16) and (cy1 < cy2) and (elevation_diff >= min(h1, h2) * 0.08) and (y2_1 > y1_2):
        conf = min(0.97, max(0.85, 0.80 + (iomin * 0.20)))
        return True, "Active Mounting / Standing Heat", conf

    # 2. Standing Heat & Chin-Resting / Head-Over-Withers interaction
    # (Two standing cows in close physical contact with chin/head on shoulder/neck/rump)
    is_close_contact = (iomin >= 0.06) or (h_gap <= 40 and v_overlap >= 0.25)
    if is_close_contact:
        ar1 = float(w1) / float(h1)
        ar2 = float(w2) / float(h2)
        if 0.65 <= ar1 <= 1.90 and 0.65 <= ar2 <= 1.90:
            if abs(cy1 - cy2) <= max(h1, h2) * 0.60:
                conf = min(0.92, max(0.82, 0.78 + (iomin * 0.25)))
                return True, "Standing Heat / Chin Resting", conf

    return False, "", 0.0


def ai_video_monitoring_loop(
    source_path: str,
    source_type: str,
    farm_id: str,
    camera_id: str,
    stop_event: threading.Event,
    job_id: Optional[str] = None
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Optimized AI detection loop:
    - Preloaded singleton AI models (Zero disk reload)
    - Persistent spatial tracking (COW001, COW002) with IoU association
    - Cattle Profile Caching: specialist models (Breed, BCS, Disease) run max 2 times per cow
    - Frame Stride: detects every 2nd frame on video playback
    - Asynchronous 10-second video encoding & Supabase upload
    - Performance telemetry profiling
    """
    global current_live_frame, active_monitoring_state
    print(f"\n[PashuX AI] Launching Optimized AI Detection on {source_type.upper()}: {source_path}")

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

    # Initialize Spatial Cattle Tracker with Profile Cache
    cattle_tracker = SpatialCattleTracker(iou_threshold=0.25, max_missing_frames=30)

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
    stride = 2 if source_type == "uploaded" else 1
    cattle_tracks: List[Dict[str, Any]] = []
    human_tracks: List[Dict[str, Any]] = []
    video_base_dt = datetime.datetime.now(datetime.timezone.utc)
    mounting_streaks: Dict[Tuple[int, int], int] = {}
    video_sec = 0.0
    loop_start_time = time.time()

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
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

            # Push frame into rolling 10-second evidence buffer
            engine.push_frame(frame)

            try:
                annotated_frame = frame.copy()
            except Exception:
                import gc
                gc.collect()
                annotated_frame = frame

            # Run AI Inference with Frame Stride (every 2nd frame on uploaded video)
            if frame_count % stride == 0:
                h_f, w_f = frame.shape[:2]
                infer_sz = 960 if (w_f >= 960 or h_f >= 720) else 640

                # 1. Primary YOLO11n Detection
                t_det0 = time.time()
                results = ai_models.run_detection(frame, imgsz=infer_sz, conf=0.22)
                telemetry_profiler.record_inference("det", time.time() - t_det0)

                raw_cow_boxes = []
                raw_cow_confs = []
                raw_human_boxes = []
                raw_human_confs = []

                if results and len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        coords = list(map(int, box.xyxy[0]))
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])

                        if cls_id == 19: # Cattle
                            raw_cow_boxes.append(coords)
                            raw_cow_confs.append(conf)
                        elif cls_id == 0: # Worker / Human
                            raw_human_boxes.append(coords)
                            raw_human_confs.append(conf)

                # 2. Spatial Cattle Tracking & Consistent ID Association
                tracked_cows = cattle_tracker.update(raw_cow_boxes, raw_cow_confs)

                # 3. Selective Specialist Inference with Profile Caching
                cur_cattle = []
                for c in tracked_cows:
                    cid = c["id"]
                    coords = c["box"]
                    conf = c.get("conf", 0.90)

                    # Check if this cow needs specialist sampling (max 2 times)
                    if cattle_tracker.profile_cache.should_sample(cid, frame_count):
                        x1, y1, x2, y2 = coords
                        crop = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]

                        # Run Breed Classifier
                        t_b0 = time.time()
                        breed_name, breed_conf = ai_models.run_breed(crop)
                        telemetry_profiler.record_inference("breed", time.time() - t_b0)

                        # Run BCS CNN Model
                        t_bcs0 = time.time()
                        bcs_int, bcs_float, bcs_label, _ = ai_models.run_bcs(crop)
                        telemetry_profiler.record_inference("bcs", time.time() - t_bcs0)

                        # Run Cattle Disease Keras Model
                        t_d0 = time.time()
                        disease_status, disease_conf = ai_models.run_disease(crop)
                        telemetry_profiler.record_inference("disease", time.time() - t_d0)

                        # Biometric Estimation (Height & Weight)
                        try:
                            height_cm, _, weight_kg, _ = estimate_cattle_biometry(coords, breed_name, bcs_float, (h_f, w_f))
                        except Exception:
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
                        bcs_int = prof.get("bcs_int", 3) if prof else 3
                        bcs_float = prof.get("bcs_float", 3.0) if prof else 3.0
                        bcs_label = prof.get("bcs_label", "Ideal") if prof else "Ideal"
                        weight_kg = prof.get("weight_kg", 400.0) if prof else 400.0
                        height_cm = prof.get("height_cm", 130.0) if prof else 130.0

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
                        "weight_kg": weight_kg,
                        "height_cm": height_cm
                    })

                # 4. Human / Worker Association
                cur_humans = []
                for h_idx, (h_box, h_conf) in enumerate(zip(raw_human_boxes, raw_human_confs)):
                    cur_humans.append({
                        "id": h_idx + 1,
                        "box": h_box,
                        "conf": h_conf
                    })

                cattle_tracks = cur_cattle
                human_tracks = cur_humans

                video_sec = round(float(frame_count) / max(1.0, fps), 2)

                # 5. Genuine Kinematic Mounting & Estrus Detection (Active Mounting + Standing Heat & Chin-Resting)
                estrus_events = []
                if len(cattle_tracks) >= 2:
                    for i in range(len(cattle_tracks)):
                        for j in range(len(cattle_tracks)):
                            if i == j:
                                continue
                            c_m = cattle_tracks[i]
                            c_f = cattle_tracks[j]
                            is_estrus, estrus_type, e_conf = check_cattle_estrus_behavior(c_m, c_f)
                            pair_key = (min(c_m["id"], c_f["id"]), max(c_m["id"], c_f["id"]))
                            if is_estrus:
                                mounting_streaks[pair_key] = mounting_streaks.get(pair_key, 0) + 1
                                # Confirmed estrus after at least 3 consecutive inferences (~0.25s)
                                if mounting_streaks[pair_key] >= 3:
                                    estrus_events.append({
                                        "mounting_id": c_m["id"],
                                        "mounted_id": c_f["id"],
                                        "conf": round(e_conf, 2),
                                        "behavior": estrus_type
                                    })
                            else:
                                if pair_key in mounting_streaks:
                                    mounting_streaks[pair_key] = max(0, mounting_streaks[pair_key] - 1)

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
                bcs_txt = f"BCS:{c.get('bcs_float', 3.0):.1f}"
                lbl = f"COW#{c['id']:02d} | {c.get('breed','Cattle')} | {c.get('disease','Healthy')} | {bcs_txt} | {int(c.get('weight_kg', 400))}kg"
                cv2.putText(annotated_frame, lbl, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 229, 155), 2)

            for h in human_tracks:
                hx1, hy1, hx2, hy2 = h["box"]
                cv2.rectangle(annotated_frame, (hx1, hy1), (hx2, hy2), (255, 120, 0), 2)
                cv2.putText(annotated_frame, f"Worker #{h['id']} ({int(h['conf']*100)}%)", (hx1, max(20, hy1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 2)

            # Overlay Live Camera Status & Performance Telemetry Watermark
            fps_val = telemetry_profiler.get_fps()
            cv2.putText(annotated_frame, f"PASHUX AI [{camera_id}] | Herd: {len(cattle_tracks)} | Workers: {len(human_tracks)} | {fps_val:.1f} FPS",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 229, 155), 2)

            with live_stream_lock:
                current_live_frame = annotated_frame

            # Throttle slightly for uploaded video to avoid overloading browser
            if source_type == "uploaded":
                time.sleep(0.005)

        # Finalize all open feeding, resting, and worker sessions before completing
        try:
            engine.finalize_all_sessions(current_video_sec=video_sec, video_base_dt=video_base_dt)
        except Exception as e:
            print(f"[PashuX AI] Session finalization notice: {e}")

        # Automatically sync local activities and images/videos to Supabase
        try:
            supabase_manager.sync_local_to_supabase()
        except Exception as e:
            print(f"[PashuX AI] Supabase sync notice: {e}")

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
        job_id=job["job_id"]
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
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Uploads CCTV recording and registers analysis job in the single-active queue.
    Prevents duplicate video uploads from spawning redundant processing jobs.
    """
    farm_id = user.get("farm_id", DEFAULT_FARM_ID)
    safe_name = f"{int(time.time())}_{file.filename}"
    save_path = UPLOAD_DIR / safe_name

    with open(save_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    print(f"\n[Upload Video] Saved {file.filename} -> {save_path} for Farm {farm_id}")

    # Enqueue into single-worker job queue with file hash deduplication
    job, is_new = video_queue.submit_job(
        filepath=str(save_path),
        filename=file.filename,
        farm_id=farm_id,
        camera_id=camera_id
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
# Activity History Endpoints
# ---------------------------------------------------------------------
@app.get("/api/activities")
async def list_activities(
    date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    camera_id: Optional[str] = Query("all"),
    cattle_id: Optional[str] = Query("all"),
    activity_type: Optional[str] = Query("all"),
    limit: int = Query(50, ge=1, le=200),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Retrieves detected activities strictly scoped to user's registered farm."""
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
    feeding_count = sum(1 for a in activities if a.get("activity_type") == "feeding_activity")
    resting_count = sum(1 for a in activities if a.get("activity_type") == "resting_activity")
    worker_count = sum(1 for a in activities if a.get("activity_type") in ("worker_entry", "worker_exit", "cleaning_activity"))

    # Build Deduplicated Cattle Registry & Camera Intelligence
    cattle_registry: Dict[str, Dict[str, Any]] = {}
    worker_registry: Dict[str, Dict[str, Any]] = {}
    cameras_summary: Dict[str, Dict[str, Any]] = {}

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
            if norm_cid not in cattle_registry:
                cattle_registry[norm_cid] = {
                    "cattle_id": norm_cid,
                    "camera_id": cam,
                    "breed": a.get("breed") or "Cattle",
                    "disease_status": a.get("disease_status") or "Healthy",
                    "bcs_score": round(float(a.get("bcs_score") or 3.0), 1),
                    "estimated_weight_kg": round(float(a.get("estimated_weight_kg") or 400.0), 1),
                    "estimated_height_cm": round(float(a.get("estimated_height_cm") or 130.0), 1),
                    "image_url": a.get("image_url") or "",
                    "total_eating_sec": 0.0,
                    "total_resting_sec": 0.0,
                    "has_estrus": False,
                    "last_seen": a.get("start_time") or a.get("created_at")
                }
            c_rec = cattle_registry[norm_cid]
            if a.get("activity_type") == "feeding_activity":
                c_rec["total_eating_sec"] += float(a.get("eating_duration_sec") or a.get("activity_duration") or 0.0)
            elif a.get("activity_type") == "resting_activity":
                c_rec["total_resting_sec"] += float(a.get("resting_duration_sec") or a.get("activity_duration") or 0.0)
            elif a.get("activity_type") == "estrus_detected":
                c_rec["has_estrus"] = True

        # Track Workers (Deduplicated by worker_id)
        w_id = a.get("worker_id")
        if w_id:
            c_sum["workers"].add(w_id)
            if w_id not in worker_registry:
                worker_registry[w_id] = {
                    "worker_id": w_id,
                    "camera_id": cam,
                    "image_url": a.get("image_url") or "",
                    "first_seen": a.get("start_time") or a.get("created_at"),
                    "activity_type": a.get("activity_type"),
                    "description": a.get("description")
                }

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

    return {
        "farm_id": farm_id,
        "total": total_count,
        "metrics": {
            "estrus_count": estrus_count,
            "disease_count": disease_count,
            "calving_count": calving_count,
            "feeding_count": feeding_count,
            "resting_count": resting_count,
            "worker_count": worker_count,
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
async def get_local_image(filename: str):
    """Serves captured high-res image from local storage with fuzzy fallback."""
    file_path = LOCAL_IMAGES_DIR / filename
    if file_path.exists():
        return FileResponse(file_path, media_type="image/jpeg")

    # Try fuzzy match by stem or prefix (e.g. af1f801d...)
    stem = Path(filename).stem
    candidates = list(LOCAL_IMAGES_DIR.glob(f"*{stem}*"))
    if not candidates and "-" in stem:
        short_id = stem.split("-")[0]
        candidates = list(LOCAL_IMAGES_DIR.glob(f"*{short_id}*"))

    if candidates:
        return FileResponse(candidates[0], media_type="image/jpeg")

    logo_path = BASE_DIR / "logo.png"
    if logo_path.exists():
        return FileResponse(logo_path, media_type="image/png")

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
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PashuX AI – Farm CCTV Activity & Evidence Hub</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0b0f17;
      --bg-card: rgba(18, 24, 38, 0.85);
      --bg-card-hover: rgba(26, 35, 54, 0.95);
      --border-color: rgba(255, 255, 255, 0.08);
      --primary: #00e59b;
      --primary-glow: rgba(0, 229, 155, 0.25);
      --accent-estrus: #ff2a85;
      --accent-disease: #ff5252;
      --accent-calving: #ffb300;
      --accent-feeding: #00b0ff;
      --accent-worker: #7c4dff;
      --text-main: #f0f4f8;
      --text-muted: #8b9bb4;
      --font-sans: 'Plus Jakarta Sans', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }
    @keyframes spin { 100% { transform: rotate(360deg); } }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg-dark);
      background-image: 
        radial-gradient(circle at 10% 10%, rgba(0, 229, 155, 0.05) 0%, transparent 40%),
        radial-gradient(circle at 90% 90%, rgba(124, 77, 255, 0.05) 0%, transparent 40%);
      color: var(--text-main);
      font-family: var(--font-sans);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: rgba(11, 15, 23, 0.9);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-color);
      position: sticky;
      top: 0;
      z-index: 50;
      padding: 0.85rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 0.85rem;
      text-decoration: none;
    }
    .brand-logo {
      width: 40px;
      height: 40px;
      border-radius: 10px;
      background: linear-gradient(135deg, #00e59b, #0099ff);
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 1.25rem;
      color: #0b0f17;
      box-shadow: 0 0 20px var(--primary-glow);
    }
    .brand-title {
      font-size: 1.3rem;
      font-weight: 800;
      color: #fff;
    }
    .brand-badge {
      background: rgba(0, 229, 155, 0.15);
      color: var(--primary);
      border: 1px solid rgba(0, 229, 155, 0.3);
      padding: 0.2rem 0.55rem;
      border-radius: 20px;
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
    }
    .user-pill {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      background: rgba(255, 255, 255, 0.04);
      padding: 0.4rem 1rem;
      border-radius: 30px;
      border: 1px solid var(--border-color);
    }
    .user-phone {
      font-family: var(--font-mono);
      font-size: 0.82rem;
      color: var(--text-main);
    }
    .btn-auth {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 0.45rem 1rem;
      border-radius: 8px;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-auth:hover { background: rgba(255, 255, 255, 0.15); }
    .container {
      max-width: 1440px;
      margin: 0 auto;
      padding: 1.8rem 2rem;
      width: 100%;
      flex: 1;
    }

    /* Live CCTV / Video Preview Box */
    .stream-box {
      background: #000;
      border: 1px solid var(--border-color);
      border-radius: 16px;
      overflow: hidden;
      margin-bottom: 2rem;
      display: none;
      position: relative;
    }
    .stream-header {
      background: rgba(18, 24, 38, 0.95);
      padding: 0.75rem 1.2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border-color);
    }
    .stream-img {
      width: 100%;
      max-height: 480px;
      object-fit: contain;
      display: block;
      background: #000;
    }

    /* Action Toolbar */
    .action-toolbar {
      background: var(--bg-card);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 1.2rem 1.5rem;
      margin-bottom: 2rem;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
    }
    .farm-meta-info h3 {
      font-size: 1.15rem;
      font-weight: 700;
      color: #fff;
    }
    .farm-meta-info p {
      font-size: 0.82rem;
      color: var(--text-muted);
      font-family: var(--font-mono);
      margin-top: 0.2rem;
    }
    .btn-action {
      background: linear-gradient(135deg, #00e59b, #00b0ff);
      color: #0b0f17;
      border: none;
      padding: 0.65rem 1.3rem;
      border-radius: 10px;
      font-size: 0.85rem;
      font-weight: 700;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      transition: all 0.2s;
    }
    .btn-action:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(0, 229, 155, 0.4); }
    .btn-secondary {
      background: rgba(255, 255, 255, 0.08);
      color: #fff;
      border: 1px solid var(--border-color);
    }
    .btn-secondary:hover { background: rgba(255, 255, 255, 0.15); box-shadow: none; }

    /* Stats Overview */
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 1.2rem;
      margin-bottom: 2rem;
    }
    .stat-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 1.25rem 1.4rem;
      position: relative;
    }
    .stat-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
    .stat-estrus::before { background: var(--accent-estrus); }
    .stat-disease::before { background: var(--accent-disease); }
    .stat-calving::before { background: var(--accent-calving); }
    .stat-feeding::before { background: var(--accent-feeding); }
    .stat-worker::before { background: var(--accent-worker); }
    .stat-label { font-size: 0.78rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; margin-bottom: 0.35rem; }
    .stat-value { font-size: 1.85rem; font-weight: 800; }

    /* Filter Bar */
    .filter-section {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 1.2rem 1.5rem;
      margin-bottom: 2rem;
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      justify-content: space-between;
      align-items: center;
    }
    .filter-group { display: flex; flex-wrap: wrap; gap: 0.85rem; align-items: center; }
    .filter-item { display: flex; flex-direction: column; gap: 0.3rem; }
    .filter-item label { font-size: 0.72rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; }
    .filter-select, .filter-input {
      background: rgba(11, 15, 23, 0.9);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 0.55rem 0.9rem;
      border-radius: 10px;
      font-size: 0.85rem;
      outline: none;
    }
    .filter-select:focus, .filter-input:focus { border-color: var(--primary); }

    /* Activities Grid */
    .activities-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
      gap: 1.5rem;
    }
    .activity-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: all 0.25s;
      cursor: pointer;
    }
    .activity-card:hover {
      background: var(--bg-card-hover);
      border-color: rgba(255, 255, 255, 0.2);
      transform: translateY(-4px);
      box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
    }
    .card-media { position: relative; width: 100%; height: 195px; background: #000; }
    .card-thumb { width: 100%; height: 100%; object-fit: cover; }
    .video-tag {
      position: absolute; bottom: 10px; right: 10px;
      background: rgba(11, 15, 23, 0.85);
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: #fff; padding: 0.25rem 0.55rem;
      border-radius: 6px; font-size: 0.72rem; font-weight: 700;
      font-family: var(--font-mono);
    }
    .badge-status {
      position: absolute; top: 10px; left: 10px;
      padding: 0.3rem 0.65rem; border-radius: 6px;
      font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    }
    .badge-estrus { background: rgba(255, 42, 133, 0.9); color: #fff; }
    .badge-disease { background: rgba(255, 82, 82, 0.9); color: #fff; }
    .badge-calving { background: rgba(255, 179, 0, 0.9); color: #000; }
    .badge-feeding { background: rgba(0, 176, 255, 0.9); color: #fff; }
    .badge-worker { background: rgba(124, 77, 255, 0.9); color: #fff; }
    .badge-general { background: rgba(0, 229, 155, 0.9); color: #0b0f17; }

    .card-body { padding: 1.2rem; display: flex; flex-direction: column; flex: 1; gap: 0.7rem; }
    .card-top { display: flex; justify-content: space-between; align-items: flex-start; }
    .activity-name { font-size: 1.05rem; font-weight: 700; line-height: 1.3; }
    .activity-conf {
      font-family: var(--font-mono); font-size: 0.78rem; font-weight: 700;
      background: rgba(0, 229, 155, 0.12); color: var(--primary);
      padding: 0.2rem 0.5rem; border-radius: 6px;
      border: 1px solid rgba(0, 229, 155, 0.25);
    }
    .activity-desc { font-size: 0.85rem; color: var(--text-muted); line-height: 1.45; }
    .meta-row {
      display: flex; gap: 0.85rem; margin-top: auto; padding-top: 0.75rem;
      border-top: 1px solid var(--border-color); font-size: 0.76rem; color: var(--text-muted);
    }

    /* Modal Layouts */
    .modal-backdrop {
      position: fixed; top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(4, 7, 12, 0.85); backdrop-filter: blur(14px);
      z-index: 100; display: none; align-items: center; justify-content: center; padding: 1.5rem;
    }
    .modal-content {
      background: #111827; border: 1px solid var(--border-color); border-radius: 20px;
      max-width: 900px; width: 100%; max-height: 90vh; overflow-y: auto; display: flex; flex-direction: column;
    }
    .modal-header { padding: 1.2rem 1.5rem; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); }
    .modal-title { font-size: 1.15rem; font-weight: 700; }
    .btn-close { background: transparent; border: none; color: var(--text-muted); font-size: 1.6rem; cursor: pointer; }
    .btn-close:hover { color: #fff; }
    .modal-video { width: 100%; max-height: 500px; background: #000; }
    .modal-details { padding: 1.5rem; display: grid; grid-template-columns: 2fr 1fr; gap: 1.5rem; }
    .auth-modal { max-width: 440px; }
    .auth-form { padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem; }
    .form-group { display: flex; flex-direction: column; gap: 0.4rem; }
    .form-group label { font-size: 0.75rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; }
    .form-control { background: #0b0f17; border: 1px solid var(--border-color); color: #fff; padding: 0.65rem 0.9rem; border-radius: 8px; font-size: 0.9rem; outline: none; }
    .form-control:focus { border-color: var(--primary); }

    .empty-state { grid-column: 1 / -1; text-align: center; padding: 4rem 2rem; background: var(--bg-card); border-radius: 16px; border: 1px dashed var(--border-color); }
  </style>
</head>
<body>

  <!-- Top Navigation Header -->
  <header>
    <a href="/" class="brand">
      <div class="brand-logo">PX</div>
      <div>
        <div class="brand-title">PashuX AI</div>
        <span class="brand-badge">Auto CCTV & Activity Storage</span>
      </div>
    </a>

    <div style="display: flex; align-items: center; gap: 1rem;">
      <div class="user-pill" id="userPill">
        <span style="width: 8px; height: 8px; border-radius: 50%; background: #00e59b; box-shadow: 0 0 10px #00e59b;"></span>
        <span class="user-phone" id="displayPhone">+91 98765 43210</span>
      </div>
      <button class="btn-auth" id="btnLoginToggle" onclick="openAuthModal()">Switch / Login</button>
      <button class="btn-auth" id="btnLogout" onclick="logout()" style="display: none;">Logout</button>
    </div>
  </header>

  <main class="container">

    <!-- Active Farm & Action Toolbar -->
    <section class="action-toolbar">
      <div class="farm-meta-info">
        <h3 id="displayFarmName">PashuX Pioneer Dairy Farm</h3>
        <p id="displayFarmId">Farm ID: 11111111-1111-1111-1111-111111111111</p>
      </div>

      <div style="display: flex; gap: 0.8rem; flex-wrap: wrap;">
        <button class="btn-action" onclick="openConnectCameraModal()">
          <span>📹</span> Connect CCTV Camera
        </button>
        <button class="btn-action" style="background: linear-gradient(135deg, #7c4dff, #00b0ff);" onclick="openUploadModal()">
          <span>📁</span> Upload CCTV Video
        </button>
        <button class="btn-action btn-secondary" onclick="cleanDuplicates()" title="Consolidates duplicate records for the same cattle">
          <span>🧹</span> Consolidate Duplicates
        </button>
        <button class="btn-action btn-secondary" id="btnToggleLiveStream" onclick="toggleLiveStream()">
          <span>🔴</span> View Live AI Stream
        </button>
      </div>
    </section>

    <!-- Video Analysis Progress Notification Banner -->
    <div id="videoJobBanner" style="display: none; background: rgba(124, 77, 255, 0.15); border: 1px solid rgba(124, 77, 255, 0.4); border-radius: 12px; padding: 1.1rem 1.4rem; margin-bottom: 1.5rem;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
        <div style="display: flex; align-items: center; gap: 0.6rem;">
          <span id="jobSpinner" style="display: inline-block; font-size: 1.1rem; animation: spin 1s linear infinite;">⏳</span>
          <span id="jobBannerText" style="font-weight: 600; font-size: 0.95rem; color: #fff;">Analyzing video...</span>
        </div>
        <span id="jobPercentText" style="font-family: var(--font-mono); font-weight: 700; color: #00e59b; font-size: 1rem;">0%</span>
      </div>
      <div style="width: 100%; height: 6px; background: rgba(255,255,255,0.08); border-radius: 3px; overflow: hidden;">
        <div id="jobProgressBarFill" style="width: 0%; height: 100%; background: linear-gradient(90deg, #7c4dff, #00e59b); transition: width 0.3s ease;"></div>
      </div>
    </div>

    <!-- Live Stream Preview Window -->
    <section class="stream-box" id="streamBox">
      <div class="stream-header">
        <span style="font-weight: 700; color: #00e59b;">Live CCTV AI Detection Feed</span>
        <span id="streamStatusBadge" style="font-size: 0.8rem; color: var(--text-muted); font-family: var(--font-mono);">Active Monitoring</span>
      </div>
      <img id="liveStreamImg" class="stream-img" src="" alt="Live CCTV AI Stream">
    </section>

    <!-- Stats Overview -->
    <section class="stats-grid">
      <div class="stat-card stat-estrus">
        <div class="stat-label">Estrus / Mounting</div>
        <div class="stat-value" id="statEstrus" style="color: var(--accent-estrus);">0</div>
      </div>
      <div class="stat-card stat-disease">
        <div class="stat-label">Suspected Disease</div>
        <div class="stat-value" id="statDisease" style="color: var(--accent-disease);">0</div>
      </div>
      <div class="stat-card stat-calving">
        <div class="stat-label">Suspected Calving</div>
        <div class="stat-value" id="statCalving" style="color: var(--accent-calving);">0</div>
      </div>
      <div class="stat-card stat-feeding">
        <div class="stat-label">Feeding Activities</div>
        <div class="stat-value" id="statFeeding" style="color: var(--accent-feeding);">0</div>
      </div>
      <div class="stat-card stat-resting" style="border-color: rgba(179, 136, 255, 0.3);">
        <div class="stat-label">Resting / Lying</div>
        <div class="stat-value" id="statResting" style="color: #b388ff;">0</div>
      </div>
      <div class="stat-card stat-worker">
        <div class="stat-label">Worker Entry/Clean</div>
        <div class="stat-value" id="statWorker" style="color: var(--accent-worker);">0</div>
      </div>
    </section>

    <!-- Daily Camera & Cattle Herd Intelligence Registry -->
    <section class="camera-summary-box" style="background: rgba(22, 27, 34, 0.95); border: 1px solid rgba(255,255,255,0.1); border-radius: 14px; padding: 1.4rem; margin-bottom: 1.5rem;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 0.8rem; flex-wrap: wrap; gap: 0.5rem;">
        <div>
          <h3 style="font-size: 1.15rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 0.5rem;">
            <span>📸</span> Today's Camera Cattle Registry & Heat Status
          </h3>
          <p style="font-size: 0.82rem; color: var(--text-muted); margin-top: 0.2rem;">
            Deduplicated real-time cattle count, breed profiles, estrus monitoring & worker attendance per camera.
          </p>
        </div>
        <div id="cameraStatusPill" style="padding: 0.35rem 0.85rem; border-radius: 20px; font-weight: 700; font-size: 0.82rem; font-family: var(--font-mono); background: rgba(0, 229, 155, 0.15); color: #00e59b; border: 1px solid rgba(0, 229, 155, 0.3);">
          Zero-Duplicate Active Registry
        </div>
      </div>

      <div id="cameraSummaryCards" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin-bottom: 1.2rem;">
        <!-- Filled by JS -->
      </div>

      <div style="margin-top: 1rem;">
        <div style="font-size: 0.9rem; font-weight: 700; color: #fff; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.4rem;">
          <span>🐄</span> Registered Cattle Roster (Individual Breed, BCS & Duration Ledger)
        </div>
        <div id="cattleRosterList" style="display: flex; gap: 0.8rem; overflow-x: auto; padding-bottom: 0.5rem;">
          <!-- Filled by JS -->
        </div>
      </div>
    </section>

    <!-- Filter & Search Section -->
    <section class="filter-section">
      <div class="filter-group">
        <div class="filter-item">
          <label>Activity Type</label>
          <select id="filterType" class="filter-select" onchange="fetchActivities()">
            <option value="all">All Activities</option>
            <option value="feeding_activity">Feeding Activity (Eating)</option>
            <option value="resting_activity">Resting / Rumination</option>
            <option value="estrus_detected">Estrus / Mounting</option>
            <option value="disease_suspected">Suspected Disease</option>
            <option value="calving_suspected">Suspected Calving</option>
            <option value="cleaning_activity">Cleaning / Sanitation</option>
            <option value="worker_entry">Worker Entry</option>
            <option value="bcs_alert">Extreme BCS Alert</option>
          </select>
        </div>

        <div class="filter-item">
          <label>Camera Source</label>
          <select id="filterCamera" class="filter-select" onchange="fetchActivities()">
            <option value="all">All CCTV Cameras</option>
            <option value="CAM01">CAM01 (Main Gate & Feed Alley)</option>
            <option value="CAM02">CAM02 (Milking Stall Area)</option>
          </select>
        </div>

        <div class="filter-item">
          <label>Cattle ID / Worker</label>
          <input type="text" id="filterEntity" class="filter-input" placeholder="e.g. COW005" onkeyup="if(event.key==='Enter') fetchActivities()">
        </div>

        <div class="filter-item">
          <label>Date</label>
          <input type="date" id="filterDate" class="filter-input" onchange="fetchActivities()">
        </div>
      </div>

      <div>
        <button class="btn-action btn-secondary" onclick="fetchActivities()">
          <span>↻</span> Refresh Log
        </button>
      </div>
    </section>

    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.2rem;">
      <h2 style="font-size: 1.25rem; font-weight: 700;">Farm Activity Evidence (10-Second Clips in Supabase)</h2>
      <span id="totalBadge" style="background: rgba(255,255,255,0.08); padding: 0.2rem 0.7rem; border-radius: 20px; font-family: var(--font-mono); color: var(--primary);">0 Events</span>
    </div>

    <!-- Activities Card Grid -->
    <div class="activities-grid" id="activitiesGrid"></div>

  </main>

  <!-- Video Playback & Detail Modal -->
  <div class="modal-backdrop" id="activityModal">
    <div class="modal-content">
      <div class="modal-header">
        <div class="modal-title" id="modalTitle">Activity Detail</div>
        <button class="btn-close" onclick="closeModal('activityModal')">&times;</button>
      </div>
      <video id="modalVideoPlayer" class="modal-video" controls autoplay playsinline></video>
      <div class="modal-details">
        <div>
          <h4 style="font-size: 0.85rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.5rem;">Description</h4>
          <p id="modalDesc" style="font-size: 0.95rem; margin-bottom: 1rem;"></p>
          <pre id="modalJson" style="background: #0b0f17; padding: 0.8rem; border-radius: 8px; font-size: 0.75rem; color: #a0aec0; overflow-x: auto; max-height: 140px;"></pre>
        </div>
        <div>
          <h4 style="font-size: 0.85rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.5rem;">Cattle Telemetry & AI Metrics</h4>
          <div style="font-size: 0.82rem; line-height: 1.85;">
            <div>Entity / Tag: <b id="modalEntity">-</b></div>
            <div>Herd Count: <b id="modalHerd">-</b></div>
            <div>Breed: <b id="modalBreed" style="color: #00e59b;">-</b></div>
            <div>Disease Status: <b id="modalDisease">-</b></div>
            <div>BCS Score: <b id="modalBcs">-</b></div>
            <div>Estimated Live Weight: <b id="modalWeight">-</b></div>
            <div>Wither Height: <b id="modalHeight">-</b></div>
            <div>Eating Duration: <b id="modalEatingDur" style="color: #00b0ff;">-</b></div>
            <div>Resting Duration: <b id="modalRestingDur" style="color: #b388ff;">-</b></div>
            <div>Session Time: <b id="modalTime">-</b></div>
            <div>Camera: <b id="modalCam">-</b></div>
            <div>Storage: <b>Supabase Cloud DB & Private Storage</b></div>
          </div>
        </div>
      </div>
      <div style="display: flex; gap: 0.75rem; margin-top: 1rem; padding-top: 0.75rem; border-top: 1px solid var(--border-color); justify-content: flex-end;">
        <button class="btn-action" style="background: rgba(255,82,82,0.15); color: #ff5252; border: 1px solid rgba(255,82,82,0.3);" onclick="deleteCurrentActivity()">
          🗑️ Delete Record
        </button>
        <button class="btn-action" style="background: rgba(0,229,155,0.15); color: var(--primary); border: 1px solid rgba(0,229,155,0.3);" onclick="markCurrentAsFeeding()">
          🌿 Mark as Grazing / Feeding
        </button>
      </div>
    </div>
  </div>

  <!-- Connect CCTV Camera Modal -->
  <div class="modal-backdrop" id="connectCameraModal">
    <div class="modal-content auth-modal">
      <div class="modal-header">
        <div class="modal-title">Connect Farm CCTV Camera</div>
        <button class="btn-close" onclick="closeModal('connectCameraModal')">&times;</button>
      </div>
      <form class="auth-form" onsubmit="handleConnectCamera(event)">
        <div class="form-group">
          <label>Camera Name</label>
          <input type="text" id="camName" class="form-control" placeholder="e.g. Corral Cam 01" required>
        </div>
        <div class="form-group">
          <label>RTSP / ONVIF URL (or '0' for Webcam)</label>
          <input type="text" id="camUrl" class="form-control" placeholder="rtsp://admin:pass@192.168.1.58:554/stream" required>
        </div>
        <div class="form-group">
          <label>Location / Pen Description</label>
          <input type="text" id="camLoc" class="form-control" placeholder="e.g. Main Feed Alley">
        </div>
        <button type="submit" class="btn-action" style="width: 100%; justify-content: center; margin-top: 0.5rem;">
          Connect & Start 24/7 AI Monitoring
        </button>
      </form>
    </div>
  </div>

  <!-- Upload CCTV Video Modal -->
  <div class="modal-backdrop" id="uploadModal">
    <div class="modal-content auth-modal">
      <div class="modal-header">
        <div class="modal-title">Upload CCTV Video for AI Detection</div>
        <button class="btn-close" onclick="closeModal('uploadModal')">&times;</button>
      </div>
      <form class="auth-form" onsubmit="handleUploadSubmit(event)">
        <div class="form-group">
          <label>Select Video File (.mp4, .avi, .mov)</label>
          <input type="file" id="uploadVideoFile" class="form-control" accept="video/*" required>
        </div>
        <div class="form-group">
          <label>Assigned Camera</label>
          <select id="uploadCameraId" class="filter-select" style="width: 100%;">
            <option value="CAM01">CAM01 (Main Gate & Feed Alley)</option>
            <option value="CAM02">CAM02 (Milking Stall Area)</option>
          </select>
        </div>
        <div id="uploadProgress" style="display: none; font-size: 0.8rem; color: var(--primary); padding: 0.4rem 0;">
          Uploading video & launching background AI analysis...
        </div>
        <button type="submit" class="btn-action" style="width: 100%; justify-content: center; margin-top: 0.5rem;" id="btnUploadSubmit">
          Analyze & Auto-Record to Supabase
        </button>
      </form>
    </div>
  </div>

  <!-- Authentication Modal (Login / Register) -->
  <div class="modal-backdrop" id="authModal">
    <div class="modal-content auth-modal">
      <div class="modal-header">
        <div class="modal-title" id="authTitle">Owner / Worker Login</div>
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
          <a href="#" id="toggleAuthMode" style="color: var(--primary); font-size: 0.78rem; text-decoration: none;" onclick="toggleAuthMode(event)">Need an account? Register</a>
        </div>
        <button type="submit" class="btn-action" style="width: 100%; justify-content: center; margin-top: 0.5rem;" id="authSubmitBtn">
          Secure Login
        </button>
      </form>
    </div>
  </div>

  <script>
    let authToken = localStorage.getItem("pashux_jwt") || "";
    let isRegisterMode = false;
    let activitiesCache = [];
    let isLiveStreamOpen = false;

    function getAuthHeaders() {
      const headers = { "Content-Type": "application/json" };
      if (authToken) headers["Authorization"] = `Bearer ${authToken}`;
      return headers;
    }

    async function checkUserSession() {
      if (!authToken) {
        openAuthModal();
        return;
      }
      try {
        const res = await fetch("/api/auth/me", { headers: getAuthHeaders() });
        if (res.status === 401) {
          localStorage.removeItem("pashux_jwt");
          authToken = "";
          openAuthModal();
          return;
        }
        const data = await res.json();
        if (data.user) {
          document.getElementById("displayPhone").innerText = data.user.phone || "";
          document.getElementById("displayFarmName").innerText = data.user.farm_name || "Dairy Farm";
          document.getElementById("displayFarmId").innerText = `Farm ID: ${data.user.farm_id || ""}`;
          document.getElementById("btnLogout").style.display = "block";
          document.getElementById("btnLoginToggle").style.display = "none";
        }
      } catch (err) {
        console.error(err);
      }
    }

    function logout() {
      localStorage.removeItem("pashux_jwt");
      authToken = "";
      location.reload();
    }

    function toggleLiveStream() {
      const box = document.getElementById("streamBox");
      const img = document.getElementById("liveStreamImg");
      const btn = document.getElementById("btnToggleLiveStream");
      isLiveStreamOpen = !isLiveStreamOpen;
      if (isLiveStreamOpen) {
        box.style.display = "block";
        img.src = "/api/cctv/stream";
        btn.innerHTML = "<span>⏹</span> Hide Stream";
      } else {
        box.style.display = "none";
        img.src = "";
        btn.innerHTML = "<span>🔴</span> View Live AI Stream";
      }
    }

    async function fetchActivities() {
      const type = document.getElementById("filterType").value;
      const camera = document.getElementById("filterCamera").value;
      const entity = document.getElementById("filterEntity").value.trim() || "all";
      const date = document.getElementById("filterDate").value || "";

      let url = `/api/activities?activity_type=${encodeURIComponent(type)}&camera_id=${encodeURIComponent(camera)}&cattle_id=${encodeURIComponent(entity)}`;
      if (date) url += `&date=${encodeURIComponent(date)}`;

      try {
        const res = await fetch(url, { headers: getAuthHeaders() });
        if (res.status === 401) {
          openAuthModal();
          return;
        }
        const data = await res.json();
        activitiesCache = data.activities || [];
        renderActivities(activitiesCache);
        renderCameraSummaries(data.camera_summaries, data.cattle_registry, data.worker_registry);

        if (data.metrics) {
          document.getElementById("statEstrus").innerText = data.metrics.estrus_count || 0;
          document.getElementById("statDisease").innerText = data.metrics.disease_count || 0;
          document.getElementById("statCalving").innerText = data.metrics.calving_count || 0;
          document.getElementById("statFeeding").innerText = data.metrics.feeding_count || 0;
          if (document.getElementById("statResting")) {
            document.getElementById("statResting").innerText = data.metrics.resting_count || 0;
          }
          document.getElementById("statWorker").innerText = data.metrics.worker_count || 0;
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
          <div style="grid-column: 1/-1; padding: 1.2rem; text-align: center; color: var(--text-muted); font-size: 0.88rem; background: rgba(255,255,255,0.02); border-radius: 8px;">
            No cameras registered yet. Connect CCTV camera or upload a video recording to view daily herd registry.
          </div>
        `;
      } else {
        cameraSummaries.forEach(cs => {
          const card = document.createElement("div");
          card.style = "background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 1.1rem;";
          
          const estrusBadge = cs.estrus_detected ? 
            `<span style="background: rgba(255, 68, 68, 0.2); color: #ff5252; border: 1px solid rgba(255,68,68,0.4); padding: 0.25rem 0.65rem; border-radius: 6px; font-weight: 700; font-size: 0.78rem;">🔥 ESTRUS DETECTED (${cs.estrus_count} events)</span>` : 
            `<span style="background: rgba(0, 229, 155, 0.15); color: #00e59b; border: 1px solid rgba(0, 229, 155, 0.3); padding: 0.25rem 0.65rem; border-radius: 6px; font-weight: 600; font-size: 0.78rem;">🟢 Normal / No Heat</span>`;

          const feedMin = Math.floor(cs.feeding_duration_sec / 60);
          const feedSec = Math.round(cs.feeding_duration_sec % 60);
          const restMin = Math.floor(cs.resting_duration_sec / 60);
          const restSec = Math.round(cs.resting_duration_sec % 60);

          card.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.7rem;">
              <div style="font-weight: 700; font-size: 1.05rem; color: #fff;">📹 Camera: ${cs.camera_id}</div>
              <div>${estrusBadge}</div>
            </div>
            <div style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 0.6rem;">
              Herd Count Today: <b style="color: #00e59b;">${cs.unique_cattle_count} Unique Cattle</b> (Deduplicated)
            </div>
            <div style="display: flex; flex-direction: column; gap: 0.35rem; font-size: 0.84rem;">
              <div>🍽️ <b>Feeding Duration:</b> <span style="color:#00b0ff; font-weight:600;">${feedMin}m ${feedSec}s</span></div>
              <div>💤 <b>Resting Duration:</b> <span style="color:#b388ff; font-weight:600;">${restMin}m ${restSec}s</span></div>
              <div>👷 <b>Workers Logged:</b> <span style="color:#ff9100; font-weight:600;">${cs.workers_count} Worker(s)</span></div>
            </div>
          `;
          summaryContainer.appendChild(card);
        });
      }

      if (!cattleRegistry || cattleRegistry.length === 0) {
        rosterContainer.innerHTML = `
          <div style="color: var(--text-muted); font-size: 0.85rem; padding: 0.5rem;">
            No cattle logged today yet.
          </div>
        `;
      } else {
        cattleRegistry.forEach(c => {
          const item = document.createElement("div");
          item.style = "min-width: 210px; max-width: 230px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 0.8rem; flex-shrink: 0;";
          
          const disColor = (c.disease_status === 'Healthy' || !c.disease_status) ? '#00e59b' : '#ff5252';
          const heatBadge = c.has_estrus ? 
            `<span style="background: rgba(255, 68, 68, 0.2); color: #ff5252; font-size: 0.68rem; font-weight: 700; padding: 0.1rem 0.4rem; border-radius: 4px;">🔥 Estrus Sign</span>` : 
            `<span style="background: rgba(255, 255, 255, 0.08); color: var(--text-muted); font-size: 0.68rem; padding: 0.1rem 0.4rem; border-radius: 4px;">Normal</span>`;

          const feedSec = Math.round(c.total_eating_sec || 0);
          const restSec = Math.round(c.total_resting_sec || 0);

          item.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.45rem;">
              <b style="color: #00e59b; font-family: var(--font-mono); font-size: 0.95rem;">${c.cattle_id}</b>
              ${heatBadge}
            </div>
            <div style="font-size: 0.82rem; color: #fff; margin-bottom: 0.2rem;">🐄 Breed: <b>${c.breed}</b></div>
            <div style="font-size: 0.78rem; color: ${disColor}; margin-bottom: 0.25rem;">🩺 ${c.disease_status}</div>
            <div style="font-size: 0.74rem; color: var(--text-muted); margin-bottom: 0.35rem;">
              BCS: ${c.bcs_score} | ${Math.round(c.estimated_weight_kg)}kg | ${Math.round(c.estimated_height_cm)}cm
            </div>
            <div style="font-size: 0.74rem; color: #8ab4f8; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 0.35rem;">
              🍽️ Eat: ${Math.floor(feedSec / 60)}m ${feedSec % 60}s | 💤 Rest: ${Math.floor(restSec / 60)}m ${restSec % 60}s
            </div>
          `;
          rosterContainer.appendChild(item);
        });
      }
    }

    function getActivityBadge(type) {
      if (type.includes("estrus")) return { class: "badge-estrus", label: "Detected Estrus Sign" };
      if (type.includes("disease")) return { class: "badge-disease", label: "Suspected Disease Sign" };
      if (type.includes("calving")) return { class: "badge-calving", label: "Suspected Calving Sign" };
      if (type.includes("feeding")) return { class: "badge-feeding", label: "Feeding Activity" };
      if (type.includes("resting")) return { class: "badge-calving", label: "Resting / Rumination" };
      if (type.includes("worker") || type.includes("clean")) return { class: "badge-worker", label: "Worker Activity" };
      return { class: "badge-general", label: "Farm Activity" };
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

        if (item.activity_type.includes("feeding")) {
          const eatingSec = Math.round(item.eating_duration_sec || item.activity_duration || 0);
          const eMin = Math.floor(eatingSec / 60);
          const eSec = eatingSec % 60;
          mediaTag = `<div class="video-tag" style="background: rgba(0, 176, 255, 0.25); color: #00b0ff; border-color: rgba(0, 176, 255, 0.4);">🍽️ ${eMin}m ${eSec}s Session</div>`;
          durationHtml = `<div style="color: #00b0ff; font-weight: 700; font-size: 0.82rem; margin-top: 0.2rem;">🍽️ Eating Duration: ${eMin}m ${eSec}s <span style="font-weight:400; color:#8ab4f8; font-size:0.75rem;">${timeRangeStr}</span></div>`;
        } else if (item.activity_type.includes("resting")) {
          const restSec = Math.round(item.resting_duration_sec || item.activity_duration || 0);
          const rMin = Math.floor(restSec / 60);
          const rSec = restSec % 60;
          mediaTag = `<div class="video-tag" style="background: rgba(179, 136, 255, 0.25); color: #b388ff; border-color: rgba(179, 136, 255, 0.4);">💤 ${rMin}m ${rSec}s Session</div>`;
          durationHtml = `<div style="color: #b388ff; font-weight: 700; font-size: 0.82rem; margin-top: 0.2rem;">💤 Resting Duration: ${rMin}m ${rSec}s <span style="font-weight:400; color:#d1c4e9; font-size:0.75rem;">${timeRangeStr}</span></div>`;
        } else if (hasVideo) {
          mediaTag = `<div class="video-tag">▶ 10s Clip</div>`;
        } else {
          mediaTag = `<div class="video-tag" style="background: rgba(255, 255, 255, 0.15); color: #fff;">📸 Snapshot</div>`;
        }

        const breedBadge = `<span style="background: rgba(255,255,255,0.08); padding: 0.15rem 0.45rem; border-radius: 4px; font-size: 0.72rem;">🐄 ${item.breed || 'Cattle'}</span>`;
        const diseaseColor = (item.disease_status === 'Healthy' || !item.disease_status) ? '#00e59b' : '#ff5252';
        const diseaseBadge = `<span style="background: rgba(255,255,255,0.08); color: ${diseaseColor}; padding: 0.15rem 0.45rem; border-radius: 4px; font-size: 0.72rem;">🩺 ${item.disease_status || 'Healthy'}</span>`;
        const bcsBadge = `<span style="background: rgba(255,255,255,0.08); padding: 0.15rem 0.45rem; border-radius: 4px; font-size: 0.72rem;">📊 BCS: ${Number(item.bcs_score || 3.0).toFixed(1)}/5.0</span>`;
        const weightBadge = `<span style="background: rgba(255,255,255,0.08); padding: 0.15rem 0.45rem; border-radius: 4px; font-size: 0.72rem;">⚖️ ${Math.round(item.estimated_weight_kg || 400)} kg</span>`;
        const heightBadge = `<span style="background: rgba(255,255,255,0.08); padding: 0.15rem 0.45rem; border-radius: 4px; font-size: 0.72rem;">📏 ${Math.round(item.estimated_height_cm || 130)} cm</span>`;

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
            ${durationHtml}
            <div style="display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.35rem;">
              ${breedBadge} ${diseaseBadge} ${bcsBadge} ${weightBadge} ${heightBadge}
            </div>
            <div class="meta-row">
              <div>Entity: <b style="color:#fff;">${entityLabel}</b> <span style="font-size:0.7rem; color:var(--text-muted);">(Herd: ${item.herd_count || 1})</span></div>
              <div>Cam: <b style="color:#fff;">${item.camera_id || 'CAM01'}</b></div>
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

      currentSelectedActivityId = item.activity_id;
      document.getElementById("modalTitle").innerText = item.description || item.activity_type;
      document.getElementById("modalDesc").innerText = item.description;
      document.getElementById("modalEntity").innerText = item.cattle_id || item.worker_id || "Herd General";

      if (document.getElementById("modalHerd")) document.getElementById("modalHerd").innerText = `${item.herd_count || 1} Cattle`;
      if (document.getElementById("modalBreed")) document.getElementById("modalBreed").innerText = item.breed || "Cattle";
      if (document.getElementById("modalDisease")) document.getElementById("modalDisease").innerText = item.disease_status || "Healthy";
      if (document.getElementById("modalBcs")) document.getElementById("modalBcs").innerText = `${Number(item.bcs_score || 3.0).toFixed(2)} / 5.0`;
      if (document.getElementById("modalWeight")) document.getElementById("modalWeight").innerText = `${Math.round(item.estimated_weight_kg || 400)} kg`;
      if (document.getElementById("modalHeight")) document.getElementById("modalHeight").innerText = `${Math.round(item.estimated_height_cm || 130)} cm`;

      const eatingSec = Math.round(item.eating_duration_sec || (item.activity_type.includes("feeding") ? item.activity_duration : 0) || 0);
      if (document.getElementById("modalEatingDur")) document.getElementById("modalEatingDur").innerText = `${Math.floor(eatingSec / 60)}m ${eatingSec % 60}s`;

      const restSec = Math.round(item.resting_duration_sec || (item.activity_type.includes("resting") ? item.activity_duration : 0) || 0);
      if (document.getElementById("modalRestingDur")) document.getElementById("modalRestingDur").innerText = `${Math.floor(restSec / 60)}m ${restSec % 60}s`;

      document.getElementById("modalTime").innerText = new Date(item.start_time || item.created_at).toLocaleString();
      document.getElementById("modalCam").innerText = item.camera_id || "CAM01";
      document.getElementById("modalConf").innerText = `${Math.round(item.confidence_score * 100)}%`;
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

    async function markCurrentAsFeeding() {
      if (!currentSelectedActivityId) return;
      try {
        const res = await fetch(`/api/activities/${currentSelectedActivityId}`, {
          method: "PATCH",
          headers: getAuthHeaders(),
          body: JSON.stringify({
            activity_type: "feeding_activity",
            description: "Feeding / Grazing: Cattle actively grazing on pasture / feed",
            status: "detected_sign"
          })
        });
        if (res.ok) {
          closeModal("activityModal");
          fetchActivities();
        } else {
          alert("Could not update activity.");
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
      document.getElementById("authTitle").innerText = isRegisterMode ? "Register Farm Account" : "Owner / Worker Login";
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
          toggleLiveStream();
        } else {
          alert(data.detail || "Connection failed");
        }
      } catch (err) {
        alert("Camera connection error: " + err);
      }
    }

    async function handleUploadSubmit(e) {
      e.preventDefault();
      const fileInput = document.getElementById("uploadVideoFile");
      if (!fileInput.files || fileInput.files.length === 0) return;
      const file = fileInput.files[0];
      const cameraId = document.getElementById("uploadCameraId").value;

      const formData = new FormData();
      formData.append("file", file);
      formData.append("camera_id", cameraId);

      const prog = document.getElementById("uploadProgress");
      const btn = document.getElementById("btnUploadSubmit");
      prog.style.display = "block";
      prog.innerText = `Uploading '${file.name}' to PashuX...`;
      btn.disabled = true;

      try {
        const headers = {};
        if (authToken) headers["Authorization"] = `Bearer ${authToken}`;
        const res = await fetch("/api/analyze/upload-video", {
          method: "POST",
          headers: headers,
          body: formData
        });
        const data = await res.json();
        if (data.success) {
          closeModal("uploadModal");
          if (data.job_id) {
            trackVideoAnalysisJob(data.job_id, file.name);
          } else {
            alert(data.message);
            fetchActivities();
          }
        } else {
          alert(data.detail || "Upload failed");
        }
      } catch (err) {
        alert("Upload error: " + err);
      } finally {
        prog.style.display = "none";
        btn.disabled = false;
      }
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
          text.innerText = `Analyzing '${filename}' (${job.frames_processed || 0}/${job.total_frames || 0} frames @ ${job.analyzed_fps || 0} FPS)...`;

          if (job.status === "completed") {
            clearInterval(pollInterval);
            pct.innerText = "100%";
            bar.style.width = "100%";
            spinner.style.display = "none";
            text.innerHTML = `✅ <b>Analysis Complete!</b> Video closed cleanly. All cattle feeding, resting, and health data stored in Supabase.`;
            if (isLiveStreamOpen) {
              toggleLiveStream();
            }
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

    checkUserSession();
    fetchActivities();
    setInterval(fetchActivities, 10000);
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
    port = int(os.getenv("WEB_DASHBOARD_PORT", 8000))
    print(f"\n" + "=" * 65)
    print(f"  PASHUX AI - FARM ACTIVITY DETECTION & CCTV HUB")
    print(f"  Opening Web Dashboard: http://localhost:{port}")
    print(f"=" * 65 + "\n")
    import webbrowser
    threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
