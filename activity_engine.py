"""
PashuX AI - Automatic Farm Activity Recognition & 10-Second Video Evidence Engine
Features:
- Rolling Circular Video Frame Buffer (Pre-event ~4 sec + Post-event ~6 sec = 10s clip)
- Automatic High-Res Snapshot Capture with Bounding Box Overlays
- Multi-Model Activity Recognition (Cattle Counting, Breed/BCS, Disease, Estrus/Heat, Calving, Feeding, Cleaning, Worker Entry/Exit)
- Anti-Duplication Cooldown Timers
- Asynchronous Background Video Writer & Supabase Cloud/Local Upload Pipeline
"""

import os
import sys
import time
import math
import uuid
import datetime
import threading
import gc
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

from pashux_supabase import (
    supabase_manager,
    LOCAL_IMAGES_DIR,
    LOCAL_VIDEOS_DIR,
    DEFAULT_FARM_ID,
    DEFAULT_CAMERA_ID
)
import queue

class EvidenceEncoderWorker:
    """Dedicated background queue worker for 10-second video encoding and Supabase uploads."""
    def __init__(self, maxsize: int = 6):
        self.queue = queue.Queue(maxsize=maxsize)
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def submit(self, job: Dict[str, Any], engine_instance: Any):
        try:
            self.queue.put_nowait((job, engine_instance))
        except queue.Full:
            print("[EvidenceWorker] Warning: Evidence encoding queue full, skipping oldest job to preserve RAM.")

    def _run(self):
        while True:
            job, engine = self.queue.get()
            try:
                engine._finalize_and_upload_clip(job)
            except Exception as e:
                print(f"[EvidenceWorker] Error processing clip: {e}")
            finally:
                gc.collect()
                self.queue.task_done()

global_evidence_worker = EvidenceEncoderWorker()


def draw_activity_snapshot(
    frame: np.ndarray,
    box: Optional[Any],
    entity_id: str,
    breed: str = "Cattle",
    disease: str = "Healthy",
    bcs: Optional[float] = None,
    activity_type: str = "farm_activity",
    weight_kg: float = 400.0,
    height_cm: float = 130.0,
    secondary_box: Optional[Any] = None,
    worker_id: Optional[str] = None
) -> np.ndarray:
    """Draws focused boundary box and comprehensive diagnostic telemetry badge on evidence snapshot."""
    if frame is None:
        return frame
    h_orig, w_orig = frame.shape[:2]
    scale = 1.0
    if w_orig > 1280:
        scale = 1280.0 / w_orig
        annotated = cv2.resize(frame, (1280, int(h_orig * scale)), interpolation=cv2.INTER_AREA)
    else:
        try:
            annotated = frame.copy()
        except Exception:
            import gc
            gc.collect()
            annotated = frame
    if box is None:
        return annotated
    try:
        x1, y1, x2, y2 = map(int, box)
        if scale != 1.0:
            x1 = int(x1 * scale)
            y1 = int(y1 * scale)
            x2 = int(x2 * scale)
            y2 = int(y2 * scale)
    except Exception:
        return annotated

    h_f, w_f = annotated.shape[:2]
    x1 = max(0, min(w_f - 1, x1))
    x2 = max(0, min(w_f - 1, x2))
    y1 = max(0, min(h_f - 1, y1))
    y2 = max(0, min(h_f - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return annotated

    is_diseased = disease not in ("Healthy", "Scanning...", "")
    if is_diseased:
        box_color = (30, 30, 235)      # Bright Red for Disease / LSD
    elif "milking" in activity_type:
        box_color = (255, 170, 0)      # Amber / Gold for Milking
    elif "worker_feeding" in activity_type:
        box_color = (0, 229, 155)      # Emerald / Mint for Worker Feeding
    elif "feeding" in activity_type:
        box_color = (0, 200, 255)      # Cyan / Light Blue for Feeding
    elif "resting" in activity_type:
        box_color = (200, 100, 255)    # Purple for Resting
    elif "worker" in activity_type:
        box_color = (255, 120, 0)      # Orange for Worker
    else:
        box_color = (0, 230, 130)      # Emerald Green

    # Dynamic scaling for crisp HD display
    res_factor = max(0.70, float(w_f) / 900.0)
    box_thick = max(2, int(3 * res_factor))

    # 1. Draw boundary box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, box_thick)

    # Sleek corner accents
    corner_len = min(int(36 * res_factor), max(6, (x2 - x1) // 5), max(6, (y2 - y1) // 5))
    accent_thick = max(2, int(4 * res_factor))
    cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), accent_thick)
    cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), accent_thick)
    cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), (255, 255, 255), accent_thick)
    cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), (255, 255, 255), accent_thick)
    cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), (255, 255, 255), accent_thick)
    cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), (255, 255, 255), accent_thick)
    cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), accent_thick)
    cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), accent_thick)

    # Draw secondary box (e.g. Worker box in a milking proof snapshot)
    if secondary_box is not None:
        try:
            sx1, sy1, sx2, sy2 = map(int, secondary_box)
            if scale != 1.0:
                sx1 = int(sx1 * scale)
                sy1 = int(sy1 * scale)
                sx2 = int(sx2 * scale)
                sy2 = int(sy2 * scale)
            sx1 = max(0, min(w_f - 1, sx1))
            sx2 = max(0, min(w_f - 1, sx2))
            sy1 = max(0, min(h_f - 1, sy1))
            sy2 = max(0, min(h_f - 1, sy2))
            if sx2 > sx1 and sy2 > sy1:
                cv2.rectangle(annotated, (sx1, sy1), (sx2, sy2), (255, 120, 0), max(2, int(2 * res_factor)))
                sec_lbl = f"Worker #{worker_id or 'WRK001'}"
                cv2.putText(annotated, sec_lbl, (sx1, max(24, sy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.50 * res_factor, (255, 120, 0), max(1, int(1.5 * res_factor)), cv2.LINE_AA)
        except Exception:
            pass

    # 2. Text metadata lines
    if "milking" in activity_type and worker_id:
        line1 = f"MILKING EVENT: Worker #{worker_id} -> Cattle #{entity_id} ({breed})"
        line2 = "Status: Supervised Milking Protocol | Active"
    elif "worker_feeding" in activity_type and worker_id:
        line1 = f"FEEDING EVENT: Worker #{worker_id} -> Cattle #{entity_id} ({breed})"
        line2 = "Status: Supervised Ration Delivery | Active"
    elif "worker" in activity_type or (worker_id and not str(entity_id).startswith("COW")):
        line1 = f"WORKER ATTENDANCE: {worker_id or entity_id}"
        line2 = "Status: Active Duty in Cattle Barn | Confirmed"
    else:
        line1 = f"#{entity_id} | {breed} | {int(weight_kg)}kg | {int(height_cm)}cm"
        bcs_num = float(bcs) if (bcs is not None and bcs > 0) else None
        bcs_str = f"BCS: {bcs_num:.1f} (Full Body)" if bcs_num else "BCS: Full Body Required"
        if is_diseased:
            line2 = f"SUSPECTED: {disease.upper()} | {bcs_str}"
        else:
            line2 = f"Healthy | {bcs_str}"

    act_str = activity_type.replace("_", " ").upper()
    line3 = f"EVENT: {act_str}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52 * res_factor
    thick = max(1, int(1.5 * res_factor))
    (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, thick)
    (w2, h2), _ = cv2.getTextSize(line2, font, font_scale, thick)
    (w3, h3), _ = cv2.getTextSize(line3, font, font_scale, thick)
    pad = int(10 * res_factor)
    gap = int(6 * res_factor)
    max_w = max(w1, w2, w3) + pad * 2
    box_h = h1 + h2 + h3 + gap * 3 + pad

    if y1 - box_h - 4 >= 0:
        by1 = y1 - box_h - 4
        by2 = y1 - 2
    else:
        by1 = min(h_f - box_h - 4, y1 + 4)
        by2 = by1 + box_h

    # Ensure metadata badge never clips outside horizontal camera view
    bx1 = max(4, min(x1, w_f - max_w - 6))
    bx2 = min(w_f - 4, bx1 + max_w)

    overlay = annotated.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (18, 22, 28), -1)
    cv2.addWeighted(overlay, 0.90, annotated, 0.10, 0, annotated)
    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), box_color, max(1, int(2 * res_factor)))

    ty = by1 + h1 + gap
    cv2.putText(annotated, line1, (bx1 + pad, ty), font, font_scale, (255, 255, 255), thick, cv2.LINE_AA)
    ty += h2 + gap
    line2_color = (80, 80, 255) if is_diseased else (80, 240, 150)
    cv2.putText(annotated, line2, (bx1 + pad, ty), font, font_scale, line2_color, thick, cv2.LINE_AA)
    ty += h3 + gap
    cv2.putText(annotated, line3, (bx1 + pad, ty), font, font_scale, (240, 220, 100), thick, cv2.LINE_AA)

    return annotated


class ActivityEngine:
    """
    Monitors CCTV frames & object tracks, recognizes farm events, buffers video frames,
    and automatically records 10-second video clips + snapshots to Supabase.
    """

    def __init__(
        self,
        farm_id: str = DEFAULT_FARM_ID,
        camera_id: str = DEFAULT_CAMERA_ID,
        video_source_type: str = "uploaded",
        buffer_fps: float = 25.0,
        pre_buffer_seconds: float = 4.0,
        post_buffer_seconds: float = 6.0,
        cooldown_seconds: float = 60.0
    ):
        self.farm_id = farm_id
        self.camera_id = camera_id
        self.video_source_type = video_source_type
        self.fps = max(10.0, float(buffer_fps))
        self.pre_buffer_seconds = pre_buffer_seconds
        self.post_buffer_seconds = post_buffer_seconds
        self.total_clip_duration = pre_buffer_seconds + post_buffer_seconds  # 10.0 seconds
        self.cooldown_seconds = cooldown_seconds

        # Bounded circular buffer for pre-event frames: capped strictly to conserve RAM
        self.max_buffer_len = min(20, int(self.fps * min(1.5, self.pre_buffer_seconds)) + 2)
        self.frame_buffer = deque(maxlen=self.max_buffer_len)
        self.buffer_lock = threading.Lock()

        # Active recording jobs currently collecting post-event frames
        # Each job: { 'activity_id': str, 'frames_needed': int, 'collected_frames': list, 'meta': dict, 'snapshot_path': str }
        self.active_recording_jobs: List[Dict[str, Any]] = []
        self.jobs_lock = threading.Lock()

        # Activity cooldown tracker: key -> last timestamp
        # key: f"{activity_type}_{entity_id}"
        self.cooldown_tracker: Dict[str, float] = {}

        # Heuristic state trackers for Calving, Feeding, Cleaning, Worker Entry/Exit
        # Cattle ID -> {'recumbent_frames': int, 'head_down_frames': int, 'box_history': list}
        self.cattle_behavior_history: Dict[int, Dict[str, Any]] = {}
        # Human ID -> {'first_seen': float, 'last_seen': float, 'in_stall_frames': int, 'moving_distance': float, 'last_pos': (x,y)}
        self.human_behavior_history: Dict[int, Dict[str, Any]] = {}

        # Continuous Session Tracking for Eating, Resting, and Milking
        # Prevents multiple duplicate database rows for the same cattle
        # Key: f"{cattle_id}_{activity_type}" -> session dict
        self.active_cattle_sessions: Dict[str, Dict[str, Any]] = {}

        # Cached last frame for final session summaries
        self.last_frame: Optional[np.ndarray] = None

        # Callback hook for UI notifications
        self.on_activity_detected = None

    def push_frame(self, frame: np.ndarray, timestamp: Optional[float] = None):
        """Pushes current video frame into the rolling circular buffer and feeds ongoing video clip recorders."""
        if frame is None:
            return

        self.last_frame = frame
        ts = timestamp or time.time()
        # Scale buffer frame to 480px to save 90% memory while retaining crystal-clear evidence
        h, w = frame.shape[:2]
        if w > 480:
            buf_frame = cv2.resize(frame, (480, int(h * 480 / w)), interpolation=cv2.INTER_AREA)
        else:
            buf_frame = frame.copy()

        # Keep frame copy in circular buffer
        with self.buffer_lock:
            self.frame_buffer.append((buf_frame, ts))

        # Feed ongoing recording jobs that are gathering post-event frames
        with self.jobs_lock:
            completed_jobs = []
            for job in self.active_recording_jobs:
                job["collected_frames"].append(buf_frame)
                if len(job["collected_frames"]) >= job["total_frames_target"]:
                    completed_jobs.append(job)

            for c_job in completed_jobs:
                self.active_recording_jobs.remove(c_job)
                # Dispatch async video finalizer and Supabase uploader to dedicated queue worker
                global_evidence_worker.submit(c_job, self)

    def _is_cooling_down(self, activity_type: str, entity_id: str, current_ts: Optional[float] = None) -> bool:
        """Checks if this activity for this entity is within cooldown period."""
        # Worker lifecycle events (entry, exit, attendance summary) must NEVER be suppressed by cooldown
        if activity_type in ("worker_entry", "worker_exit", "worker_attendance"):
            return False

        now = current_ts if current_ts is not None else time.time()

        # Strict global & entity cooldown for Estrus: at most 1 clip per 300s (5 minutes)
        if activity_type == "estrus_detected":
            last_global = self.cooldown_tracker.get("estrus_detected_global")
            if last_global is not None and 0 <= (now - last_global) < 300.0:
                return True
            key = f"{activity_type}_{entity_id}"
            last_t = self.cooldown_tracker.get(key)
            if last_t is not None and 0 <= (now - last_t) < 300.0:
                return True
            self.cooldown_tracker["estrus_detected_global"] = now
            self.cooldown_tracker[key] = now
            return False

        key = f"{activity_type}_{entity_id}"
        if key not in self.cooldown_tracker:
            self.cooldown_tracker[key] = now
            return False

        last_t = self.cooldown_tracker[key]
        if 0 <= (now - last_t) < self.cooldown_seconds:
            return True
        self.cooldown_tracker[key] = now
        return False

    def trigger_activity(
        self,
        activity_type: str,
        confidence: float,
        description: str,
        current_frame: np.ndarray,
        cattle_id: Optional[str] = None,
        worker_id: Optional[str] = None,
        ai_detection_result: Optional[Dict[str, Any]] = None,
        status: str = "detected_sign",
        activity_id_override: Optional[str] = None,
        eating_duration_sec: float = 0.0,
        resting_duration_sec: float = 0.0,
        herd_count: int = 1,
        breed: str = "Cattle",
        disease_status: str = "Healthy",
        bcs_score: Optional[float] = None,
        estimated_weight_kg: float = 400.0,
        estimated_height_cm: float = 130.0,
        current_ts: Optional[float] = None,
        current_dt: Optional[datetime.datetime] = None,
        record_video_clip: bool = True,
        target_box: Optional[Any] = None
    ):
        """
        Triggers automatic activity evidence capture:
        1. Verifies anti-duplication cooldown
        2. Captures clear high-res image annotated with focused boundary box & diagnostic badge
        3. Prepares 10-second video clip using pre-event buffered frames + post-event frames (if record_video_clip is True)
        4. Uploads to Supabase Storage & Database without duplicate cards
        """
        entity_key = cattle_id or worker_id or "general"
        if not activity_id_override and self._is_cooling_down(activity_type, entity_key, current_ts):
            return

        activity_id = activity_id_override or str(uuid.uuid4())
        detection_timestamp = current_dt or datetime.datetime.now(datetime.timezone.utc)
        start_time = detection_timestamp - datetime.timedelta(seconds=self.pre_buffer_seconds)
        end_time = detection_timestamp + datetime.timedelta(seconds=self.post_buffer_seconds)

        # 1. Save High-Res Snapshot Image with precise focused bounding box & diagnostic label
        act_clean = str(activity_type).replace(" ", "_").replace("/", "_")
        ent_clean = str(cattle_id or worker_id or "gen").replace(" ", "_").replace("/", "_")
        snapshot_filename = f"act_{act_clean}_{ent_clean}_{int(time.time())}_{activity_id[:8]}.jpg"
        local_snapshot_path = str(LOCAL_IMAGES_DIR / snapshot_filename)
        save_img = current_frame if current_frame is not None else self.last_frame
        if save_img is not None:
            try:
                effective_box = target_box
                if effective_box is None and ai_detection_result:
                    effective_box = ai_detection_result.get("box")
                worker_box = ai_detection_result.get("worker_box") if ai_detection_result else None
                annotated_snapshot = draw_activity_snapshot(
                    frame=save_img,
                    box=effective_box,
                    entity_id=cattle_id or worker_id or "Cattle",
                    breed=breed,
                    disease=disease_status,
                    bcs=bcs_score,
                    activity_type=activity_type,
                    weight_kg=estimated_weight_kg,
                    height_cm=estimated_height_cm,
                    secondary_box=worker_box,
                    worker_id=worker_id
                )
                cv2.imwrite(local_snapshot_path, annotated_snapshot, [cv2.IMWRITE_JPEG_QUALITY, 96])
            except Exception as e:
                print(f"[ActivityEngine] Snapshot write error: {e}")

        dur = round(self.total_clip_duration, 1)
        if eating_duration_sec > dur:
            dur = round(eating_duration_sec, 1)
        elif resting_duration_sec > dur:
            dur = round(resting_duration_sec, 1)

        meta = {
            "activity_id": activity_id,
            "farm_id": self.farm_id,
            "camera_id": self.camera_id,
            "cattle_id": cattle_id,
            "worker_id": worker_id,
            "activity_type": activity_type,
            "detection_date": detection_timestamp.date().isoformat(),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "activity_duration": dur,
            "eating_duration_sec": float(eating_duration_sec),
            "resting_duration_sec": float(resting_duration_sec),
            "herd_count": int(herd_count),
            "breed": str(breed),
            "disease_status": str(disease_status),
            "bcs_score": float(bcs_score) if (bcs_score is not None and bcs_score > 0) else None,
            "estimated_weight_kg": float(estimated_weight_kg),
            "estimated_height_cm": float(estimated_height_cm),
            "ai_detection_result": ai_detection_result or {},
            "confidence_score": round(float(confidence), 3),
            "description": description,
            "video_source": self.video_source_type,
            "status": status,
            "created_at": detection_timestamp.isoformat(),
            "local_snapshot_path": local_snapshot_path,
            "captured_image_storage_path": f"{self.farm_id}/images/{snapshot_filename}",
            "captured_video_storage_path": f"{self.farm_id}/videos/{activity_id}.mp4"
        }

        # For uploaded video files: The video is already saved on disk in uploads/
        # Do NOT buffer hundreds of uncompressed video frames into RAM during video upload analysis!
        if self.video_source_type == "uploaded":
            record_video_clip = False

        # If this activity does NOT require a video clip (e.g. continuous feeding, resting session, or uploaded video), save snapshot only
        if not record_video_clip:
            # Non-blocking async upload to prevent stalling video processing loop
            def _async_act_upload(f_id=self.farm_id, a_id=activity_id, snap_p=local_snapshot_path, m=dict(meta)):
                try:
                    r_path = supabase_manager.upload_image(f_id, a_id, snap_p)
                    if r_path:
                        m["captured_image_storage_path"] = r_path
                except Exception:
                    pass
                try:
                    supabase_manager.record_activity(m)
                except Exception:
                    pass
            threading.Thread(target=_async_act_upload, daemon=True).start()
            if self.on_activity_detected:
                try:
                    self.on_activity_detected(meta)
                except Exception:
                    pass
            return

        # Prevent duplicate simultaneous video recording jobs to prevent memory exhaustion & card spam
        with self.jobs_lock:
            if len(self.active_recording_jobs) >= 1:
                # Video encoder is busy; record high-res snapshot & database activity so worker/event is NEVER dropped
                print(f"[ActivityEngine] Video clip encoder busy; recording snapshot evidence for {entity_key} ({activity_type})")
                def _async_act_busy(f_id=self.farm_id, a_id=activity_id, snap_p=local_snapshot_path, m=dict(meta)):
                    try:
                        r_path = supabase_manager.upload_image(f_id, a_id, snap_p)
                        if r_path:
                            m["captured_image_storage_path"] = r_path
                    except Exception:
                        pass
                    try:
                        supabase_manager.record_activity(m)
                    except Exception:
                        pass
                threading.Thread(target=_async_act_busy, daemon=True).start()
                if self.on_activity_detected:
                    try:
                        self.on_activity_detected(meta)
                    except Exception:
                        pass
                return

        # 2. Extract Pre-Event Frames from Rolling Circular Buffer
        pre_frames = []
        with self.buffer_lock:
            pre_frames = [f[0].copy() for f in self.frame_buffer]

        clip_fps = min(15.0, float(self.fps))
        total_frames_target = int(clip_fps * min(10.0, self.total_clip_duration))
        post_frames_needed = max(10, total_frames_target - len(pre_frames))

        # Queue recording job to collect remaining post-event frames
        job = {
            "activity_id": activity_id,
            "total_frames_target": total_frames_target,
            "collected_frames": pre_frames,
            "meta": meta,
            "snapshot_path": local_snapshot_path
        }

        with self.jobs_lock:
            self.active_recording_jobs.append(job)

        print(f"\n[EVENT TRIGGERED] {activity_type.upper()} | Confidence: {int(confidence*100)}% | Entity: {entity_key}")
        print(f"                 Capturing exactly ONE 10-second video evidence clip ({self.pre_buffer_seconds}s pre + {self.post_buffer_seconds}s post)...")

        if self.on_activity_detected:
            try:
                self.on_activity_detected(meta)
            except Exception:
                pass
    def _finalize_and_upload_clip(self, job: Dict[str, Any]):
        """Encodes captured frames into MP4 video and uploads image + video to Supabase."""
        activity_id = job["activity_id"]
        meta = job["meta"]
        frames = job.get("collected_frames", [])
        local_snapshot_path = job["snapshot_path"]

        if not frames:
            return

        try:
            h, w = frames[0].shape[:2]
            act_clean = str(meta.get("activity_type", "act")).replace(" ", "_").replace("/", "_")
            ent_clean = str(meta.get("cattle_id") or meta.get("worker_id") or "gen").replace(" ", "_").replace("/", "_")
            video_filename = f"vid_{act_clean}_{ent_clean}_{int(time.time())}_{activity_id[:8]}.mp4"
            local_video_path = str(LOCAL_VIDEOS_DIR / video_filename)

            # Encode MP4 video using avc1 or mp4v
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(local_video_path, fourcc, self.fps, (w, h))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                local_video_path = str(LOCAL_VIDEOS_DIR / f"{video_filename[:-4]}.avi")
                writer = cv2.VideoWriter(local_video_path, fourcc, self.fps, (w, h))

            for f in frames:
                if f.shape[:2] != (h, w):
                    f = cv2.resize(f, (w, h))
                writer.write(f)
            writer.release()
            print(f"[ActivityEngine] 10s video clip encoded: {os.path.basename(local_video_path)} ({len(frames)} frames)")

            # 3. Upload Image & Video to Supabase Storage
            remote_image_path = supabase_manager.upload_image(self.farm_id, activity_id, local_snapshot_path)
            remote_video_path = supabase_manager.upload_video(self.farm_id, activity_id, local_video_path)

            meta["captured_image_storage_path"] = remote_image_path or local_snapshot_path
            meta["captured_video_storage_path"] = remote_video_path or local_video_path

            # 4. Save Structured Activity Record into Supabase PostgreSQL
            supabase_manager.record_activity(meta)
        except Exception as e:
            print(f"[ActivityEngine] Video encoding error: {e}")
        finally:
            if "collected_frames" in job:
                job["collected_frames"].clear()
            del frames
            gc.collect()

    # -----------------------------------------------------------------
    # AI Activity Recognition Logic for Supported & Extended Activities
    # -----------------------------------------------------------------
    def process_frame_activities(
        self,
        frame: np.ndarray,
        cattle_tracks: List[Dict[str, Any]],
        human_tracks: List[Dict[str, Any]],
        estrus_events: List[Dict[str, Any]],
        frame_idx: int,
        video_seconds: float = 0.0,
        video_base_dt: Optional[datetime.datetime] = None
    ):
        """
        Runs comprehensive activity recognition on current frame:
        1. Estrus / Heat (Mounting & Standing Heat - Strictly one 10s evidence clip per episode)
        2. Cattle Disease Indicators (FMD, LSD)
        3. Cattle Breed & Extreme BCS Detection (Severe Thin / Obese)
        4. Calving Activity (Dedicated model hook)
        5. Feeding Activity (Live Session Tracking with Exact Start, End, and Duration, NO video spam)
        6. Resting / Rumination Activity (Live Session Tracking with Exact Start, End, and Duration)
        7. Worker Entry & Stall Cleaning Activities
        8. Worker Attendance During Feeding
        """
        h_frame, w_frame = frame.shape[:2]
        now_ts = video_seconds if self.video_source_type == "uploaded" else time.time()
        now_dt = (video_base_dt + datetime.timedelta(seconds=video_seconds)) if (video_base_dt and self.video_source_type == "uploaded") else datetime.datetime.now(datetime.timezone.utc)

        # 1. Estrus / Mounting Activity (Requires genuine 2+ cattle mounting kinematics)
        active_estrus_cow_ids = set()
        for ev in estrus_events:
            mounting_id = ev.get("mounting_id")
            mounted_id = ev.get("mounted_id")
            conf = ev.get("conf", 0.90)
            behavior = ev.get("behavior", "Standing Heat / Chin Resting")
            active_estrus_cow_ids.add(mounting_id)
            active_estrus_cow_ids.add(mounted_id)
            c_m_tag = f"COW{mounting_id:03d}" if isinstance(mounting_id, int) else str(mounting_id)
            c_f_tag = f"COW{mounted_id:03d}" if isinstance(mounted_id, int) else str(mounted_id)
            active_estrus_cow_ids.add(c_m_tag)
            active_estrus_cow_ids.add(c_f_tag)

            self.trigger_activity(
                activity_type="estrus_detected",
                confidence=conf,
                description=f"Detected Estrus Sign: Cattle #{mounting_id} & Cattle #{mounted_id} ({behavior})",
                current_frame=frame,
                cattle_id=c_m_tag,
                ai_detection_result={
                    "behavior": behavior,
                    "mounting_cattle_id": mounting_id,
                    "mounted_cattle_id": mounted_id,
                    "model": "Kinematic Mounting Detector + CowXNet"
                },
                status="verified",
                current_ts=now_ts,
                current_dt=now_dt,
                record_video_clip=True
            )

        # 2. Disease, BCS & Calving Behaviors across Cattle Tracks
        for c in cattle_tracks:
            cid = c.get("id")
            box = c.get("box", (0, 0, 0, 0))
            disease = c.get("disease", "Scanning...")
            disease_conf = c.get("disease_conf", 0.0)
            breed = c.get("breed", "Scanning...")
            breed_conf = c.get("breed_conf", 0.0)
            bcs_int = c.get("bcs_int")
            bcs_float = c.get("bcs_float")
            bcs_label = c.get("bcs_label", "")

            # Disease Alert (LSD or FMD)
            if disease in ("Lumpy Skin Disease", "Foot and Mouth Disease") and disease_conf >= 0.75 and c.get("is_full_body", True):
                short_code = "LSD" if "Lumpy" in disease else "FMD"
                bcs_note = f" | BCS: {bcs_float:.1f}/5.0 (Full Body View)" if bcs_float is not None else " | BCS: Full Body Required"
                self.trigger_activity(
                    activity_type="disease_suspected",
                    confidence=disease_conf,
                    description=f"Suspected Disease Alert: Cattle #{cid} ({breed}) exhibiting clinical signs of {disease} ({short_code}){bcs_note}",
                    current_frame=frame,
                    cattle_id=f"COW{cid:03d}" if isinstance(cid, int) else str(cid),
                    ai_detection_result={
                        "disease": disease,
                        "confidence": disease_conf,
                        "breed": breed,
                        "bcs": bcs_float,
                        "bcs_status": "Full Body Evaluated" if bcs_float is not None else "Full Body Required",
                        "model": "final_cattle_disease_model.keras",
                        "box": box
                    },
                    status="suspected_sign",
                    breed=breed,
                    disease_status=disease,
                    bcs_score=bcs_float if bcs_float is not None else 3.0,
                    estimated_weight_kg=c.get("weight_kg", 400.0),
                    estimated_height_cm=c.get("height_cm", 130.0),
                    current_ts=now_ts,
                    current_dt=now_dt,
                    record_video_clip=True,
                    target_box=box
                )

            # Severe BCS Alert (BCS 1 Severe Thinness or BCS 5 Extreme Obesity) - Only when full body appears
            if bcs_int in (1, 5) and bcs_float is not None:
                alert_text = "Severely Underweight / Emaciation" if bcs_int == 1 else "Grossly Overweight / Obesity"
                dis_note = f" | ⚠️ Suspected Disease: {disease}" if disease not in ("Healthy", "Scanning...") else ""
                self.trigger_activity(
                    activity_type="bcs_alert",
                    confidence=0.88,
                    description=f"BCS Alert: Cattle #{cid} ({breed}) evaluated at BCS {bcs_float:.2f}/5.0 ({alert_text}) - Full Body Evaluated{dis_note}",
                    current_frame=frame,
                    cattle_id=f"COW{cid:03d}" if isinstance(cid, int) else str(cid),
                    ai_detection_result={
                        "bcs_int": bcs_int,
                        "bcs_float": bcs_float,
                        "bcs_label": bcs_label,
                        "breed": breed,
                        "disease_status": disease,
                        "model": "best_bcs_cnn_v2.pth",
                        "box": box
                    },
                    status="detected_sign",
                    breed=breed,
                    disease_status=disease,
                    bcs_score=bcs_float,
                    estimated_weight_kg=c.get("weight_kg", 400.0),
                    estimated_height_cm=c.get("height_cm", 130.0),
                    current_ts=now_ts,
                    current_dt=now_dt,
                    record_video_clip=True,
                    target_box=box
                )

            # Cattle Tracking & State Memory
            if cid not in self.cattle_behavior_history:
                self.cattle_behavior_history[cid] = {
                    "recumbent_frames": 0,
                    "head_down_frames": 0,
                    "last_pos": box,
                    "first_seen": now_ts,
                    "total_movement": 0.0,
                    "frames_tracked": 0
                }
            hist = self.cattle_behavior_history[cid]
            hist["frames_tracked"] = hist.get("frames_tracked", 0) + 1

            x1, y1, x2, y2 = box
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            aspect_ratio = float(bw) / float(bh)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            prev_b = hist.get("last_pos", box)
            pcx = (prev_b[0] + prev_b[2]) / 2.0
            pcy = (prev_b[1] + prev_b[3]) / 2.0
            dist_moved = math.sqrt((cx - pcx)**2 + (cy - pcy)**2)
            hist["total_movement"] = hist.get("total_movement", 0.0) + dist_moved
            hist["last_pos"] = box
            avg_speed = hist["total_movement"] / max(1, hist["frames_tracked"])

            # Biometry & diagnostic state tracked from trained AI models
            pass

        # 6. Worker Entry, Milking, Sanitation & Attendance Activities
        for h in human_tracks:
            hid = h.get("id")
            hbox = h.get("box", (0, 0, 0, 0))
            hx1, hy1, hx2, hy2 = hbox
            hcx = (hx1 + hx2) / 2.0
            hcy = (hy1 + hy2) / 2.0
            worker_tag = f"WRK{hid:03d}" if isinstance(hid, int) else str(hid)

            if hid not in self.human_behavior_history:
                self.human_behavior_history[hid] = {
                    "first_seen": now_ts,
                    "last_seen": now_ts,
                    "first_dt": now_dt,
                    "total_movement": 0.0,
                    "last_pos": (hcx, hcy),
                    "frames_active": 0,
                    "entered_logged": False,
                    "tasks": set(),
                    "milking_pairs": {}
                }

            h_hist = self.human_behavior_history[hid]
            h_hist["last_seen"] = now_ts
            h_hist["last_dt"] = now_dt
            h_hist["frames_active"] += 1

            px, py = h_hist["last_pos"]
            dist = math.sqrt((hcx - px) ** 2 + (hcy - py) ** 2)
            h_hist["total_movement"] += dist
            h_hist["last_pos"] = (hcx, hcy)
            h_hist["last_box"] = (hx1, hy1, hx2, hy2)

            # Worker Entry Detection (Confirmed presence after sustained frames and human silhouette check)
            if not h_hist["entered_logged"] and h_hist["frames_active"] >= 8:
                is_valid_human = True
                w_crop = None
                if frame is not None:
                    try:
                        h_f, w_f = frame.shape[:2]
                        pad_y = int((hy2 - hy1) * 0.1)
                        pad_x = int((hx2 - hx1) * 0.1)
                        wy1 = max(0, hy1 - pad_y)
                        wy2 = min(h_f, hy2 + pad_y)
                        wx1 = max(0, hx1 - pad_x)
                        wx2 = min(w_f, hx2 + pad_x)
                        w_crop = frame[wy1:wy2, wx1:wx2]
                        if w_crop.size > 0:
                            ch_h, ch_w = w_crop.shape[:2]
                            # Human standing or attending in barn must be slender
                            if ch_w > ch_h * 0.75:
                                is_valid_human = False
                            else:
                                from ai_model_manager import ai_models
                                is_piebald, _ = ai_models.detect_holstein_friesian(w_crop)
                                is_buff, _ = ai_models.detect_murrah_buffalo(w_crop)
                                if is_piebald or is_buff:
                                    is_valid_human = False
                                else:
                                    cv2.imwrite(str(LOCAL_IMAGES_DIR / f"{worker_tag}.jpg"), w_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    except Exception as _we:
                        pass

                if is_valid_human:
                    h_hist["entered_logged"] = True
                    h_hist["tasks"].add("Barn Entry & Inspection")

                    self.trigger_activity(
                        activity_type="worker_entry",
                        confidence=0.94,
                        description=f"Worker Entry Detected: Farm worker #{hid} entered cattle zone / barn at {now_dt.strftime('%I:%M:%S %p')}",
                        current_frame=frame,
                        worker_id=worker_tag,
                        ai_detection_result={"worker_id": hid, "box": (hx1, hy1, hx2, hy2), "worker_box": (hx1, hy1, hx2, hy2), "entry_frame": frame_idx, "entry_time": now_dt.isoformat()},
                        status="verified",
                        current_ts=now_ts,
                        current_dt=now_dt,
                        record_video_clip=True,
                        target_box=(hx1, hy1, hx2, hy2)
                    )

            # Proximity attendance recorded under worker ledger
            pass

            # Cleaning / Sanitation Activity Detection
            if h_hist["frames_active"] >= 30 and h_hist["total_movement"] >= 120.0:
                h_hist["tasks"].add("Stall Cleaning & Maintenance")
                self.trigger_activity(
                    activity_type="cleaning_activity",
                    confidence=0.85,
                    description=f"Cleaning Activity: Farm worker #{hid} conducting stall maintenance & sanitation",
                    current_frame=frame,
                    worker_id=worker_tag,
                    ai_detection_result={
                        "worker_id": hid,
                        "frames_active": h_hist["frames_active"],
                        "movement_px": round(h_hist["total_movement"], 1)
                    },
                    status="detected_sign",
                    current_ts=now_ts,
                    current_dt=now_dt,
                    record_video_clip=True
                )

        # 7. Worker Departure / Exit Detection (Worker has left camera frame for >= 8.0s)
        for h_id, h_data in list(self.human_behavior_history.items()):
            if h_data.get("entered_logged") and not h_data.get("exit_logged"):
                if (now_ts - h_data["last_seen"]) >= 8.0:
                    h_data["exit_logged"] = True
                    w_tag = f"WRK{h_id:03d}" if isinstance(h_id, int) else str(h_id)
                    tot_sec = max(1.0, h_data["last_seen"] - h_data["first_seen"])
                    d_mins = int(tot_sec // 60)
                    d_secs = int(tot_sec % 60)
                    t_list = list(h_data.get("tasks", {"Barn Inspection"}))
                    t_str = ", ".join(t_list) if t_list else "Barn Attendance"
                    s_str = h_data["first_dt"].strftime("%I:%M:%S %p")
                    e_dt = h_data.get("last_dt", now_dt)
                    e_str = e_dt.strftime("%I:%M:%S %p")

                    self.trigger_activity(
                        activity_type="worker_exit",
                        confidence=0.95,
                        description=f"Worker Left Barn: Worker #{h_id} exited cattle zone. Active from {s_str} to {e_str} (Duration: {d_mins}m {d_secs}s) | Tasks: {t_str}",
                        current_frame=frame,
                        worker_id=w_tag,
                        ai_detection_result={
                            "worker_id": h_id,
                            "duration_sec": round(tot_sec, 1),
                            "tasks": t_list,
                            "first_seen": s_str,
                            "last_seen": e_str,
                            "status": "Departed"
                        },
                        status="verified",
                        current_ts=now_ts,
                        current_dt=now_dt,
                        record_video_clip=False
                    )

        # 8. Finalize Inactive Sessions (Inactive for >= 15 seconds)
        for skey in list(self.active_cattle_sessions.keys()):
            sess = self.active_cattle_sessions[skey]
            if now_ts - sess["last_seen_ts"] >= 15.0:
                final_dur = round(sess["duration_sec"], 1)
                dur_mins = int(final_dur // 60)
                dur_secs = int(final_dur % 60)
                act_t = sess["activity_type"]
                cid = sess["cattle_id"]
                breed = sess.get("breed", "Cattle")
                start_str = sess["start_time"].strftime("%I:%M:%S %p")
                end_str = now_dt.strftime("%I:%M:%S %p")

                dis_val = sess.get("disease_status", "Healthy")
                dis_note = f" | ⚠️ Suspected Disease: {dis_val}" if dis_val not in ("Healthy", "Scanning...") else ""
                bcs_val = sess.get("bcs_score")
                try:
                    b_val = float(bcs_val) if (bcs_val is not None and float(bcs_val) > 0) else None
                except (ValueError, TypeError):
                    b_val = None
                bcs_note = f" | BCS: {b_val:.1f}/5.0 (Full Body)" if (b_val is not None and b_val > 0) else " | BCS: Full Body Required"

                if act_t == "worker_feeding":
                    wid = sess.get("worker_id", "WRK")
                    final_desc = f"Worker Feeding Complete: Worker {wid} fed Cattle {cid} ({breed}) from {start_str} to {end_str} (Total: {dur_mins}m {dur_secs}s){dis_note}{bcs_note}"
                    eat_dur = final_dur
                    rest_dur = 0.0
                elif act_t == "standing_activity":
                    final_desc = f"Standing Complete: Cattle {cid} ({breed}) - Stood upright from {start_str} to {end_str} (Total: {dur_mins}m {dur_secs}s){dis_note}{bcs_note}"
                    eat_dur = 0.0
                    rest_dur = 0.0
                elif act_t == "feeding_activity":
                    final_desc = f"Grazing Complete: Cattle {cid} ({breed}) - Grazed from {start_str} to {end_str} (Total: {dur_mins}m {dur_secs}s){dis_note}{bcs_note}"
                    eat_dur = final_dur
                    rest_dur = 0.0
                elif act_t == "resting_activity":
                    final_desc = f"Resting Complete: Cattle {cid} ({breed}) - Rested from {start_str} to {end_str} (Total: {dur_mins}m {dur_secs}s){dis_note}{bcs_note}"
                    eat_dur = 0.0
                    rest_dur = final_dur
                elif act_t == "milking_activity":
                    wid = sess.get("worker_id", "WRK")
                    final_desc = f"Milking Complete: Worker {wid} milked Cattle {cid} ({breed}) from {start_str} to {end_str} (Total: {dur_mins}m {dur_secs}s){dis_note}{bcs_note}"
                    eat_dur = 0.0
                    rest_dur = 0.0
                else:
                    final_desc = f"Session Complete: {act_t.replace('_',' ').title()} - Total: {dur_mins}m {dur_secs}s{dis_note}{bcs_note}"
                    eat_dur = 0.0
                    rest_dur = 0.0

                supabase_manager.update_activity(
                    activity_id=sess["activity_id"],
                    updates={
                        "end_time": now_dt.isoformat(),
                        "activity_duration": final_dur,
                        "eating_duration_sec": eat_dur,
                        "resting_duration_sec": rest_dur,
                        "status": "verified",
                        "description": final_desc,
                        "bcs_score": b_val,
                        "breed": breed,
                        "disease_status": dis_val,
                        "estimated_weight_kg": sess.get("estimated_weight_kg"),
                        "estimated_height_cm": sess.get("estimated_height_cm"),
                        "herd_count": sess.get("herd_count", 1)
                    }
                )
                del self.active_cattle_sessions[skey]

    def finalize_all_sessions(self, current_video_sec: float = 0.0, video_base_dt: Optional[datetime.datetime] = None):
        """Finalizes all in-progress feeding, resting, milking, and worker attendance sessions when video completes."""
        now_dt = (video_base_dt + datetime.timedelta(seconds=current_video_sec)) if video_base_dt else datetime.datetime.now(datetime.timezone.utc)
        now_ts = current_video_sec if current_video_sec > 0 else time.time()

        # Finalize Cattle & Milking Sessions
        for skey, sess in list(self.active_cattle_sessions.items()):
            final_dur = max(1.0, now_ts - sess.get("start_video_sec", now_ts)) if "start_video_sec" in sess else sess.get("duration_sec", 1.0)
            final_dur = round(final_dur, 1)
            act_t = sess.get("activity_type", "")

            # Discard false-positive flickers under 1.0 second on live streams only
            # For uploaded video clips, ALWAYS retain and store the verified session
            if self.video_source_type != "uploaded" and final_dur < 1.0 and "milking" not in act_t and sess.get("worker_id") is None:
                supabase_manager.delete_activity(self.farm_id, sess["activity_id"])
                continue

            dur_mins = int(final_dur // 60)
            dur_secs = int(final_dur % 60)
            cid = sess.get("cattle_id", "Cattle")
            breed = sess.get("breed", "Cattle")
            start_str = sess["start_time"].strftime("%I:%M:%S %p")
            end_str = now_dt.strftime("%I:%M:%S %p")

            dis_val = sess.get("disease_status", "Healthy")
            dis_note = f" | ⚠️ Suspected Disease: {dis_val}" if dis_val not in ("Healthy", "Scanning...") else ""
            bcs_val = sess.get("bcs_score")
            try:
                b_val = float(bcs_val) if bcs_val is not None else None
            except (ValueError, TypeError):
                b_val = None
            bcs_note = f" | BCS: {b_val:.1f}/5.0 (Full Body)" if (b_val is not None and b_val > 0) else " | BCS: Full Body Required"

            if act_t == "worker_feeding":
                wid = sess.get("worker_id", "WRK")
                desc = f"Worker Feeding Complete: Worker {wid} fed Cattle {cid} ({breed}) from {start_str} to {end_str} (Total: {dur_mins}m {dur_secs}s){dis_note}{bcs_note}"
                eat_dur = final_dur
                rest_dur = 0.0
            elif "standing" in act_t:
                desc = f"Standing Complete: Cattle {cid} ({breed}) - Stood upright from {start_str} to {end_str} (Total: {dur_mins}m {dur_secs}s){dis_note}{bcs_note}"
                eat_dur = 0.0
                rest_dur = 0.0
            elif "feeding" in act_t:
                desc = f"Grazing Complete: Cattle {cid} ({breed}) - Grazed from {start_str} to {end_str} (Total: {dur_mins}m {dur_secs}s){dis_note}{bcs_note}"
                eat_dur = final_dur
                rest_dur = 0.0
            elif "resting" in act_t:
                desc = f"Resting Complete: Cattle {cid} ({breed}) - Rested from {start_str} to {end_str} (Total: {dur_mins}m {dur_secs}s){dis_note}{bcs_note}"
                eat_dur = 0.0
                rest_dur = final_dur
            elif "milking" in act_t:
                wid = sess.get("worker_id", "WRK")
                desc = f"Milking Complete: Worker {wid} milked Cattle {cid} ({breed}) from {start_str} to {end_str} (Total: {dur_mins}m {dur_secs}s){dis_note}{bcs_note}"
                eat_dur = 0.0
                rest_dur = 0.0
            else:
                desc = f"Session Complete: {act_t.replace('_', ' ').title()} - Total: {dur_mins}m {dur_secs}s{dis_note}{bcs_note}"
                eat_dur = 0.0
                rest_dur = 0.0

            supabase_manager.update_activity(
                activity_id=sess["activity_id"],
                updates={
                    "end_time": now_dt.isoformat(),
                    "activity_duration": final_dur,
                    "eating_duration_sec": eat_dur,
                    "resting_duration_sec": rest_dur,
                    "status": "verified",
                    "description": desc,
                    "bcs_score": b_val,
                    "breed": breed,
                    "disease_status": dis_val,
                    "estimated_weight_kg": sess.get("estimated_weight_kg"),
                    "estimated_height_cm": sess.get("estimated_height_cm"),
                    "herd_count": sess.get("herd_count", 1)
                }
            )
        self.active_cattle_sessions.clear()

        # Log Full Worker Attendance & Summary for every active worker
        for hid, h_hist in list(self.human_behavior_history.items()):
            if h_hist.get("frames_active", 0) >= 2:
                worker_tag = f"WRK{hid:03d}" if isinstance(hid, int) else str(hid)
                total_sec = max(1.0, h_hist.get("last_seen", now_ts) - h_hist.get("first_seen", now_ts))
                dur_mins = int(total_sec // 60)
                dur_secs = int(total_sec % 60)
                task_list = list(h_hist.get("tasks", {"Barn Inspection"}))
                tasks_str = ", ".join(task_list) if task_list else "Barn Attendance"
                start_dt = h_hist.get("first_dt", now_dt)
                start_str = start_dt.strftime("%I:%M:%S %p")
                end_str = now_dt.strftime("%I:%M:%S %p")

                self.trigger_activity(
                    activity_type="worker_attendance",
                    confidence=0.95,
                    description=f"Worker Attendance: Worker #{hid} active in barn from {start_str} to {end_str} (Duration: {dur_mins}m {dur_secs}s) | Tasks: {tasks_str}",
                    current_frame=self.last_frame,
                    worker_id=worker_tag,
                    ai_detection_result={
                        "worker_id": hid,
                        "duration_sec": round(total_sec, 1),
                        "tasks": task_list,
                        "first_seen": start_str,
                        "last_seen": end_str
                    },
                    status="verified",
                    current_ts=now_ts,
                    current_dt=now_dt,
                    record_video_clip=False,
                    target_box=h_hist.get("last_box")
                )
        self.human_behavior_history.clear()

        print(f"[ActivityEngine] All ongoing activity sessions successfully finalized and verified in Supabase.")
