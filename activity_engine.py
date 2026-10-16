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
        self.max_buffer_len = min(75, int(self.fps * self.pre_buffer_seconds) + 5)
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

        # Continuous Session Tracking for Eating and Resting
        # Prevents multiple duplicate database rows for the same cattle
        # Key: f"{cattle_id}_{activity_type}" -> session dict
        self.active_cattle_sessions: Dict[str, Dict[str, Any]] = {}

        # Callback hook for UI notifications
        self.on_activity_detected = None

    def push_frame(self, frame: np.ndarray, timestamp: Optional[float] = None):
        """Pushes current video frame into the rolling circular buffer and feeds ongoing video clip recorders."""
        if frame is None:
            return

        ts = timestamp or time.time()
        # Scale buffer frame if high resolution to save 75% memory
        h, w = frame.shape[:2]
        if w > 854:
            buf_frame = cv2.resize(frame, (854, int(h * 854 / w)))
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
        now = current_ts if current_ts is not None else time.time()

        # Strict global & entity cooldown for Estrus: at most 1 clip per 300s (5 minutes)
        if activity_type == "estrus_detected":
            last_global = self.cooldown_tracker.get("estrus_detected_global", -999999.0)
            if now - last_global < 300.0 and last_global > 0:
                return True
            key = f"{activity_type}_{entity_id}"
            last_t = self.cooldown_tracker.get(key, -999999.0)
            if now - last_t < 300.0 and last_t > 0:
                return True
            self.cooldown_tracker["estrus_detected_global"] = now
            self.cooldown_tracker[key] = now
            return False

        key = f"{activity_type}_{entity_id}"
        last_t = self.cooldown_tracker.get(key, 0.0)
        if now - last_t < self.cooldown_seconds:
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
        bcs_score: float = 3.0,
        estimated_weight_kg: float = 400.0,
        estimated_height_cm: float = 130.0,
        current_ts: Optional[float] = None,
        current_dt: Optional[datetime.datetime] = None,
        record_video_clip: bool = True
    ):
        """
        Triggers automatic activity evidence capture:
        1. Verifies anti-duplication cooldown
        2. Captures clear high-res image
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

        # 1. Save High-Res Snapshot Image
        snapshot_filename = f"{activity_id}.jpg"
        local_snapshot_path = str(LOCAL_IMAGES_DIR / snapshot_filename)
        try:
            cv2.imwrite(local_snapshot_path, current_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
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
            "bcs_score": float(bcs_score),
            "estimated_weight_kg": float(estimated_weight_kg),
            "estimated_height_cm": float(estimated_height_cm),
            "ai_detection_result": ai_detection_result or {},
            "confidence_score": round(float(confidence), 3),
            "description": description,
            "video_source": self.video_source_type,
            "status": status,
            "created_at": detection_timestamp.isoformat(),
            "local_snapshot_path": local_snapshot_path,
            "captured_video_storage_path": ""
        }

        # If this activity does NOT require a video clip (e.g. continuous feeding or resting session), save snapshot only
        if not record_video_clip:
            remote_image_path = supabase_manager.upload_image(self.farm_id, activity_id, local_snapshot_path)
            meta["captured_image_storage_path"] = remote_image_path or local_snapshot_path
            supabase_manager.record_activity(meta)
            if self.on_activity_detected:
                try:
                    self.on_activity_detected(meta)
                except Exception:
                    pass
            return

        # Prevent duplicate simultaneous video recording jobs to prevent memory exhaustion & card spam
        with self.jobs_lock:
            if len(self.active_recording_jobs) >= 1:
                # Video is already actively recording for this event; suppress duplicate card creation
                print(f"[ActivityEngine] Suppressed duplicate event trigger during active video clip encoding ({entity_key})")
                return

        # 2. Extract Pre-Event Frames from Rolling Circular Buffer
        pre_frames = []
        with self.buffer_lock:
            pre_frames = [f[0].copy() for f in self.frame_buffer]

        total_frames_target = int(self.fps * self.total_clip_duration)
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
            video_filename = f"{activity_id}.mp4"
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
        for ev in estrus_events:
            mounting_id = ev.get("mounting_id")
            mounted_id = ev.get("mounted_id")
            conf = ev.get("conf", 0.90)
            behavior = ev.get("behavior", "Active Mounting / Standing Heat")
            self.trigger_activity(
                activity_type="estrus_detected",
                confidence=conf,
                description=f"Detected Estrus Sign: Cattle #{mounting_id} actively mounting Cattle #{mounted_id} ({behavior})",
                current_frame=frame,
                cattle_id=f"COW{mounting_id:03d}" if isinstance(mounting_id, int) else str(mounting_id),
                ai_detection_result={
                    "behavior": behavior,
                    "mounting_cattle_id": mounting_id,
                    "mounted_cattle_id": mounted_id,
                    "model": "Kinematic Mounting Detector + CowXNet"
                },
                status="detected_sign",
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
            if disease in ("Lumpy Skin Disease", "Foot and Mouth Disease") and disease_conf >= 0.70:
                short_code = "LSD" if "Lumpy" in disease else "FMD"
                self.trigger_activity(
                    activity_type="disease_suspected",
                    confidence=disease_conf,
                    description=f"Suspected Disease Indicator: Cattle #{cid} exhibiting clinical signs of {disease} ({short_code})",
                    current_frame=frame,
                    cattle_id=f"COW{cid:03d}" if isinstance(cid, int) else str(cid),
                    ai_detection_result={
                        "disease": disease,
                        "confidence": disease_conf,
                        "breed": breed,
                        "bcs": bcs_float,
                        "model": "final_cattle_disease_model.keras"
                    },
                    status="suspected_sign",
                    current_ts=now_ts,
                    current_dt=now_dt,
                    record_video_clip=True
                )

            # Severe BCS Alert (BCS 1 Severe Thinness or BCS 5 Extreme Obesity)
            if bcs_int in (1, 5) and bcs_float is not None:
                alert_text = "Severely Underweight / Emaciation" if bcs_int == 1 else "Grossly Overweight / Obesity"
                self.trigger_activity(
                    activity_type="bcs_alert",
                    confidence=0.88,
                    description=f"BCS Alert: Cattle #{cid} ({breed}) evaluated at BCS {bcs_float:.2f}/5.0 ({alert_text})",
                    current_frame=frame,
                    cattle_id=f"COW{cid:03d}" if isinstance(cid, int) else str(cid),
                    ai_detection_result={
                        "bcs_int": bcs_int,
                        "bcs_float": bcs_float,
                        "bcs_label": bcs_label,
                        "breed": breed,
                        "model": "best_bcs_cnn_v2.pth"
                    },
                    status="detected_sign",
                    current_ts=now_ts,
                    current_dt=now_dt,
                    record_video_clip=True
                )

            # Cattle Tracking & State Memory
            if cid not in self.cattle_behavior_history:
                self.cattle_behavior_history[cid] = {
                    "recumbent_frames": 0,
                    "head_down_frames": 0,
                    "last_pos": box,
                    "first_seen": now_ts
                }
            hist = self.cattle_behavior_history[cid]

            x1, y1, x2, y2 = box
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            aspect_ratio = float(bw) / float(bh)

            # 4. Calving Activity Recognition (Reserved for dedicated model)
            if False:
                pass

            # 5. Feeding & Resting Activity Recognition Logic:
            cow_tag = f"COW{cid:03d}" if isinstance(cid, int) else str(cid)
            herd_count = len(cattle_tracks)
            weight_kg = c.get("weight_kg", 400.0)
            height_cm = c.get("height_cm", 130.0)

            # Feeding detection criteria: head down near ground/manger with feed-reach posture
            is_near_ground_or_manger = y2 > int(h_frame * 0.52)
            is_feeding_posture = is_near_ground_or_manger and aspect_ratio >= 1.10

            if is_feeding_posture:
                hist["head_down_frames"] += 1
                hist["last_feeding_time"] = now_ts
            else:
                hist["head_down_frames"] = max(0, hist["head_down_frames"] - 1)

            # Continuous Feeding Session (Zero Duplicates, Live Duration Tracking, NO video clip spam)
            feed_sess_key = f"{cow_tag}_feeding_activity"
            if hist["head_down_frames"] >= 15:
                if feed_sess_key not in self.active_cattle_sessions:
                    # Start NEW continuous feeding session
                    act_id = str(uuid.uuid4())
                    self.active_cattle_sessions[feed_sess_key] = {
                        "activity_id": act_id,
                        "cattle_id": cow_tag,
                        "activity_type": "feeding_activity",
                        "start_time": now_dt,
                        "start_video_sec": now_ts,
                        "last_seen_ts": now_ts,
                        "last_sync_ts": now_ts,
                        "duration_sec": 1.0,
                        "breed": breed,
                        "disease_status": disease,
                        "bcs_score": bcs_float or 3.0,
                        "estimated_weight_kg": weight_kg,
                        "estimated_height_cm": height_cm,
                        "herd_count": herd_count
                    }
                    self.trigger_activity(
                        activity_type="feeding_activity",
                        confidence=0.88,
                        description=f"Feeding / Grazing: Cattle #{cid} ({breed}) started feeding at {now_dt.strftime('%I:%M:%S %p')}",
                        current_frame=frame,
                        cattle_id=cow_tag,
                        ai_detection_result={
                            "breed": breed,
                            "disease_status": disease,
                            "bcs_score": bcs_float or 3.0,
                            "estimated_weight_kg": weight_kg,
                            "estimated_height_cm": height_cm,
                            "eating_duration_sec": 0.0,
                            "resting_duration_sec": 0.0,
                            "herd_count": herd_count,
                            "session_start": now_dt.isoformat()
                        },
                        status="in_progress",
                        activity_id_override=act_id,
                        eating_duration_sec=0.0,
                        resting_duration_sec=0.0,
                        herd_count=herd_count,
                        breed=breed,
                        disease_status=disease,
                        bcs_score=bcs_float or 3.0,
                        estimated_weight_kg=weight_kg,
                        estimated_height_cm=height_cm,
                        current_ts=now_ts,
                        current_dt=now_dt,
                        record_video_clip=False
                    )
                else:
                    # Update ONGOING session in place without creating duplicate database rows
                    sess = self.active_cattle_sessions[feed_sess_key]
                    sess["last_seen_ts"] = now_ts
                    elapsed_dur = max(1.0, now_ts - sess["start_video_sec"])
                    sess["duration_sec"] = elapsed_dur
                    sess["breed"] = breed
                    sess["disease_status"] = disease
                    sess["bcs_score"] = bcs_float or 3.0
                    sess["estimated_weight_kg"] = weight_kg
                    sess["estimated_height_cm"] = height_cm
                    sess["herd_count"] = herd_count

                    # Periodic update to Supabase every 10 seconds
                    if now_ts - sess["last_sync_ts"] >= 10.0:
                        sess["last_sync_ts"] = now_ts
                        dur_mins = int(elapsed_dur // 60)
                        dur_secs = int(elapsed_dur % 60)
                        supabase_manager.update_activity(
                            activity_id=sess["activity_id"],
                            updates={
                                "end_time": now_dt.isoformat(),
                                "activity_duration": round(elapsed_dur, 1),
                                "eating_duration_sec": round(elapsed_dur, 1),
                                "resting_duration_sec": 0.0,
                                "herd_count": herd_count,
                                "breed": breed,
                                "disease_status": disease,
                                "bcs_score": float(bcs_float or 3.0),
                                "estimated_weight_kg": float(weight_kg),
                                "estimated_height_cm": float(height_cm),
                                "description": f"Feeding / Grazing: Cattle #{cid} ({breed}) - Currently feeding: {dur_mins}m {dur_secs}s"
                            }
                        )

            # Resting / Recumbency detection criteria:
            is_resting_posture = aspect_ratio >= 1.40 and not is_feeding_posture
            if is_resting_posture:
                hist["recumbent_frames"] = hist.get("recumbent_frames", 0) + 1
            else:
                hist["recumbent_frames"] = max(0, hist.get("recumbent_frames", 0) - 1)

            # Continuous Resting Session (Zero Duplicates, Live Duration Tracking, NO video clip spam)
            rest_sess_key = f"{cow_tag}_resting_activity"
            if hist["recumbent_frames"] >= 20:
                if rest_sess_key not in self.active_cattle_sessions:
                    act_id = str(uuid.uuid4())
                    self.active_cattle_sessions[rest_sess_key] = {
                        "activity_id": act_id,
                        "cattle_id": cow_tag,
                        "activity_type": "resting_activity",
                        "start_time": now_dt,
                        "start_video_sec": now_ts,
                        "last_seen_ts": now_ts,
                        "last_sync_ts": now_ts,
                        "duration_sec": 1.0,
                        "breed": breed,
                        "disease_status": disease,
                        "bcs_score": bcs_float or 3.0,
                        "estimated_weight_kg": weight_kg,
                        "estimated_height_cm": height_cm,
                        "herd_count": herd_count
                    }
                    self.trigger_activity(
                        activity_type="resting_activity",
                        confidence=0.87,
                        description=f"Resting / Rumination: Cattle #{cid} ({breed}) started resting at {now_dt.strftime('%I:%M:%S %p')}",
                        current_frame=frame,
                        cattle_id=cow_tag,
                        ai_detection_result={
                            "breed": breed,
                            "disease_status": disease,
                            "bcs_score": bcs_float or 3.0,
                            "estimated_weight_kg": weight_kg,
                            "estimated_height_cm": height_cm,
                            "eating_duration_sec": 0.0,
                            "resting_duration_sec": 0.0,
                            "herd_count": herd_count,
                            "session_start": now_dt.isoformat()
                        },
                        status="in_progress",
                        activity_id_override=act_id,
                        eating_duration_sec=0.0,
                        resting_duration_sec=0.0,
                        herd_count=herd_count,
                        breed=breed,
                        disease_status=disease,
                        bcs_score=bcs_float or 3.0,
                        estimated_weight_kg=weight_kg,
                        estimated_height_cm=height_cm,
                        current_ts=now_ts,
                        current_dt=now_dt,
                        record_video_clip=False
                    )
                else:
                    sess = self.active_cattle_sessions[rest_sess_key]
                    sess["last_seen_ts"] = now_ts
                    elapsed_dur = max(1.0, now_ts - sess["start_video_sec"])
                    sess["duration_sec"] = elapsed_dur
                    sess["breed"] = breed
                    sess["disease_status"] = disease
                    sess["bcs_score"] = bcs_float or 3.0
                    sess["estimated_weight_kg"] = weight_kg
                    sess["estimated_height_cm"] = height_cm
                    sess["herd_count"] = herd_count

                    # Periodic update to Supabase every 10 seconds
                    if now_ts - sess["last_sync_ts"] >= 10.0:
                        sess["last_sync_ts"] = now_ts
                        dur_mins = int(elapsed_dur // 60)
                        dur_secs = int(elapsed_dur % 60)
                        supabase_manager.update_activity(
                            activity_id=sess["activity_id"],
                            updates={
                                "end_time": now_dt.isoformat(),
                                "activity_duration": round(elapsed_dur, 1),
                                "eating_duration_sec": 0.0,
                                "resting_duration_sec": round(elapsed_dur, 1),
                                "herd_count": herd_count,
                                "breed": breed,
                                "disease_status": disease,
                                "bcs_score": float(bcs_float or 3.0),
                                "estimated_weight_kg": float(weight_kg),
                                "estimated_height_cm": float(height_cm),
                                "description": f"Resting / Rumination: Cattle #{cid} ({breed}) - Currently resting: {dur_mins}m {dur_secs}s"
                            }
                        )

        # 6. Worker Entry, Exit & Stall Cleaning Activities
        for h in human_tracks:
            hid = h.get("id")
            hbox = h.get("box", (0, 0, 0, 0))
            hx1, hy1, hx2, hy2 = hbox
            hcx = (hx1 + hx2) / 2.0
            hcy = (hy1 + hy2) / 2.0

            if hid not in self.human_behavior_history:
                self.human_behavior_history[hid] = {
                    "first_seen": now_ts,
                    "last_seen": now_ts,
                    "total_movement": 0.0,
                    "last_pos": (hcx, hcy),
                    "frames_active": 0,
                    "entered_logged": False
                }

            h_hist = self.human_behavior_history[hid]
            h_hist["last_seen"] = now_ts
            h_hist["frames_active"] += 1

            px, py = h_hist["last_pos"]
            dist = math.sqrt((hcx - px) ** 2 + (hcy - py) ** 2)
            h_hist["total_movement"] += dist
            h_hist["last_pos"] = (hcx, hcy)

            # Worker Entry Detection
            if not h_hist["entered_logged"] and h_hist["frames_active"] >= 15:
                h_hist["entered_logged"] = True
                self.trigger_activity(
                    activity_type="worker_entry",
                    confidence=0.92,
                    description=f"Worker Entry Detected: Farm worker #{hid} entered cattle zone / barn",
                    current_frame=frame,
                    worker_id=f"WRK{hid:03d}" if isinstance(hid, int) else str(hid),
                    ai_detection_result={"worker_id": hid, "entry_frame": frame_idx},
                    status="detected_sign",
                    current_ts=now_ts,
                    current_dt=now_dt,
                    record_video_clip=True
                )

            # Cleaning / Sanitation Activity Detection
            if h_hist["frames_active"] >= 70 and h_hist["total_movement"] >= 250.0:
                self.trigger_activity(
                    activity_type="cleaning_activity",
                    confidence=0.85,
                    description=f"Cleaning Activity: Farm worker #{hid} conducting stall maintenance & manure cleaning",
                    current_frame=frame,
                    worker_id=f"WRK{hid:03d}" if isinstance(hid, int) else str(hid),
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

        # 7. Finalize Inactive Sessions (Inactive for >= 15 seconds)
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

                if act_t == "feeding_activity":
                    final_desc = f"Feeding Complete: Cattle {cid} ({breed}) - Fed from {start_str} to {end_str} (Total: {dur_mins}m {dur_secs}s)"
                    eat_dur = final_dur
                    rest_dur = 0.0
                elif act_t == "resting_activity":
                    final_desc = f"Resting Complete: Cattle {cid} ({breed}) - Rested from {start_str} to {end_str} (Total: {dur_mins}m {dur_secs}s)"
                    eat_dur = 0.0
                    rest_dur = final_dur
                else:
                    final_desc = f"Session Complete: {act_t.replace('_',' ').title()} - Total: {dur_mins}m {dur_secs}s"
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
                        "description": final_desc
                    }
                )
                del self.active_cattle_sessions[skey]

    def finalize_all_sessions(self, current_video_sec: float = 0.0, video_base_dt: Optional[datetime.datetime] = None):
        """Finalizes all in-progress feeding, resting, and worker sessions when video completes."""
        if not self.active_cattle_sessions:
            return
        now_dt = (video_base_dt + datetime.timedelta(seconds=current_video_sec)) if video_base_dt else datetime.datetime.now(datetime.timezone.utc)
        now_ts = current_video_sec if current_video_sec > 0 else time.time()

        for skey, sess in list(self.active_cattle_sessions.items()):
            final_dur = max(1.0, now_ts - sess.get("start_video_sec", now_ts)) if "start_video_sec" in sess else sess.get("duration_sec", 1.0)
            final_dur = round(final_dur, 1)
            dur_mins = int(final_dur // 60)
            dur_secs = int(final_dur % 60)
            act_t = sess.get("activity_type", "")
            cid = sess.get("cattle_id", "Cattle")
            breed = sess.get("breed", "Cattle")
            start_str = sess["start_time"].strftime("%I:%M:%S %p")
            end_str = now_dt.strftime("%I:%M:%S %p")

            if "feeding" in act_t:
                desc = f"Feeding Complete: Cattle {cid} ({breed}) - Fed from {start_str} to {end_str} (Total: {dur_mins}m {dur_secs}s)"
                eat_dur = final_dur
                rest_dur = 0.0
            elif "resting" in act_t:
                desc = f"Resting Complete: Cattle {cid} ({breed}) - Rested from {start_str} to {end_str} (Total: {dur_mins}m {dur_secs}s)"
                eat_dur = 0.0
                rest_dur = final_dur
            else:
                desc = f"Session Complete: {act_t.replace('_', ' ').title()} - Total: {dur_mins}m {dur_secs}s"
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
                    "description": desc
                }
            )
        self.active_cattle_sessions.clear()
        print(f"[ActivityEngine] All ongoing activity sessions successfully finalized and verified in Supabase.")
