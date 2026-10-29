
"""
PashuX AI - Single Active Video Analysis Job Queue & Deduplication Engine
Features:
- Single active background worker thread pulling video analysis jobs sequentially.
- Job states: 'queued', 'processing', 'completed', 'failed'.
- File hash deduplication prevents duplicate video uploads from spawning redundant processing.
- Thread-safe job state query and management without blocking FastAPI.
"""

import os
import time
import uuid
import hashlib
import threading
import queue
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple

def compute_file_hash(filepath: str, chunk_size: int = 1024 * 1024) -> str:
    """Computes fast SHA-256 hash of file (reads first 4MB for instant deduplication)."""
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for _ in range(4):
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
            # Incorporate file size for boundary collision protection
            h.update(str(os.path.getsize(filepath)).encode("utf-8"))
    except Exception:
        h.update(filepath.encode("utf-8"))
    return h.hexdigest()

class VideoJobQueue:
    """Thread-safe queue manager for single active video processing."""
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super(VideoJobQueue, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.hash_to_job: Dict[str, str] = {}
        self.work_queue = queue.Queue()
        self.lock = threading.Lock()

        self.current_job_id: Optional[str] = None
        self.active_processor_func: Optional[Callable] = None
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("[VideoJobQueue] Single active video queue worker started.")

    def set_processor(self, processor_func: Callable):
        """Sets the callback function to execute video analysis for a job."""
        self.active_processor_func = processor_func

    def submit_job(
        self,
        filepath: str,
        filename: str,
        farm_id: str,
        camera_id: str = "CAM01",
        enabled_models: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Submits video to the analysis queue with optional user-selected AI models.
        Returns: (job_dict, is_new_job)
        """
        file_hash = compute_file_hash(filepath)

        with self.lock:
            # Check for existing actively queued or processing job with identical hash
            if file_hash in self.hash_to_job:
                existing_id = self.hash_to_job[file_hash]
                existing_job = self.jobs.get(existing_id)
                if existing_job and existing_job["status"] in ("queued", "processing"):
                    print(f"[VideoJobQueue] Active job {existing_id} is already in state: {existing_job['status']}")
                    return existing_job, False

            job_id = f"job_{uuid.uuid4().hex[:8]}"
            job = {
                "job_id": job_id,
                "file_hash": file_hash,
                "filepath": filepath,
                "filename": filename,
                "farm_id": farm_id,
                "camera_id": camera_id,
                "enabled_models": enabled_models or ["breed", "bcs", "biometry", "disease", "estrus", "worker"],
                "status": "queued",
                "progress_percent": 0.0,
                "frames_processed": 0,
                "total_frames": 0,
                "created_at": time.time(),
                "started_at": None,
                "completed_at": None,
                "duration_seconds": 0.0,
                "analyzed_fps": 0.0,
                "metrics": {},
                "error": None
            }
            self.jobs[job_id] = job
            self.hash_to_job[file_hash] = job_id
            self.work_queue.put(job_id)
            print(f"[VideoJobQueue] Job {job_id} submitted for '{filename}'. Queue depth: {self.work_queue.qsize()}")
            return job, True

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.jobs.get(job_id)

    def list_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self.lock:
            all_jobs = list(self.jobs.values())
            all_jobs.sort(key=lambda j: j.get("created_at", 0), reverse=True)
            return all_jobs[:limit]

    def get_active_count(self) -> int:
        with self.lock:
            return sum(1 for j in self.jobs.values() if j["status"] in ("queued", "processing"))

    def update_progress(self, job_id: str, frames_processed: int, total_frames: int, fps: float = 0.0, metrics: Dict[str, Any] = None):
        with self.lock:
            job = self.jobs.get(job_id)
            if job and job["status"] == "processing":
                job["frames_processed"] = frames_processed
                job["total_frames"] = total_frames
                job["analyzed_fps"] = fps
                if total_frames > 0:
                    job["progress_percent"] = round((frames_processed / float(total_frames)) * 100.0, 1)
                if metrics:
                    job["metrics"].update(metrics)

    def _worker_loop(self):
        """Worker executing single video analysis job at a time."""
        while True:
            job_id = self.work_queue.get()
            try:
                with self.lock:
                    job = self.jobs.get(job_id)
                    if not job:
                        continue
                    job["status"] = "processing"
                    job["started_at"] = time.time()
                    self.current_job_id = job_id

                print(f"\n[VideoJobQueue] >>> Starting processing job {job_id}: {job['filename']}")
                t0 = time.time()

                if self.active_processor_func:
                    # Execute analysis via processor callback
                    success, error_msg, perf_metrics = self.active_processor_func(job)
                else:
                    success, error_msg, perf_metrics = False, "No processor registered", {}

                t_dur = time.time() - t0
                with self.lock:
                    if success:
                        job["status"] = "completed"
                        job["progress_percent"] = 100.0
                    else:
                        job["status"] = "failed"
                        job["error"] = error_msg

                    job["completed_at"] = time.time()
                    job["duration_seconds"] = round(t_dur, 2)
                    if perf_metrics:
                        job["metrics"].update(perf_metrics)
                        if "analyzed_fps" in perf_metrics:
                            job["analyzed_fps"] = perf_metrics["analyzed_fps"]

                print(f"[VideoJobQueue] <<< Job {job_id} {job['status'].upper()} in {t_dur:.2f}s (FPS: {job.get('analyzed_fps', 0.0)})\n")
            except Exception as e:
                print(f"[VideoJobQueue] Job {job_id} exception: {e}")
                with self.lock:
                    if job_id in self.jobs:
                        self.jobs[job_id]["status"] = "failed"
                        self.jobs[job_id]["error"] = str(e)
            finally:
                with self.lock:
                    self.current_job_id = None
                self.work_queue.task_done()

# Global job queue singleton
video_queue = VideoJobQueue()
