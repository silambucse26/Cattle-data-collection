"""
PashuX AI - Pipeline Performance Telemetry & Profiler
Measures:
- Per-model inference latencies (YOLO det, Breed, BCS, Disease)
- Analyzed FPS & total video processing duration
- Process RAM usage & System RAM availability
- GPU utilization (when available) or CPU load
- Active & queued job counts
"""

import time
import os
import psutil
import torch
from typing import Dict, Any, Optional

class PerformanceProfiler:
    """Collects real-time performance metrics for video analysis pipeline."""
    def __init__(self, sample_interval: int = 30):
        self.sample_interval = sample_interval
        self.process = psutil.Process(os.getpid())
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.frames_analyzed = 0
        self.total_det_time = 0.0
        self.total_breed_time = 0.0
        self.total_bcs_time = 0.0
        self.total_disease_time = 0.0
        self.model_calls = {"det": 0, "breed": 0, "bcs": 0, "disease": 0}

    def record_inference(self, model_name: str, duration_sec: float):
        if model_name == "det":
            self.total_det_time += duration_sec
            self.model_calls["det"] += 1
        elif model_name == "breed":
            self.total_breed_time += duration_sec
            self.model_calls["breed"] += 1
        elif model_name == "bcs":
            self.total_bcs_time += duration_sec
            self.model_calls["bcs"] += 1
        elif model_name == "disease":
            self.total_disease_time += duration_sec
            self.model_calls["disease"] += 1

    def step_frame(self, active_jobs_count: int = 1) -> Optional[Dict[str, Any]]:
        self.frames_analyzed += 1
        if self.frames_analyzed % self.sample_interval == 0:
            return self.get_summary(active_jobs_count)
        return None

    def get_fps(self) -> float:
        elapsed = max(0.001, time.time() - self.start_time)
        return round(self.frames_analyzed / elapsed, 1)

    def get_summary(self, active_jobs_count: int = 1) -> Dict[str, Any]:
        elapsed = max(0.001, time.time() - self.start_time)
        fps = round(self.frames_analyzed / elapsed, 2)

        # Memory telemetry
        mem_info = self.process.memory_info()
        ram_used_mb = round(mem_info.rss / (1024 * 1024), 1)
        sys_ram_avail_gb = round(psutil.virtual_memory().available / (1024**3), 2)

        # GPU telemetry (if available)
        gpu_stat = "N/A (CPU Mode)"
        if torch.cuda.is_available():
            try:
                gpu_mem_used_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
                gpu_stat = f"{torch.cuda.get_device_name(0)} ({gpu_mem_used_mb}MB Allocated)"
            except Exception:
                gpu_stat = "CUDA Active"

        # Per-model average latencies in ms
        det_avg_ms = round((self.total_det_time / max(1, self.model_calls["det"])) * 1000, 1)
        breed_avg_ms = round((self.total_breed_time / max(1, self.model_calls["breed"])) * 1000, 1) if self.model_calls["breed"] else 0.0
        bcs_avg_ms = round((self.total_bcs_time / max(1, self.model_calls["bcs"])) * 1000, 1) if self.model_calls["bcs"] else 0.0
        disease_avg_ms = round((self.total_disease_time / max(1, self.model_calls["disease"])) * 1000, 1) if self.model_calls["disease"] else 0.0

        summary = {
            "frames_analyzed": self.frames_analyzed,
            "elapsed_seconds": round(elapsed, 2),
            "analyzed_fps": fps,
            "yolo_det_avg_ms": det_avg_ms,
            "breed_avg_ms": breed_avg_ms,
            "bcs_avg_ms": bcs_avg_ms,
            "disease_avg_ms": disease_avg_ms,
            "specialist_calls": {
                "breed": self.model_calls["breed"],
                "bcs": self.model_calls["bcs"],
                "disease": self.model_calls["disease"]
            },
            "ram_used_mb": ram_used_mb,
            "sys_ram_available_gb": sys_ram_avail_gb,
            "gpu_status": gpu_stat,
            "active_jobs_count": active_jobs_count
        }

        print(
            f"[PERF] Analyzed FPS: {fps} | YOLO: {det_avg_ms}ms | "
            f"Breed({self.model_calls['breed']}): {breed_avg_ms}ms | BCS({self.model_calls['bcs']}): {bcs_avg_ms}ms | "
            f"Disease({self.model_calls['disease']}): {disease_avg_ms}ms | RAM: {ram_used_mb}MB | GPU: {gpu_stat} | Jobs: {active_jobs_count}"
        )
        return summary

# Global singleton profiler
telemetry_profiler = PerformanceProfiler()

