import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

# Ensure Windows finds PyTorch DLL dependencies reliably
if sys.platform == "win32":
    torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib) and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass

import torch
import numpy as np
import cv2

# Ensure Keras uses torch backend
os.environ["KERAS_BACKEND"] = "torch"
import warnings
warnings.filterwarnings("ignore")

class AIModelManager:
    """Singleton Manager for loading and executing all PashuX AI models."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AIModelManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.inference_lock = threading.Lock()

        # Hardware acceleration detection
        self.has_cuda = torch.cuda.is_available()
        self.device = "cuda" if self.has_cuda else "cpu"
        self.device_name = torch.cuda.get_device_name(0) if self.has_cuda else "CPU"

        # Optimize PyTorch CPU threading if running on CPU
        if not self.has_cuda:
            cpu_threads = min(4, os.cpu_count() or 4)
            try:
                torch.set_num_threads(cpu_threads)
            except Exception:
                pass

        print(f"[AIModelManager] Initialized on device: {self.device.upper()} ({self.device_name})")

        # Models storage
        self.det_model = None
        self.breed_model = None
        self.bcs_model = None
        self.disease_model = None
        self.estrus_model = None

        self._load_all_models()

    def _load_all_models(self):
        """Loads each model exactly once with garbage collection between steps."""
        import gc
        t0 = time.time()
        from ultralytics import YOLO

        # 1. Primary YOLO Object Detector (yolo11n.pt)
        det_file = "yolo11n.pt" if os.path.exists("yolo11n.pt") else ("yolo11m.pt" if os.path.exists("yolo11m.pt") else "yolo11n.pt")
        try:
            self.det_model = YOLO(det_file)
            print(f"[AIModelManager] Primary Detector loaded: {det_file}")
        except Exception as e:
            print(f"[AIModelManager] Error loading primary detector {det_file}: {e}")
        gc.collect()

        # 2. Cattle Breed Classification Model (cattle_breed_model.pt)
        if os.path.exists("cattle_breed_model.pt"):
            try:
                self.breed_model = YOLO("cattle_breed_model.pt")
                print("[AIModelManager] Breed Classifier loaded: cattle_breed_model.pt")
            except Exception as e:
                print(f"[AIModelManager] Error loading breed model: {e}")
        gc.collect()

        # 3. BCS Body Condition Score CNN (best_bcs_cnn_v2.pth)
        if os.path.exists("best_bcs_cnn_v2.pth"):
            try:
                from camera_test import load_bcs_model
                self.bcs_model, _ = load_bcs_model("best_bcs_cnn_v2.pth")
                if self.bcs_model and self.has_cuda:
                    self.bcs_model.to("cuda")
                print(f"[AIModelManager] BCS Model loaded: best_bcs_cnn_v2.pth (Device: {self.device})")
            except Exception as e:
                print(f"[AIModelManager] Error loading BCS model: {e}")
        gc.collect()

        # 4. Disease Diagnostic Keras Model (final_cattle_disease_model.keras)
        if os.path.exists("final_cattle_disease_model.keras"):
            try:
                from camera_test import load_disease_model
                self.disease_model = load_disease_model("final_cattle_disease_model.keras")
                print("[AIModelManager] Disease Model loaded: final_cattle_disease_model.keras")
            except Exception as e:
                print(f"[AIModelManager] Error loading disease model: {e}")
        gc.collect()

        # 5. Estrus Detection Model (estrusdetection.pt)
        if os.path.exists("estrusdetection.pt"):
            try:
                self.estrus_model = YOLO("estrusdetection.pt")
                print("[AIModelManager] Estrus Detector loaded: estrusdetection.pt")
            except Exception as e:
                print(f"[AIModelManager] Error loading estrus model: {e}")
        gc.collect()

        print(f"[AIModelManager] All models loaded and cached in {time.time() - t0:.2f}s")

    def run_detection(self, frame: np.ndarray, imgsz: int = 640, conf: float = 0.22):
        """Runs primary YOLO detection on frame using optimized resolution."""
        if self.det_model is None:
            return []
        with torch.inference_mode():
            results = self.det_model.predict(
                frame,
                classes=[0, 19],  # 0: person, 19: cow
                imgsz=imgsz,
                conf=conf,
                device=self.device,
                verbose=False
            )
        return results

    def run_breed(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Runs Breed Classifier on a single cattle crop."""
        if self.breed_model is None or crop_bgr is None or crop_bgr.shape[0] < 20 or crop_bgr.shape[1] < 20:
            return "Cattle", 0.50
        try:
            with torch.inference_mode():
                results = self.breed_model.predict(crop_bgr, imgsz=224, device=self.device, verbose=False)
            if results and len(results) > 0 and results[0].probs is not None:
                top1_idx = int(results[0].probs.top1)
                conf = float(results[0].probs.top1conf)
                raw_name = results[0].names[top1_idx]
                clean_name = raw_name.replace("-", " ").replace("_", " ").title()
                return clean_name, conf
        except Exception:
            pass
        return "Cattle", 0.50

    def run_bcs(self, crop_bgr: np.ndarray) -> Tuple[int, float, str, float]:
        """Runs Body Condition Score CNN on a single cattle crop."""
        if self.bcs_model is None or crop_bgr is None or crop_bgr.shape[0] < 25 or crop_bgr.shape[1] < 25:
            return 3, 3.00, "Ideal", 0.70
        try:
            from camera_test import predict_cattle_bcs
            bcs_int, bcs_float, bcs_label, bcs_conf, _ = predict_cattle_bcs(crop_bgr, self.bcs_model, self.device)
            return bcs_int, bcs_float, bcs_label, bcs_conf
        except Exception:
            return 3, 3.00, "Ideal", 0.70

    def run_disease(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Runs Disease Diagnosis Model on a single cattle crop."""
        if self.disease_model is None or crop_bgr is None:
            return "Healthy", 0.95
        try:
            from camera_test import predict_cattle_disease
            disease_status, disease_conf, _ = predict_cattle_disease(crop_bgr, self.disease_model)
            return disease_status, disease_conf
        except Exception:
            return "Healthy", 0.95

    def has_model(self, model_name: str) -> bool:
        """Checks if a specialist AI model is loaded and ready."""
        if model_name == "det":
            return self.det_model is not None
        elif model_name == "breed":
            return self.breed_model is not None
        elif model_name == "bcs":
            return self.bcs_model is not None
        elif model_name == "disease":
            return self.disease_model is not None
        elif model_name == "estrus":
            return self.estrus_model is not None
        return False

    def run_estrus(self, frame: np.ndarray, conf: float = 0.30) -> List[Dict[str, Any]]:
        """Runs estrus / mounting detector on frame."""
        if self.estrus_model is None:
            return []
        try:
            with torch.inference_mode():
                results = self.estrus_model.predict(frame, conf=conf, device=self.device, verbose=False)
            events = []
            if results and len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    box_conf = float(box.conf[0])
                    lbl = results[0].names.get(cls_id, "Estrus")
                    events.append({"label": lbl, "conf": box_conf})
            return events
        except Exception:
            return []

# Singleton global instance
ai_models = AIModelManager()
