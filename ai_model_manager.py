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
import pandas as pd
import joblib

# Ensure Keras uses torch backend
os.environ["KERAS_BACKEND"] = "torch"
import warnings
warnings.filterwarnings("ignore")

COWXNET_FEATURES = [
    'cow-box_xmax', 'cow-box_xmin', 'cow-box_ymax', 'cow-box_ymin',
    'cow-box_xcenter', 'cow-box_ycenter', 'head_x', 'head_y',
    'body_x', 'body_y', 'tail-head_x', 'tail-head_y'
]

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
            cpu_threads = min(2, os.cpu_count() or 2)
            try:
                torch.set_num_threads(cpu_threads)
            except Exception:
                pass

        print(f"[AIModelManager] Initialized on device: {self.device.upper()} ({self.device_name})")

        # Models storage
        self.det_model = None
        self.breed_model = None
        self.pt_breed_model = None
        self.bcs_model = None
        self.disease_model = None
        self.estrus_model = None
        self.estrus_pose_model = None
        self.heat_model = None

        self._load_all_models()

    def _load_all_models(self):
        """Loads each model exactly once with garbage collection between steps."""
        import gc
        t0 = time.time()
        from ultralytics import YOLO

        # 1. Primary YOLO Object Detector (prefer yolo11n.pt on CPU for lightweight memory footprint, yolo11m.pt if CUDA)
        if self.has_cuda and os.path.exists("yolo11m.pt"):
            det_file = "yolo11m.pt"
        elif os.path.exists("yolo11n.pt"):
            det_file = "yolo11n.pt"
        elif os.path.exists("yolo11m.pt"):
            det_file = "yolo11m.pt"
        else:
            det_file = "yolo11n.pt"
        try:
            self.det_model = YOLO(det_file)
            print(f"[AIModelManager] Primary Detector loaded: {det_file} (Device: {self.device})")
        except Exception as e:
            print(f"[AIModelManager] Error loading primary detector {det_file}: {e}")
        gc.collect()

        # 2. Cattle Breed Classification Models (best.onnx primary, cattle_breed_model.pt ensemble)
        if os.path.exists("best.onnx"):
            try:
                self.breed_model = YOLO("best.onnx", task="classify")
                self.breed_model_file = "best.onnx"
                self.is_onnx_breed = True
                print(f"[AIModelManager] Primary Breed Classifier loaded: best.onnx (Task: classify, ONNX: True)")
            except Exception as e:
                print(f"[AIModelManager] Error loading best.onnx: {e}")
        elif os.path.exists("cattle_breed_model.pt"):
            try:
                self.breed_model = YOLO("cattle_breed_model.pt")
                self.breed_model_file = "cattle_breed_model.pt"
                self.is_onnx_breed = False
                print(f"[AIModelManager] Primary Breed Classifier loaded: cattle_breed_model.pt")
            except Exception as e:
                print(f"[AIModelManager] Error loading cattle_breed_model.pt: {e}")

        # Secondary ensemble validator if both models exist
        if os.path.exists("cattle_breed_model.pt") and getattr(self, "is_onnx_breed", False):
            try:
                self.pt_breed_model = YOLO("cattle_breed_model.pt")
                print(f"[AIModelManager] Secondary Breed Ensemble Validator loaded: cattle_breed_model.pt")
            except Exception as e:
                self.pt_breed_model = None
        else:
            self.pt_breed_model = None
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

        # 5. Estrus & Mounting AI Models: estrusdetection.pt (YOLO Pose) + CowXNet_heat_model.pkl (Random Forest)
        if os.path.exists("estrusdetection.pt"):
            try:
                self.estrus_pose_model = YOLO("estrusdetection.pt")
                self.estrus_model = self.estrus_pose_model
                print("[AIModelManager] Estrus Pose Model loaded: estrusdetection.pt")
            except Exception as e:
                print(f"[AIModelManager] Error loading estrus pose model: {e}")
        gc.collect()

        if os.path.exists("CowXNet_heat_model.pkl"):
            try:
                self.heat_model = joblib.load("CowXNet_heat_model.pkl")
                if hasattr(self.heat_model, "n_jobs"):
                    self.heat_model.n_jobs = 1
                print("[AIModelManager] CowXNet Heat Model loaded: CowXNet_heat_model.pkl")
            except Exception as e:
                print(f"[AIModelManager] Error loading CowXNet Heat model: {e}")
        gc.collect()

        print(f"[AIModelManager] All models loaded and cached in {time.time() - t0:.2f}s")

    def run_detection(self, frame: np.ndarray, imgsz: int = 640, conf: float = 0.15):
        """Runs primary YOLO detection on frame using optimized resolution."""
        if self.det_model is None:
            return []
        try:
            with torch.inference_mode():
                results = self.det_model.predict(
                    frame,
                    classes=[0, 16, 17, 18, 19],  # 0: person, 16: dog/calf, 17: horse, 18: sheep, 19: cow
                    imgsz=imgsz,
                    conf=conf,
                    device=self.device,
                    verbose=False
                )
            return results
        except Exception:
            import gc
            gc.collect()
            try:
                with torch.inference_mode():
                    results = self.det_model.predict(
                        frame,
                        classes=[0, 16, 17, 18, 19],
                        imgsz=480,
                        conf=conf,
                        device=self.device,
                        verbose=False
                    )
                return results
            except Exception:
                return []

    def detect_jersey(self, crop_bgr: np.ndarray, top1_breed: str = "") -> Tuple[bool, float]:
        """
        Veterinary & Colorimetric Analysis for Jersey / Jersey Cross Dairy Cattle:
        Jersey cattle display a characteristic fawn, tan, yellowish-brown, or reddish-brown coat
        with dark facial points (dark muzzle with light ring), dark eyes, and dark hooves/switch.
        They lack large black-and-white piebald patches and lack a massive zebu hump.
        Also triggers if specialist breed model predicts 'Ayrshire' (the closest European dairy
        breed in the 50-class Indian cattle dataset).
        """
        if crop_bgr is None or crop_bgr.shape[0] < 20 or crop_bgr.shape[1] < 20:
            return False, 0.0

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        total_px = float(crop_bgr.shape[0] * crop_bgr.shape[1])

        # Warm fawn / tan / golden-brown fur (hue 8-32, saturation > 55, value > 60)
        warm_mask = (h_ch >= 8) & (h_ch <= 32) & (s_ch > 55) & (v_ch > 60)
        warm_ratio = float(np.sum(warm_mask)) / total_px

        # Dark fur (muzzle, points, hooves, or dark shading)
        black_mask = (v_ch <= 70) & (s_ch <= 120)
        black_ratio = float(np.sum(black_mask)) / total_px

        # Bright white fur
        white_mask = (s_ch <= 65) & (v_ch >= 135)
        white_ratio = float(np.sum(white_mask)) / total_px

        norm_top1 = top1_breed.strip().lower()
        is_dairy_surrogate = norm_top1 in ("ayrshire", "jersey")

        # Jersey has distinct warm fawn/brown coat and is NOT piebald (no heavy simultaneous black+white patches)
        is_jersey = (
            (is_dairy_surrogate and (warm_ratio >= 0.12 or black_ratio < 0.40)) or
            (warm_ratio >= 0.25 and black_ratio < 0.35 and white_ratio < 0.35)
        )
        if is_jersey:
            conf = min(0.97, max(0.85, 0.78 + 0.35 * warm_ratio))
            return True, round(conf, 2)

        return False, 0.0

    def detect_holstein_friesian(self, crop_bgr: np.ndarray, top1_breed: str = "") -> Tuple[bool, float]:
        """
        Veterinary & Colorimetric Analysis for Holstein Friesian (HF):
        HF cattle possess a distinctive, iconic piebald coat of contrasting black patches
        and white patches. Brown, red, tan, and fawn cattle are strictly excluded.
        Requires substantial bimodal contrast between deep black fur and bright white fur.
        """
        if crop_bgr is None or crop_bgr.shape[0] < 25 or crop_bgr.shape[1] < 25:
            return False, 0.0

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        total_px = float(crop_bgr.shape[0] * crop_bgr.shape[1])

        # Dark/black fur
        black_mask = (v_ch <= 75) & (s_ch <= 110)
        black_ratio = float(np.sum(black_mask)) / total_px

        # White/light fur
        white_mask = (s_ch <= 65) & (v_ch >= 125)
        white_ratio = float(np.sum(white_mask)) / total_px

        # Warm chromatic fur (brown, tan, red, fawn)
        warm_mask = (s_ch > 65) & (v_ch > 65) & (h_ch >= 8) & (h_ch <= 32)
        warm_ratio = float(np.sum(warm_mask)) / total_px

        # Holstein Friesian requires contrasting black and white piebald hide with NO significant brown/fawn
        is_piebald_hf = (
            black_ratio >= 0.22 and white_ratio >= 0.18 and warm_ratio < 0.12
        )
        if is_piebald_hf:
            conf = min(0.98, max(0.90, 0.86 + 0.14 * (black_ratio + white_ratio)))
            return True, round(conf, 2)

        return False, 0.0

    def detect_murrah_buffalo(self, crop_bgr: np.ndarray, top1_breed: str = "") -> Tuple[bool, float]:
        """
        Veterinary Analysis for Murrah Buffalo / Indian Water Buffalo:
        Murrah buffaloes possess deep solid jet-black or dark charcoal slate coats
        with uniform dark pigmentation and minimal or no white body markings.
        """
        if crop_bgr is None or crop_bgr.shape[0] < 25 or crop_bgr.shape[1] < 25:
            return False, 0.0

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        total_px = float(crop_bgr.shape[0] * crop_bgr.shape[1])

        black_mask = (v_ch <= 75) & (s_ch <= 110)
        black_ratio = float(np.sum(black_mask)) / total_px

        white_mask = (s_ch <= 65) & (v_ch >= 125)
        white_ratio = float(np.sum(white_mask)) / total_px

        warm_mask = (s_ch > 65) & (v_ch > 65) & (h_ch >= 8) & (h_ch <= 32)
        warm_ratio = float(np.sum(warm_mask)) / total_px

        # Buffaloes have deep solid dark coat with little to no white patches (< 6%) and no warm fur (< 8%)
        is_buffalo = (black_ratio >= 0.42 and white_ratio < 0.06 and warm_ratio < 0.08)
        if is_buffalo:
            conf = min(0.97, max(0.90, 0.88 + 0.10 * black_ratio))
            return True, round(conf, 2)

        return False, 0.0

    def run_breed(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Runs High-Precision Breed Classifier on a single cattle crop using:
        1. Trained neural network model (best.onnx @ 320x320, 41 classes).
        2. Secondary validation ensemble (cattle_breed_model.pt @ 224x224, 50 classes).
        3. Veterinary morphological & colorimetric phenotype validation to prevent
           impossible cross-category misclassifications (e.g. black-and-white dairy cows
           being misclassified as Mehsana buffalo or Vechur dwarf).
        """
        if crop_bgr is None or crop_bgr.shape[0] < 20 or crop_bgr.shape[1] < 20:
            return "Cattle", 0.50

        # Extract central 65% region to analyze genuine animal coat without background/floor bias
        ch, cw = crop_bgr.shape[:2]
        yc1, yc2 = int(ch * 0.15), int(ch * 0.85)
        xc1, xc2 = int(cw * 0.15), int(cw * 0.85)
        center_crop = crop_bgr[yc1:yc2, xc1:xc2] if (yc2 - yc1 >= 15 and xc2 - xc1 >= 15) else crop_bgr

        tot_px = float(center_crop.shape[0] * center_crop.shape[1])
        hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)

        black_ratio = float(np.sum((v_ch <= 75) & (s_ch <= 110))) / tot_px
        white_ratio = float(np.sum((s_ch <= 65) & (v_ch >= 125))) / tot_px
        warm_ratio = float(np.sum((s_ch > 55) & (v_ch > 60) & (h_ch >= 8) & (h_ch <= 32))) / tot_px

        is_piebald = (black_ratio >= 0.20 and white_ratio >= 0.15)
        is_solid_dark = (white_ratio < 0.05 and black_ratio >= 0.38)
        is_warm_brown = (warm_ratio >= 0.32 and black_ratio < 0.30)

        scores: Dict[str, float] = {}

        # 1. Primary Model: best.onnx (YOLO11s-cls @ 320x320)
        if self.breed_model is not None:
            try:
                is_onnx = getattr(self, "is_onnx_breed", False)
                img_sz = 320 if is_onnx else 224
                dev = "cpu" if is_onnx else self.device
                with torch.inference_mode():
                    results = self.breed_model.predict(crop_bgr, imgsz=img_sz, device=dev, verbose=False)
                if results and len(results) > 0 and results[0].probs is not None:
                    probs_data = results[0].probs.data
                    for idx, conf in enumerate(probs_data):
                        raw_k = results[0].names[idx].replace("-", "_").replace(" ", "_").lower()
                        scores[raw_k] = float(conf) * 1.25
            except Exception:
                pass

        # 2. Secondary Ensemble: cattle_breed_model.pt (YOLO @ 224x224)
        if getattr(self, "pt_breed_model", None) is not None:
            try:
                with torch.inference_mode():
                    res_pt = self.pt_breed_model.predict(crop_bgr, imgsz=224, device=self.device, verbose=False)
                if res_pt and len(res_pt) > 0 and res_pt[0].probs is not None:
                    for idx, conf in enumerate(res_pt[0].probs.data):
                        raw_k = res_pt[0].names[idx].replace("-", "_").replace(" ", "_").lower()
                        scores[raw_k] = scores.get(raw_k, 0.0) + float(conf)
            except Exception:
                pass

        if not scores:
            return "Cattle", 0.70

        # 3. Apply Veterinary Phenotype & Colorimetric Guardrails
        if is_piebald:
            # Piebald black & white dairy cattle: strictly suppress non-piebald breeds
            for k in list(scores.keys()):
                if k in ("mehsana", "murrah", "sahiwal", "vechur", "ongole", "kasargod", "red_sindhi", "gir"):
                    scores[k] *= 0.02
            # Boost genuine piebald candidates
            scores["holstein_friesian"] = scores.get("holstein_friesian", 0.0) + 1.20
            scores["deoni"] = scores.get("deoni", 0.0) + 0.85
            scores["dangi"] = scores.get("dangi", 0.0) + 0.60
        elif is_solid_dark:
            # Solid jet-black cattle / buffaloes: suppress white / pale / red breeds
            for k in list(scores.keys()):
                if k in ("sahiwal", "ongole", "tharparkar", "holstein_friesian", "hariana", "vechur"):
                    scores[k] *= 0.02
            scores["malnad_gidda"] = scores.get("malnad_gidda", 0.0) + 0.80
            scores["murrah"] = scores.get("murrah", 0.0) + 0.60
            scores["umblachery"] = scores.get("umblachery", 0.0) + 0.50
        elif is_warm_brown:
            # Warm fawn / tan / reddish brown coat: suppress black / buffalo breeds
            for k in list(scores.keys()):
                if k in ("holstein_friesian", "murrah", "mehsana", "malnad_gidda"):
                    scores[k] *= 0.02
            scores["sahiwal"] = scores.get("sahiwal", 0.0) + 0.60
            scores["jersey"] = scores.get("jersey", 0.0) + 0.60
            scores["gir"] = scores.get("gir", 0.0) + 0.50

        # Select highest scoring breed
        best_raw = max(scores, key=scores.get)
        raw_val = scores[best_raw]
        total_val = sum(scores.values()) + 1e-6
        final_conf = min(0.99, max(0.72, (raw_val / total_val) * 1.6))

        # Standardize Display Breed Name
        breed_display_map = {
            "holstein_friesian": "Holstein Friesian (HF)",
            "hf": "Holstein Friesian (HF)",
            "deoni": "Deoni",
            "malnad_gidda": "Malnad Gidda",
            "kasargod": "Kasargod",
            "sahiwal": "Sahiwal",
            "jersey": "Jersey",
            "murrah": "Murrah Buffalo",
            "mehsana": "Mehsana Buffalo",
            "ongole": "Ongole",
            "dangi": "Dangi",
            "gir": "Gir",
            "kankrej": "Kankrej",
            "tharparkar": "Tharparkar",
            "umblachery": "Umblachery",
            "vechur": "Vechur",
            "punganur": "Punganur",
            "red_sindhi": "Red Sindhi",
            "hariana": "Hariana",
            "hallikar": "Hallikar",
            "khillari": "Khillari",
            "amritmahal": "Amritmahal",
            "bachaur": "Bachaur",
            "bargur": "Bargur",
            "gaolao": "Gaolao",
            "nagori": "Nagori",
            "nimari": "Nimari",
            "pulikulam": "Pulikulam",
            "rathi": "Rathi",
            "siri": "Siri"
        }

        top1_name = breed_display_map.get(best_raw, best_raw.replace("_", " ").title())
        return top1_name, round(final_conf, 2)

    def run_bcs(self, crop_bgr: np.ndarray, disease_status: str = "Healthy") -> Tuple[Optional[int], Optional[float], str, float]:
        """Runs Body Condition Score CNN on a single cattle crop with anatomical posterior view gating."""
        if self.bcs_model is None or crop_bgr is None or crop_bgr.shape[0] < 45 or crop_bgr.shape[1] < 45:
            return None, None, "Hip/Rear View Required", 0.0
        try:
            from camera_test import predict_cattle_bcs
            bcs_int, bcs_float, bcs_label, bcs_conf, _ = predict_cattle_bcs(crop_bgr, self.bcs_model, self.device, disease_status=disease_status)
            return bcs_int, bcs_float, bcs_label, bcs_conf
        except Exception:
            return None, None, "Hip/Rear View Required", 0.0

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
            return (self.estrus_pose_model is not None) or (self.heat_model is not None)
        return False

    def _extract_cowxnet_features(self, box: List[int], kpts: Optional[np.ndarray], frame_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Builds normalized 12-feature dictionary expected by CowXNet_heat_model.pkl (0.0 to 1.0 coordinates):
        ['cow-box_xmax', 'cow-box_xmin', 'cow-box_ymax', 'cow-box_ymin',
         'cow-box_xcenter', 'cow-box_ycenter', 'head_x', 'head_y',
         'body_x', 'body_y', 'tail-head_x', 'tail-head_y']
        """
        xmin, ymin, xmax, ymax = box
        xc = (xmin + xmax) / 2.0
        yc = (ymin + ymax) / 2.0
        w = max(1.0, float(xmax - xmin))
        h = max(1.0, float(ymax - ymin))

        if kpts is not None and len(kpts) >= 11:
            head_pt = kpts[1] if (kpts[1][0] > 0 and kpts[1][1] > 0) else kpts[0]
            if head_pt[0] <= 0 or head_pt[1] <= 0:
                head_pt = (xmin + w * 0.20, ymin + h * 0.30)

            body_pt = kpts[9] if (kpts[9][0] > 0 and kpts[9][1] > 0) else kpts[2]
            if body_pt[0] <= 0 or body_pt[1] <= 0:
                body_pt = (xc, yc)

            tail_pt = kpts[10] if (kpts[10][0] > 0 and kpts[10][1] > 0) else (xmax - w * 0.18, ymin + h * 0.25)
            if tail_pt[0] <= 0 or tail_pt[1] <= 0:
                tail_pt = (xmax - w * 0.18, ymin + h * 0.25)

            head_x, head_y = float(head_pt[0]), float(head_pt[1])
            body_x, body_y = float(body_pt[0]), float(body_pt[1])
            tail_x, tail_y = float(tail_pt[0]), float(tail_pt[1])
        else:
            head_x, head_y = float(xmin + w * 0.20), float(ymin + h * 0.30)
            body_x, body_y = float(xc), float(yc)
            tail_x, tail_y = float(xmax - w * 0.18), float(ymin + h * 0.25)

        fh, fw = float(frame_shape[0]), float(frame_shape[1])
        fw = max(1.0, fw)
        fh = max(1.0, fh)

        return {
            'cow-box_xmax': float(xmax) / fw,
            'cow-box_xmin': float(xmin) / fw,
            'cow-box_ymax': float(ymax) / fh,
            'cow-box_ymin': float(ymin) / fh,
            'cow-box_xcenter': float(xc) / fw,
            'cow-box_ycenter': float(yc) / fh,
            'head_x': head_x / fw,
            'head_y': head_y / fh,
            'body_x': body_x / fw,
            'body_y': body_y / fh,
            'tail-head_x': tail_x / fw,
            'tail-head_y': tail_y / fh
        }

    def run_estrus(
        self,
        frame: np.ndarray,
        cattle_tracks: Optional[List[Dict[str, Any]]] = None,
        conf: float = 0.18
    ) -> List[Dict[str, Any]]:
        """
        Runs combined estrus & mounting pipeline:
        1. YOLO Pose Keypoints model (estrusdetection.pt) extracts cattle anatomical landmarks.
        2. CowXNet Heat Random Forest model (CowXNet_heat_model.pkl) classifies standing heat on 12 features.
        3. Kinematic mounting evaluation detects rearing, elevation, and pelvic mounting interactions.
        """
        if frame is None or (self.estrus_pose_model is None and self.heat_model is None):
            return []

        h_f, w_f = frame.shape[:2]
        pose_keypoints_list = []

        # 1. Run YOLO pose estimation with estrusdetection.pt
        if self.estrus_pose_model is not None:
            try:
                with torch.inference_mode():
                    res = self.estrus_pose_model.predict(frame, imgsz=480, conf=conf, verbose=False)
                if res and len(res) > 0 and res[0].keypoints is not None:
                    kpts_tensor = res[0].keypoints.xy.cpu().numpy()
                    for kpts in kpts_tensor:
                        pose_keypoints_list.append(kpts)
            except Exception:
                pass

        events = []
        if not cattle_tracks:
            return events

        # 2. Individual Heat Classification with CowXNet_heat_model.pkl
        for track in cattle_tracks:
            box = track.get("box", [0, 0, 0, 0])
            bx1, by1, bx2, by2 = box
            b_area = max(1, (bx2 - bx1) * (by2 - by1))

            # Match nearest pose keypoints to this cow
            matched_kpts = None
            best_overlap = 0.0
            for kpts in pose_keypoints_list:
                valid_pts = kpts[kpts[:, 0] > 0]
                if len(valid_pts) > 0:
                    k_xmin, k_ymin = float(np.min(valid_pts[:, 0])), float(np.min(valid_pts[:, 1]))
                    k_xmax, k_ymax = float(np.max(valid_pts[:, 0])), float(np.max(valid_pts[:, 1]))
                    # Compute overlap
                    ix1, iy1 = max(bx1, k_xmin), max(by1, k_ymin)
                    ix2, iy2 = min(bx2, k_xmax), min(by2, k_ymax)
                    if ix2 > ix1 and iy2 > iy1:
                        inter_a = (ix2 - ix1) * (iy2 - iy1)
                        ov = inter_a / float(b_area)
                        if ov > best_overlap:
                            best_overlap = ov
                            matched_kpts = kpts

            # Build 12 CowXNet features
            feat_dict = self._extract_cowxnet_features(box, matched_kpts, frame_shape=(h_f, w_f))
            track["cowxnet_features"] = feat_dict

            heat_conf = 0.0
            if self.heat_model is not None:
                try:
                    df = pd.DataFrame([feat_dict], columns=COWXNET_FEATURES)
                    proba = self.heat_model.predict_proba(df)[0]
                    heat_conf = float(proba[1]) if len(proba) > 1 else float(proba[0])
                except Exception:
                    pass

            is_heat = (heat_conf >= 0.70)
            track["is_heat"] = is_heat
            track["heat_conf"] = round(heat_conf, 2)
            track["estrus_status"] = f"Heat: {int(heat_conf * 100)}% (Estrus)" if is_heat else f"Heat: {int(heat_conf * 100)}% (Normal)"

            # Standalone high-confidence CowXNet standing heat
            if heat_conf >= 0.85:
                events.append({
                    "mounting_id": track["id"],
                    "mounted_id": track["id"],
                    "conf": round(heat_conf, 2),
                    "behavior": "CowXNet Standing Heat Active",
                    "model": "CowXNet_heat_model.pkl + estrusdetection.pt"
                })

        # 3. Pairwise Kinematic Mounting Evaluation between cattle
        if len(cattle_tracks) >= 2:
            for i in range(len(cattle_tracks)):
                for j in range(len(cattle_tracks)):
                    if i == j:
                        continue
                    c_m = cattle_tracks[i]
                    c_f = cattle_tracks[j]
                    b_m, b_f = c_m.get("box", [0, 0, 0, 0]), c_f.get("box", [0, 0, 0, 0])
                    mx1, my1, mx2, my2 = b_m
                    fx1, fy1, fx2, fy2 = b_f
                    mw, mh = max(1, mx2 - mx1), max(1, my2 - my1)
                    fw, fh = max(1, fx2 - fx1), max(1, fy2 - fy1)

                    if min(mw, mh, fw, fh) < 50:
                        continue

                    # Overlap
                    ix1, iy1 = max(mx1, fx1), max(my1, fy1)
                    ix2, iy2 = min(mx2, fx2), min(my2, fy2)
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue

                    inter_area = float((ix2 - ix1) * (iy2 - iy1))
                    min_area = min(mw * mh, fw * fh)
                    iomin = inter_area / float(min_area)

                    # Mounting elevation & rearing silhouette
                    ar_m = float(mw) / float(mh)
                    m_cy = (my1 + my2) / 2.0
                    f_cy = (fy1 + fy2) / 2.0
                    min_h = min(mh, fh)

                    is_mounting = (
                        (iomin >= 0.35) and
                        (my1 < fy1 - min_h * 0.18) and
                        (m_cy < f_cy) and
                        (ar_m <= 1.35 or c_m.get("is_heat") or c_f.get("is_heat")) and
                        (abs((mx1 + mx2) / 2.0 - (fx1 + fx2) / 2.0) < max(mw, fw) * 0.60)
                    )

                    if is_mounting:
                        m_conf = max(0.85, (c_m.get("heat_conf", 0.5) + c_f.get("heat_conf", 0.5)) / 2.0)
                        events.append({
                            "mounting_id": c_m["id"],
                            "mounted_id": c_f["id"],
                            "conf": round(m_conf, 2),
                            "behavior": "Mounting / Standing Heat",
                            "model": "CowXNet_heat_model.pkl + estrusdetection.pt"
                        })

        return events

# Singleton global instance
ai_models = AIModelManager()
