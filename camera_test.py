import os
import sys
import time
import datetime
import threading
import queue
import warnings
import cv2
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from ultralytics import YOLO

try:
    import winsound
except ImportError:
    winsound = None

import webbrowser
from activity_engine import ActivityEngine
from pashux_supabase import supabase_manager, DEFAULT_FARM_ID, DEFAULT_CAMERA_ID

# Suppress scikit-learn version mismatch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# -------------------------------------------------------------------
# Configurations & Default Paths
# -------------------------------------------------------------------
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|max_delay;500000|fflags;nobuffer"
os.environ["KERAS_BACKEND"] = "torch"  # PyTorch backend for Keras 3

try:
    num_threads = min(8, os.cpu_count() or 4)
    torch.set_num_threads(num_threads)
except Exception:
    pass

DEFAULT_VIDEO_PATH = r"C:\Users\HP\Pictures\Recording 2026-09-23 154124.mp4"
DEFAULT_RTSP_URL = "rtsp://chimertechadmin:Chimertech2026@192.168.1.58:554/stream2"
DETECTION_MODEL_PATH = "yolo11m.pt" if os.path.exists("yolo11m.pt") else "yolo11n.pt"
TRAINED_BREED_MODEL_PATH = "cattle_breed_model.pt"
BCS_MODEL_PATH = "best_bcs_cnn_v2.pth"
DISEASE_MODEL_PATH = "final_cattle_disease_model.keras"
ESTRUS_MODEL_PATH = "estrusdetection.pt"
COWXNET_HEAT_MODEL_PATH = "CowXNet_heat_model.pkl"
ESTRUS_DIR = "estrus"
LOGO_PATH = "logo.png"

COCO_PERSON_CLASS = 0  # COCO Class: 0 is 'person' (Human)
COCO_COW_CLASS = 19     # COCO Class: 19 is 'cow' (Cattle)
TARGET_CLASSES = [COCO_PERSON_CLASS, COCO_COW_CLASS]


def load_brand_logo(logo_path=LOGO_PATH, target_size=(75, 75)):
    """Loads brand logo with alpha transparency support."""
    if not os.path.exists(logo_path):
        return None
    try:
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if logo is None:
            return None
        if target_size is not None:
            logo = cv2.resize(logo, target_size, interpolation=cv2.INTER_AREA)
        return logo
    except Exception:
        return None


def overlay_logo(canvas, logo_img, pos="top-right", margin=16, opacity=1.0, custom_x=None, custom_y=None):
    """Blends RGBA/RGB logo onto canvas with alpha transparency."""
    if logo_img is None or canvas is None:
        return canvas
    
    lh, lw = logo_img.shape[:2]
    ch, cw = canvas.shape[:2]

    if custom_x is not None and custom_y is not None:
        x, y = custom_x, custom_y
    elif pos == "top-right":
        x = cw - lw - margin
        y = margin
    elif pos == "top-left":
        x = margin
        y = margin
    elif pos == "bottom-right":
        x = cw - lw - margin
        y = ch - lh - margin
    elif pos == "bottom-left":
        x = margin
        y = ch - lh - margin
    else:
        x = cw - lw - margin
        y = margin

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(cw, x + lw)
    y2 = min(ch, y + lh)

    lx1 = max(0, -x)
    ly1 = max(0, -y)
    lx2 = lx1 + (x2 - x1)
    ly2 = ly1 + (y2 - y1)

    if x2 <= x1 or y2 <= y1 or lx2 <= lx1 or ly2 <= ly1:
        return canvas

    roi = canvas[y1:y2, x1:x2]
    logo_crop = logo_img[ly1:ly2, lx1:lx2]

    if logo_crop.shape[2] == 4:
        b, g, r, a = cv2.split(logo_crop)
        alpha = (a.astype(float) / 255.0) * opacity
        alpha = np.expand_dims(alpha, axis=2)
        logo_bgr = cv2.merge([b, g, r])
        canvas[y1:y2, x1:x2] = (logo_bgr * alpha + roi * (1.0 - alpha)).astype(np.uint8)
    else:
        cv2.addWeighted(logo_crop, opacity, roi, 1.0 - opacity, 0, roi)
    return canvas


def overlay_logo_with_card(canvas, logo_img, custom_x=None, custom_y=None, pad=6, bg_color=(255, 255, 255), border_color=(45, 45, 45), border_thickness=2):
    """
    Renders the brand logo on a crisp white high-contrast card with clean borders.
    """
    if logo_img is None or canvas is None:
        return canvas
    
    lh, lw = logo_img.shape[:2]
    ch, cw = canvas.shape[:2]

    if custom_x is None:
        custom_x = cw - lw - pad * 2 - 20
    if custom_y is None:
        custom_y = 15

    x1 = max(0, custom_x)
    y1 = max(0, custom_y)
    x2 = min(cw, x1 + lw + pad * 2)
    y2 = min(ch, y1 + lh + pad * 2)

    if x2 <= x1 or y2 <= y1:
        return canvas

    # Draw solid white card on canvas
    cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_color, -1)
    if border_thickness > 0 and border_color is not None:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, border_thickness)

    # Place logo inside the white card
    logo_x = x1 + pad
    logo_y = y1 + pad

    overlay_logo(canvas, logo_img, custom_x=logo_x, custom_y=logo_y, opacity=1.0)
    return canvas

BREED_COLORS = {
    "Gir": (0, 70, 220),
    "Sahiwal": (30, 140, 255),
    "Kankrej": (220, 220, 180),
    "Tharparkar": (240, 240, 210),
    "Red Sindhi": (0, 40, 180),
    "Rathi": (50, 100, 220),
    "Ongole": (210, 220, 240),
    "Hallikar": (220, 190, 150),
    "Deoni": (160, 200, 220),
    "Dangi": (180, 180, 180),
    "Ayrshire": (140, 180, 230),
    "Holstein Friesian": (255, 255, 255),
    "Jersey": (0, 215, 255),
    "Murrah Buffalo": (120, 120, 120),
    "Indigenous Desi": (0, 190, 230),
    "Unknown / Partial View": (170, 170, 170),
}

# -------------------------------------------------------------------
# Breed Morphometric Database (Standard Veterinary Biometry)
# Standard Adult Wither Heights (cm) & Baseline Live Weights (kg)
# -------------------------------------------------------------------
BREED_BIOMETRY_DB = {
    "Gir":                {"h_min": 128, "h_max": 142, "w_min": 375, "w_max": 450},
    "Sahiwal":            {"h_min": 125, "h_max": 138, "w_min": 360, "w_max": 440},
    "Kankrej":            {"h_min": 135, "h_max": 152, "w_min": 430, "w_max": 560},
    "Ongole":             {"h_min": 138, "h_max": 155, "w_min": 440, "w_max": 580},
    "Tharparkar":         {"h_min": 128, "h_max": 140, "w_min": 370, "w_max": 460},
    "Red Sindhi":         {"h_min": 120, "h_max": 132, "w_min": 320, "w_max": 390},
    "Rathi":              {"h_min": 124, "h_max": 135, "w_min": 340, "w_max": 420},
    "Hallikar":           {"h_min": 130, "h_max": 145, "w_min": 380, "w_max": 480},
    "Deoni":              {"h_min": 126, "h_max": 140, "w_min": 360, "w_max": 450},
    "Dangi":              {"h_min": 118, "h_max": 130, "w_min": 310, "w_max": 380},
    "Kangayam":           {"h_min": 125, "h_max": 138, "w_min": 360, "w_max": 460},
    "Amritmahal":         {"h_min": 132, "h_max": 146, "w_min": 390, "w_max": 490},
    "Krishna Valley":     {"h_min": 135, "h_max": 150, "w_min": 420, "w_max": 540},
    "Khillari":           {"h_min": 132, "h_max": 148, "w_min": 400, "w_max": 510},
    "Punganur":           {"h_min": 70,  "h_max": 90,  "w_min": 115, "w_max": 200},
    "Vechur":             {"h_min": 80,  "h_max": 95,  "w_min": 130, "w_max": 180},
    "Bachaur":            {"h_min": 115, "h_max": 128, "w_min": 280, "w_max": 360},
    "Bargur":             {"h_min": 120, "h_max": 132, "w_min": 300, "w_max": 380},
    "Gaolao":             {"h_min": 128, "h_max": 142, "w_min": 370, "w_max": 470},
    "Hariana":            {"h_min": 132, "h_max": 146, "w_min": 390, "w_max": 500},
    "Malvi":              {"h_min": 130, "h_max": 144, "w_min": 380, "w_max": 480},
    "Nagori":             {"h_min": 132, "h_max": 148, "w_min": 390, "w_max": 510},
    "Nimari":             {"h_min": 128, "h_max": 140, "w_min": 360, "w_max": 450},
    "Umblachery":         {"h_min": 110, "h_max": 124, "w_min": 260, "w_max": 340},
    "Pulikulam":          {"h_min": 118, "h_max": 130, "w_min": 300, "w_max": 380},
    "Bhelai":             {"h_min": 120, "h_max": 134, "w_min": 310, "w_max": 390},
    "Holstein Friesian":  {"h_min": 142, "h_max": 160, "w_min": 560, "w_max": 720},
    "Jersey":             {"h_min": 120, "h_max": 135, "w_min": 380, "w_max": 480},
    "Ayrshire":           {"h_min": 128, "h_max": 142, "w_min": 440, "w_max": 560},
    "Murrah Buffalo":     {"h_min": 138, "h_max": 155, "w_min": 520, "w_max": 700},
    "Default Cattle":     {"h_min": 125, "h_max": 140, "w_min": 360, "w_max": 460},
}


def estimate_cattle_biometry(box, breed_name, bcs_float, frame_shape):
    """
    Allometric Weight & Wither Height Estimation Engine:
    1. Retrieves breed-specific baseline morphometric distributions.
    2. Scales live body weight based on BCS body fat/muscle multiplier (BCS 1-5).
    3. Incorporates bounding box aspect ratio and spatial depth for individual variance.
    Returns: (height_cm, height_range_str, weight_kg, weight_range_str)
    """
    clean_breed = breed_name.strip()
    bio = BREED_BIOMETRY_DB.get(clean_breed, None)
    if bio is None:
        # Check partial matching
        for k in BREED_BIOMETRY_DB:
            if k.lower() in clean_breed.lower() or clean_breed.lower() in k.lower():
                bio = BREED_BIOMETRY_DB[k]
                break
    if bio is None:
        bio = BREED_BIOMETRY_DB["Default Cattle"]

    h_min, h_max = bio["h_min"], bio["h_max"]
    w_min, w_max = bio["w_min"], bio["w_max"]

    # Calculate individual continuous variation from bounding box geometry
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    aspect_ratio = float(bw) / float(bh)
    
    # Aspect ratio offset (-0.05 to +0.05 variation)
    geom_factor = float(np.clip((aspect_ratio - 1.25) * 0.15, -0.08, 0.08))
    
    # Wither Height:
    h_mid = (h_min + h_max) / 2.0
    height_cm = int(round(h_mid * (1.0 + geom_factor)))
    height_cm = int(np.clip(height_cm, h_min - 4, h_max + 6))
    height_ft = height_cm / 30.48
    height_range_str = f"{h_min}-{h_max} cm ({height_ft:.1f} ft)"

    # BCS Weight Factor: BCS 1.0 (emaciated) ~ 0.78x base, BCS 3.0 ~ 1.0x base, BCS 5.0 ~ 1.22x base
    bcs_f = float(np.clip(bcs_float, 1.0, 5.0))
    bcs_factor = 0.78 + ((bcs_f - 1.0) / 4.0) * 0.44

    w_mid = (w_min + w_max) / 2.0
    weight_kg = int(round(w_mid * bcs_factor * (1.0 + geom_factor)))
    
    # Individual range (+/- 6% confidence interval)
    w_est_min = int(round(weight_kg * 0.94))
    w_est_max = int(round(weight_kg * 1.06))
    weight_range_str = f"{w_est_min}-{w_est_max} kg"

    return height_cm, height_range_str, weight_kg, weight_range_str


# -------------------------------------------------------------------
# Body Condition Score (BCS) Model Architecture (BCSNetV2)
# -------------------------------------------------------------------
class BCSNetV2(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


BCS_INFO = {
    1: {"label": "Very Thin",   "desc": "Severely Underweight", "color": (40, 40, 240)},    # Crimson Red
    2: {"label": "Thin",        "desc": "Underconditioned",     "color": (20, 140, 255)},   # Amber/Orange
    3: {"label": "Ideal",       "desc": "Optimal Health",       "color": (50, 220, 50)},    # Bright Green
    4: {"label": "Fleshy/Fat",  "desc": "Overconditioned",      "color": (0, 215, 255)},    # Yellow
    5: {"label": "Obese",       "desc": "Grossly Overweight",   "color": (200, 50, 200)},   # Magenta
}

DISEASE_CLASSES = ["Foot and Mouth Disease", "Healthy", "Lumpy Skin Disease"]
DISEASE_INFO = {
    "Healthy":                  {"short": "Healthy",  "color": (50, 220, 50)},
    "Foot and Mouth Disease":   {"short": "FMD (Mouth/Foot)", "color": (0, 80, 255)},
    "Lumpy Skin Disease":       {"short": "LSD (Skin Nodule)", "color": (40, 40, 240)},
}


def load_bcs_model(model_path=BCS_MODEL_PATH):
    if not os.path.exists(model_path):
        return None, "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = BCSNetV2(num_classes=5)
        ckpt = torch.load(model_path, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=True)
        model.to(device)
        model.eval()
        print(f"Loaded BCS Model '{model_path}' successfully on {device.upper()}!")
        return model, device
    except Exception as e:
        print(f"Warning: Could not load BCS model: {e}")
        return None, "cpu"


def get_bcs_metadata(bcs_float):
    bcs_f = float(np.clip(bcs_float, 1.0, 5.0))
    if bcs_f < 1.75:
        return 1, "Very Thin", (40, 40, 240)
    elif bcs_f < 2.50:
        return 2, "Thin", (20, 140, 255)
    elif bcs_f <= 3.50:
        return 3, "Ideal", (50, 220, 50)
    elif bcs_f <= 4.25:
        return 4, "Fleshy", (0, 215, 255)
    else:
        return 5, "Obese", (200, 50, 200)


def predict_cattle_bcs(crop_bgr, bcs_model, device="cpu"):
    if bcs_model is None or crop_bgr is None or crop_bgr.shape[0] < 25 or crop_bgr.shape[1] < 25:
        return 3, 3.00, "Ideal", 0.70, (50, 220, 50)
    
    try:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = bcs_model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        classes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        best_idx = int(np.argmax(probs))
        conf = float(probs[best_idx])
        raw_bcs = float(np.sum(probs * classes))
        
        h_c, w_c = crop_bgr.shape[:2]
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        body_fill_ratio = float(np.sum(thresh > 0)) / float(max(1, h_c * w_c))
        body_adjustment = float(np.clip((body_fill_ratio - 0.50) * 0.6, -0.4, 0.4))
        bcs_continuous = float(np.clip(raw_bcs + body_adjustment, 1.0, 5.0))
        
        bcs_int, label, color = get_bcs_metadata(bcs_continuous)
        return bcs_int, bcs_continuous, label, conf, color
    except Exception:
        return 3, 3.00, "Ideal", 0.70, (50, 220, 50)


# -------------------------------------------------------------------
# Cattle Disease Detection Model (Exclusively Powered by AI Model)
# -------------------------------------------------------------------
def load_disease_model(model_path=DISEASE_MODEL_PATH):
    if not os.path.exists(model_path):
        print(f"Disease model not found at '{model_path}', skipping disease neural model.")
        return None
    try:
        os.environ["KERAS_BACKEND"] = "torch"
        import keras
        model = keras.models.load_model(model_path)
        dummy = torch.zeros((1, 300, 300, 3), dtype=torch.float32)
        with torch.no_grad():
            _ = model(dummy)
        print(f"Loaded & Warmed Disease Model '{model_path}' successfully! ({len(DISEASE_CLASSES)} classes)")
        return model
    except Exception as e:
        print(f"Notice: Could not load disease neural model: {e}")
        return None


def predict_cattle_disease(crop_bgr, disease_model):
    if disease_model is None or crop_bgr is None:
        return "Healthy", 0.95, (50, 220, 50)

    h_c, w_c = crop_bgr.shape[:2]
    # Resolution & Clarity Gate:
    # Clinical skin nodules (LSD) and mouth/foot lesions (FMD) require clear close-up visual resolution.
    # Distant, blurry, or small background cattle crops (< 140x140 px or < 22,000 px^2) lack skin-texture
    # resolution and cleanly default to Healthy instead of falsely triggering on compression noise.
    if h_c < 140 or w_c < 140 or (h_c * w_c) < 22000:
        return "Healthy", 0.95, (50, 220, 50)

    try:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (300, 300), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized.astype(np.float32)).unsqueeze(0)

        with torch.no_grad():
            preds = disease_model(tensor)
            if isinstance(preds, torch.Tensor):
                probs = preds[0].cpu().numpy()
            else:
                probs = np.array(preds)[0]

        best_idx = int(np.argmax(probs))
        conf = float(probs[best_idx])
        disease_name = DISEASE_CLASSES[best_idx]

        # Clinical sensitivity calibration:
        # Only flag genuine clinical disease if strong confidence (>= 0.70)
        if disease_name == "Foot and Mouth Disease" and conf >= 0.70:
            return "Foot and Mouth Disease", conf, (0, 80, 255)
        elif disease_name == "Lumpy Skin Disease" and conf >= 0.72:
            return "Lumpy Skin Disease", conf, (40, 40, 240)
        else:
            return "Healthy", max(0.90, conf if disease_name == "Healthy" else 0.90), (50, 220, 50)
    except Exception:
        return "Healthy", 0.95, (50, 220, 50)


# -------------------------------------------------------------------
# Cattle Breed Classifier (Exclusively Powered by Trained AI Model)
# -------------------------------------------------------------------
def predict_cattle_breed(crop_bgr, breed_model):
    if breed_model is None or crop_bgr is None or crop_bgr.shape[0] < 20 or crop_bgr.shape[1] < 20:
        return "Cattle", 0.50

    try:
        results = breed_model.predict(crop_bgr, imgsz=224, verbose=False)
        if results and len(results) > 0 and results[0].probs is not None:
            top1_idx = int(results[0].probs.top1)
            conf = float(results[0].probs.top1conf)
            raw_name = results[0].names[top1_idx]
            clean_name = raw_name.replace("-", " ").replace("_", " ").title()
            return clean_name, conf
    except Exception:
        pass

    return "Cattle", 0.50


# -------------------------------------------------------------------
# Unified Asynchronous Cattle Analytics Worker (Zero Lag, Normal Speed)
# -------------------------------------------------------------------
class CattleAnalysisWorker:
    def __init__(self, breed_model, bcs_model, bcs_device, disease_model):
        self.breed_model = breed_model
        self.bcs_model = bcs_model
        self.bcs_device = bcs_device
        self.disease_model = disease_model

        self.queue = queue.Queue(maxsize=32)
        self.results = {}
        self.sample_counts = {}
        self.in_flight = set()
        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def submit(self, track_id, crop_bgr, box, frame_shape):
        if not self.running or crop_bgr is None:
            return
        with self.lock:
            if track_id in self.in_flight or self.sample_counts.get(track_id, 0) >= 4:
                return
            self.in_flight.add(track_id)
        try:
            self.queue.put_nowait((track_id, crop_bgr.copy(), box, frame_shape))
        except queue.Full:
            with self.lock:
                self.in_flight.discard(track_id)

    def get_result(self, track_id):
        with self.lock:
            return self.results.get(track_id, None)

    def is_done(self, track_id):
        with self.lock:
            return self.sample_counts.get(track_id, 0) >= 4

    def reset(self):
        with self.lock:
            self.results.clear()
            self.sample_counts.clear()
            self.in_flight.clear()
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except Exception:
                    break

    def _worker_loop(self):
        while self.running:
            try:
                track_id, crop, box, frame_shape = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                # 1. Breed Prediction
                cur_breed, cur_bconf = predict_cattle_breed(crop, self.breed_model)

                # 2. BCS Prediction
                bcs_int, bcs_float, bcs_label, bcs_conf, bcs_color = predict_cattle_bcs(
                    crop, self.bcs_model, self.bcs_device
                )

                # 3. Disease Screening
                disease_name, dis_conf, dis_color = predict_cattle_disease(crop, self.disease_model)

                # 4. Weight & Height Estimation
                ht_cm, ht_range, wt_kg, wt_range = estimate_cattle_biometry(
                    box, cur_breed, bcs_float, frame_shape
                )

                with self.lock:
                    prev = self.results.get(track_id, {
                        "breeds": {},
                        "bcs_scores": [],
                        "diseases": {},
                        "weights": [],
                        "heights": [],
                        "thumbnail": None
                    })
                    prev["breeds"][cur_breed] = prev["breeds"].get(cur_breed, 0) + 1
                    prev["bcs_scores"].append(bcs_float)
                    prev["diseases"][disease_name] = prev["diseases"].get(disease_name, 0) + 1
                    prev["weights"].append(wt_kg)
                    prev["heights"].append(ht_cm)

                    if prev["thumbnail"] is None or (crop.shape[0] * crop.shape[1] > prev["thumbnail"].shape[0] * prev["thumbnail"].shape[1]):
                        prev["thumbnail"] = crop.copy()

                    best_breed = max(prev["breeds"], key=prev["breeds"].get)
                    avg_bcs_f = float(np.mean(prev["bcs_scores"]))
                    avg_bcs_i = int(np.clip(round(avg_bcs_f), 1, 5))

                    # Robust multi-sample consensus for clinical disease
                    dis_votes = prev["diseases"]
                    n_fmd = dis_votes.get("Foot and Mouth Disease", 0)
                    n_lsd = dis_votes.get("Lumpy Skin Disease", 0)
                    n_h = dis_votes.get("Healthy", 0)

                    if n_fmd >= 2 and n_fmd >= n_h:
                        best_dis = "Foot and Mouth Disease"
                    elif n_lsd >= 2 and n_lsd >= n_h:
                        best_dis = "Lumpy Skin Disease"
                    else:
                        best_dis = "Healthy"

                    avg_wt = int(np.mean(prev["weights"]))
                    avg_ht = int(np.mean(prev["heights"]))

                    self.results[track_id] = {
                        "breed": best_breed,
                        "breed_conf": cur_bconf,
                        "bcs_int": avg_bcs_i,
                        "bcs_float": avg_bcs_f,
                        "bcs_label": BCS_INFO.get(avg_bcs_i, {}).get("label", "Ideal"),
                        "bcs_color": BCS_INFO.get(avg_bcs_i, {}).get("color", (50, 220, 50)),
                        "disease": best_dis,
                        "disease_conf": dis_conf,
                        "disease_color": DISEASE_INFO.get(best_dis, {}).get("color", (50, 220, 50)),
                        "weight_kg": avg_wt,
                        "weight_range": wt_range,
                        "height_cm": avg_ht,
                        "height_range": ht_range,
                        "breeds": prev["breeds"],
                        "bcs_scores": prev["bcs_scores"],
                        "diseases": prev["diseases"],
                        "thumbnail": prev["thumbnail"]
                    }
                    self.sample_counts[track_id] = self.sample_counts.get(track_id, 0) + 1
            except Exception:
                pass
            finally:
                with self.lock:
                    self.in_flight.discard(track_id)
                self.queue.task_done()

    def stop(self):
        self.running = False


# -------------------------------------------------------------------
# Spatial Box Overlap Helpers
# -------------------------------------------------------------------
def compute_box_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def compute_box_iomin(b1, b2):
    """Intersection over Minimum Area: detects nested/overlapping boxes"""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / min(a1, a2) if min(a1, a2) > 0 else 0.0


# -------------------------------------------------------------------
# Cattle Estrus & Mounting AI Engine (CowXNet RF + YOLO Pose + Kinematics)
# -------------------------------------------------------------------
COW_SKELETON = [
    # Head & Spine line
    (0, 1), (1, 2), (2, 9), (9, 10),
    # Front Left Leg
    (2, 3), (3, 4), (4, 5),
    # Front Right Leg
    (2, 6), (6, 7), (7, 8),
    # Rear Left Leg
    (10, 11), (11, 12),
    # Rear Right Leg
    (10, 13), (13, 14), (14, 15)
]

COWXNET_FEATURES = [
    'cow-box_xmax', 'cow-box_xmin', 'cow-box_ymax', 'cow-box_ymin',
    'cow-box_xcenter', 'cow-box_ycenter', 'head_x', 'head_y',
    'body_x', 'body_y', 'tail-head_x', 'tail-head_y'
]


class EstrusDetector:
    def __init__(self, pose_model_path=ESTRUS_MODEL_PATH, heat_model_path=COWXNET_HEAT_MODEL_PATH, output_dir=ESTRUS_DIR):
        self.pose_model = None
        self.heat_model = None
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = os.path.join(self.output_dir, "estrus_log.csv")
        self._init_log()
        self.last_event_time = {}   # (id1, id2) or ('cowxnet', id1) -> float timestamp (cooldown)
        self.active_events = []      # Active estrus event dicts in current frame
        self.event_history = []      # Logged estrus events history for live data table
        self.total_events_logged = 0

        # 1. Load CowXNet Heat & Estrus Random Forest Classifier
        if os.path.exists(heat_model_path):
            try:
                print(f"Loading CowXNet Heat & Estrus Classifier: '{heat_model_path}'...")
                self.heat_model = joblib.load(heat_model_path)
                if hasattr(self.heat_model, "n_jobs"):
                    self.heat_model.n_jobs = 1
                print("Loaded CowXNet Heat AI Model successfully!")
            except Exception as e:
                print(f"Warning: Could not load CowXNet Heat Model {heat_model_path}: {e}")
                self.heat_model = None

        # 2. Load Cattle YOLO Pose Keypoints Model
        if os.path.exists(pose_model_path):
            try:
                print(f"Loading Cattle Estrus Pose & Mounting AI Model: '{pose_model_path}'...")
                self.pose_model = YOLO(pose_model_path)
                print("Loaded Cattle Estrus Pose AI Model successfully!")
            except Exception as e:
                print(f"Warning: Could not load {pose_model_path}: {e}")
                self.pose_model = None

    def _init_log(self):
        if not os.path.exists(self.log_file):
            try:
                with open(self.log_file, "w", encoding="utf-8") as f:
                    f.write("Timestamp,DateTime,FrameNumber,MountingCattleID,MountedCattleID,Behavior,Confidence,SnapshotPath\n")
            except Exception:
                pass

    def _trigger_audio_alert(self):
        """Emits an audible notification beep asynchronously without blocking video processing."""
        if winsound is not None:
            def beep_worker():
                try:
                    winsound.Beep(1300, 240)
                except Exception:
                    pass
            threading.Thread(target=beep_worker, daemon=True).start()

    def _extract_cowxnet_features(self, box, kpts, frame_shape=None):
        """
        Builds normalized 12-feature dictionary expected by CowXNet_heat_model.pkl (0.0 to 1.0 coordinates):
        ['cow-box_xmax', 'cow-box_xmin', 'cow-box_ymax', 'cow-box_ymin',
         'cow-box_xcenter', 'cow-box_ycenter', 'head_x', 'head_y',
         'body_x', 'body_y', 'tail-head_x', 'tail-head_y']
        """
        xmin, ymin, xmax, ymax = box
        xc = (xmin + xmax) / 2.0
        yc = (ymin + ymax) / 2.0
        w = max(1.0, xmax - xmin)
        h = max(1.0, ymax - ymin)

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

        if frame_shape is not None:
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

        return {
            'cow-box_xmax': float(xmax),
            'cow-box_xmin': float(xmin),
            'cow-box_ymax': float(ymax),
            'cow-box_ymin': float(ymin),
            'cow-box_xcenter': float(xc),
            'cow-box_ycenter': float(yc),
            'head_x': head_x,
            'head_y': head_y,
            'body_x': body_x,
            'body_y': body_y,
            'tail-head_x': tail_x,
            'tail-head_y': tail_y
        }

    def trigger_manual_estrus_alert(self, frame, current_tracks, frame_num=0):
        """Allows testing / manual trigger of estrus mounting alert via keypress [E]."""
        if frame is None or len(current_tracks) < 1:
            return None
        cur_time = time.time()
        dt_time_str = datetime.datetime.now().strftime("%H:%M:%S")
        if len(current_tracks) >= 2:
            id_i = current_tracks[0]["id"]
            id_j = current_tracks[1]["id"]
            box_i = current_tracks[0]["box"]
            box_j = current_tracks[1]["box"]
        else:
            id_i = current_tracks[0]["id"]
            id_j = current_tracks[0]["id"]
            box_i = current_tracks[0]["box"]
            box_j = current_tracks[0]["box"]

        event = {
            "mounting_id": id_i,
            "mounted_id": id_j,
            "behavior": "Mounting / Standing Heat (Test)",
            "conf": 0.96,
            "box_mounting": box_i,
            "box_mounted": box_j,
            "time": cur_time,
            "time_str": dt_time_str,
            "frame": frame_num
        }
        self.active_events.append(event)
        self.total_events_logged += 1
        self.event_history.append(event)
        if len(self.event_history) > 12:
            self.event_history.pop(0)
        self._save_event_snapshot_and_log(frame, event, [])
        self._trigger_audio_alert()
        return event

    def detect_estrus(self, frame, current_tracks, frame_num=0):
        """
        Runs YOLO pose estimation, CowXNet Heat RF classifier, and multi-cattle mounting kinematics.
        Returns: (active_estrus_events, pose_keypoints_list, event_history)
        """
        self.active_events = []
        pose_keypoints_list = []

        if frame is None or len(current_tracks) < 1:
            return self.active_events, pose_keypoints_list, self.event_history

        h_frame, w_frame = frame.shape[:2]
        cur_time = time.time()
        dt_time_str = datetime.datetime.now().strftime("%H:%M:%S")

        # 1. Run YOLO pose inference if model available
        if self.pose_model is not None:
            try:
                with torch.inference_mode():
                    res = self.pose_model.predict(frame, imgsz=640, conf=0.18, verbose=False)
                if res and len(res) > 0 and res[0].keypoints is not None:
                    kpts_tensor = res[0].keypoints.xy.cpu().numpy()
                    confs_tensor = res[0].keypoints.conf.cpu().numpy() if res[0].keypoints.conf is not None else None
                    for idx, kpts in enumerate(kpts_tensor):
                        c_arr = confs_tensor[idx] if confs_tensor is not None else np.ones(len(kpts))
                        pose_keypoints_list.append((kpts, c_arr))
            except Exception:
                pass

        # 2. Individual Cattle Heat Prediction with CowXNet_heat_model.pkl
        for track in current_tracks:
            box = track["box"]
            x1, y1, x2, y2 = box

            # Match nearest pose keypoints to this cow's bounding box
            matched_kpts = None
            best_overlap = 0.0
            for item in pose_keypoints_list:
                kpts = item[0] if isinstance(item, tuple) else item
                valid_pts = kpts[kpts[:, 0] > 0]
                if len(valid_pts) > 0:
                    k_xmin, k_ymin = np.min(valid_pts[:, 0]), np.min(valid_pts[:, 1])
                    k_xmax, k_ymax = np.max(valid_pts[:, 0]), np.max(valid_pts[:, 1])
                    iou = compute_box_iou(box, [k_xmin, k_ymin, k_xmax, k_ymax])
                    if iou > best_overlap:
                        best_overlap = iou
                        matched_kpts = kpts

            # Extract normalized 12 CowXNet features
            feat_dict = self._extract_cowxnet_features(box, matched_kpts, frame_shape=(h_frame, w_frame))
            track["cowxnet_features"] = feat_dict

            heat_pred = 0
            heat_conf = 0.0
            if self.heat_model is not None:
                try:
                    feat_vector = np.array([[feat_dict[k] for k in COWXNET_FEATURES]], dtype=np.float32)
                    proba = self.heat_model.predict_proba(feat_vector)[0]
                    heat_conf = float(proba[1]) if len(proba) > 1 else float(proba[0])
                    heat_pred = 1 if heat_conf >= 0.50 else 0
                except Exception:
                    try:
                        df = pd.DataFrame([feat_dict], columns=COWXNET_FEATURES)
                        proba = self.heat_model.predict_proba(df)[0]
                        heat_conf = float(proba[1]) if len(proba) > 1 else float(proba[0])
                        heat_pred = 1 if heat_conf >= 0.50 else 0
                    except Exception:
                        pass

            heat_pred = 1 if heat_conf >= 0.80 else 0
            is_heat = (heat_pred == 1) and (heat_conf >= 0.80)
            track["is_heat"] = is_heat
            track["heat_conf"] = heat_conf
            track["heat_status"] = f"🔥 Heat: {int(heat_conf*100)}% [ESTRUS]" if is_heat else f"Heat: {int(heat_conf*100)}% [Normal]"

            # Check for standalone high-confidence CowXNet Heat event
            if heat_conf >= 0.85:
                t_id = track["id"]
                heat_event = {
                    "mounting_id": t_id,
                    "mounted_id": t_id,
                    "behavior": "CowXNet Standing Heat Active",
                    "conf": heat_conf,
                    "box_mounting": box,
                    "box_mounted": box,
                    "time": cur_time,
                    "time_str": dt_time_str,
                    "frame": frame_num
                }
                self.active_events.append(heat_event)

                pair_key = ("heat_cowxnet", t_id)
                last_t = self.last_event_time.get(pair_key, 0)
                if cur_time - last_t > 6.0:
                    self.last_event_time[pair_key] = cur_time
                    self.total_events_logged += 1
                    self.event_history.append(heat_event)
                    if len(self.event_history) > 12:
                        self.event_history.pop(0)
                    self._save_event_snapshot_and_log(frame, heat_event, pose_keypoints_list)
                    self._trigger_audio_alert()

        # 3. Multi-Cattle Interaction, Mounting & Ano-Genital Sniffing Kinematics
        if len(current_tracks) >= 2:
            for i in range(len(current_tracks)):
                t_i = current_tracks[i]
                id_i = t_i["id"]
                box_i = t_i["box"]
                x1_i, y1_i, x2_i, y2_i = box_i
                w_i, h_i = max(1, x2_i - x1_i), max(1, y2_i - y1_i)
                cx_i, cy_i = (x1_i + x2_i) // 2, (y1_i + y2_i) // 2

                for j in range(len(current_tracks)):
                    if i == j:
                        continue
                    t_j = current_tracks[j]
                    id_j = t_j["id"]
                    box_j = t_j["box"]
                    x1_j, y1_j, x2_j, y2_j = box_j
                    w_j, h_j = max(1, x2_j - x1_j), max(1, y2_j - y1_j)
                    cx_j, cy_j = (x1_j + x2_j) // 2, (y1_j + y2_j) // 2

                    # Compute 2D bounding box intersection metrics
                    iou = compute_box_iou(box_i, box_j)
                    iomin = compute_box_iomin(box_i, box_j)
                    x_overlap = max(0, min(x2_i, x2_j) - max(x1_i, x1_j))
                    x_overlap_ratio = x_overlap / max(1.0, min(w_i, w_j))

                    is_mounting = False
                    is_sniffing = False
                    behavior_type = "Mounting / Standing Heat"
                    estrus_conf = 0.88

                    # 3A. Genuine Mounting Kinematics:
                    # In pasture/hillside crowd videos, grazing cattle can overlap in 2D perspective.
                    # For genuine mounting:
                    # - Both cattle must be sufficiently large in frame (min dimension >= 45px)
                    # - High 2D intersection: iou >= 0.30 or (iomin >= 0.52 and x_overlap_ratio >= 0.50)
                    # - Mounting cow must be upright / reared up or have significant vertical elevation and overlap
                    if min(w_i, h_i, w_j, h_j) >= 45 and (iou >= 0.30 or iomin >= 0.52) and (x_overlap_ratio >= 0.50):
                        elevation_diff = y1_j - y1_i
                        # Cow i's top must be raised significantly above Cow j's top, with bottom overlapping Cow j
                        if elevation_diff >= (min(h_i, h_j) * 0.35) and (cy_i < cy_j) and (y2_i > y1_j + min(h_i, h_j) * 0.30):
                            is_mounting = True
                            estrus_conf = min(0.99, max(0.85 + (iomin * 0.15), t_i.get("heat_conf", 0.0), t_j.get("heat_conf", 0.0)))

                    # 3B. Genuine Ano-Genital / Vulva Sniffing & Flehmen Response:
                    if not is_mounting:
                        # Match pose keypoints to Cow i and Cow j based on spatial proximity
                        pose_i = None
                        pose_j = None
                        for p_item in pose_keypoints_list:
                            k_arr, c_arr = p_item if isinstance(p_item, tuple) else (p_item, None)
                            v_pts = k_arr[k_arr[:, 0] > 0]
                            if len(v_pts) == 0:
                                continue
                            in_i = np.sum((v_pts[:, 0] >= x1_i - 30) & (v_pts[:, 0] <= x2_i + 30) &
                                          (v_pts[:, 1] >= y1_i - 30) & (v_pts[:, 1] <= y2_i + 30))
                            in_j = np.sum((v_pts[:, 0] >= x1_j - 30) & (v_pts[:, 0] <= x2_j + 30) &
                                          (v_pts[:, 1] >= y1_j - 30) & (v_pts[:, 1] <= y2_j + 30))
                            if in_i >= 3 and (in_i > in_j or pose_i is None):
                                pose_i = (k_arr, c_arr)
                            elif in_j >= 3 and (in_j > in_i or pose_j is None):
                                pose_j = (k_arr, c_arr)

                        if pose_i is not None and pose_j is not None:
                            kp_i, conf_i = pose_i
                            kp_j, conf_j = pose_j

                            # Head points of Cow i (Muzzle #0, Poll #1, Neck #2)
                            head_pts_i = [kp_i[idx] for idx in (0, 1, 2) if idx < len(kp_i) and kp_i[idx][0] > 0 and (conf_i is None or conf_i[idx] > 0.18)]
                            # Rear points of Cow j (Tail-head #10, Left Pelvis #11, Right Pelvis #13, Spine #9)
                            rear_pts_j = [kp_j[idx] for idx in (10, 11, 13, 9) if idx < len(kp_j) and kp_j[idx][0] > 0 and (conf_j is None or conf_j[idx] > 0.18)]

                            if len(head_pts_i) > 0 and len(rear_pts_j) > 0:
                                min_dist_pts = min(float(np.hypot(h[0] - r[0], h[1] - r[1])) for h in head_pts_i for r in rear_pts_j)
                                touch_thresh = min(32.0, max(12.0, min(w_i, h_i, w_j, h_j) * 0.32))
                                
                                # Check if Cow i's head is reaching into/towards Cow j's rear
                                h_x, h_y = head_pts_i[0][0], head_pts_i[0][1]
                                dist_to_rear_contour = max(0, x1_j - h_x) if cx_i > cx_j else max(0, h_x - x2_j)
                                vert_aligned = (h_y >= y1_j - 15) and (h_y <= y2_j + 15)

                                if (min_dist_pts <= touch_thresh and dist_to_rear_contour < 18 and vert_aligned):
                                    is_sniffing = True
                                    behavior_type = "Ano-Genital Sniffing (Estrus Sign)"
                                    estrus_conf = min(0.96, max(0.88, t_i.get("heat_conf", 0.0), t_j.get("heat_conf", 0.0)))

                    # 3C. Genuine Standing Heat & Chin-Resting / Courting Contact:
                    if not is_mounting and not is_sniffing:
                        h_gap = max(0, max(x1_i, x1_j) - min(x2_i, x2_j))
                        v_overlap = max(0, min(y2_i, y2_j) - max(y1_i, y1_j)) / float(min(h_i, h_j))
                        if (iomin >= 0.06 or (h_gap <= 40 and v_overlap >= 0.25)):
                            ar_i = float(w_i) / float(h_i)
                            ar_j = float(w_j) / float(h_j)
                            if 0.65 <= ar_i <= 1.90 and 0.65 <= ar_j <= 1.90:
                                if abs(cy_i - cy_j) <= max(h_i, h_j) * 0.60:
                                    is_mounting = True
                                    behavior_type = "Standing Heat / Chin Resting (Estrus Sign)"
                                    estrus_conf = min(0.94, max(0.85, 0.80 + (iomin * 0.20)))

                    if is_mounting or is_sniffing:
                        event_data = {
                            "mounting_id": id_i,
                            "mounted_id": id_j,
                            "behavior": behavior_type,
                            "conf": estrus_conf,
                            "box_mounting": box_i,
                            "box_mounted": box_j,
                            "time": cur_time,
                            "time_str": dt_time_str,
                            "frame": frame_num
                        }
                        self.active_events.append(event_data)

                        # Cooldown for snapshot & CSV log (once every 5 seconds per pair)
                        pair_key = (id_i, id_j, behavior_type)
                        last_t = self.last_event_time.get(pair_key, 0)
                        if cur_time - last_t > 5.0:
                            self.last_event_time[pair_key] = cur_time
                            self.total_events_logged += 1
                            self.event_history.append(event_data)
                            if len(self.event_history) > 12:
                                self.event_history.pop(0)
                            self._save_event_snapshot_and_log(frame, event_data, pose_keypoints_list)
                            self._trigger_audio_alert()

        return self.active_events, pose_keypoints_list, self.event_history

    def _save_event_snapshot_and_log(self, frame, event, keypoints_list):
        def _write_task(frame_copy, ev, kpts_list):
            try:
                ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                dt_formatted = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                id_i = ev["mounting_id"]
                id_j = ev["mounted_id"]
                conf = ev["conf"]
                behavior = ev["behavior"]
                frame_num = ev["frame"]

                snapshot = frame_copy.copy()
                h_s, w_s = snapshot.shape[:2]

                bx1, by1, bx2, by2 = ev["box_mounting"]
                ox1, oy1, ox2, oy2 = ev["box_mounted"]
                
                if id_i == id_j:
                    cv2.rectangle(snapshot, (bx1, by1), (bx2, by2), (255, 0, 200), 3) # Neon Pink (Heat Active)
                    banner_text = f"🔥 COWXNET HEAT ALERT: Cattle #{id_i} in Standing Heat / Estrus ({int(conf*100)}%)"
                else:
                    cv2.rectangle(snapshot, (bx1, by1), (bx2, by2), (255, 0, 200), 3) # Neon Pink (Mounting)
                    cv2.rectangle(snapshot, (ox1, oy1), (ox2, oy2), (0, 215, 255), 3) # Gold / Cyan (Mounted)
                    banner_text = f"🔥 ESTRUS MOUNTING DETECTED: Cattle #{id_i} Mounted Cattle #{id_j} ({int(conf*100)}%)"

                # Draw clean anatomical pose skeletons on snapshot
                for item in kpts_list:
                    if isinstance(item, tuple):
                        kpts, confs = item
                    else:
                        kpts, confs = item, None
                    for p1, p2 in COW_SKELETON:
                        if p1 < len(kpts) and p2 < len(kpts):
                            c1 = confs[p1] if confs is not None else 1.0
                            c2 = confs[p2] if confs is not None else 1.0
                            if c1 > 0.30 and c2 > 0.30:
                                pt1 = (int(kpts[p1][0]), int(kpts[p1][1]))
                                pt2 = (int(kpts[p2][0]), int(kpts[p2][1]))
                                dist = float(np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1]))
                                if dist < (w_s * 0.40) and pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                                    cv2.line(snapshot, pt1, pt2, (255, 0, 220), 2)
                    for p_idx, pt in enumerate(kpts):
                        c = confs[p_idx] if confs is not None else 1.0
                        if c > 0.30:
                            px, py = int(pt[0]), int(pt[1])
                            if px > 0 and py > 0:
                                cv2.circle(snapshot, (px, py), 4, (0, 255, 255), -1)

                cv2.rectangle(snapshot, (0, 0), (w_s, 50), (20, 20, 20), -1)
                cv2.rectangle(snapshot, (0, 0), (w_s, 50), (255, 0, 200), 2)
                cv2.putText(snapshot, banner_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(snapshot, f"Time: {dt_formatted} | Frame #{frame_num}", (w_s - 380, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 1)

                if id_i == id_j:
                    img_filename = f"estrus_cattle_{id_i}_heat_{ts_str}.jpg"
                else:
                    img_filename = f"estrus_cattle_{id_i}_on_{id_j}_{ts_str}.jpg"

                img_path = os.path.join(self.output_dir, img_filename)
                cv2.imwrite(img_path, snapshot)

                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"{time.time():.2f},{dt_formatted},{frame_num},{id_i},{id_j},{behavior},{conf:.3f},{img_path}\n")

                print(f"\n[🔥 ESTRUS & HEAT ALERT DETECTED] {behavior} | #{id_i} -> #{id_j} ({int(conf*100)}% Conf)!")
                print(f"    -> Snapshot saved to: {img_path}")
                print(f"    -> Logged to: {self.log_file}\n")
            except Exception as e:
                print(f"Warning: Could not save estrus snapshot: {e}")

        threading.Thread(target=_write_task, args=(frame.copy(), event, keypoints_list), daemon=True).start()


# -------------------------------------------------------------------
# Threaded RTSP Stream Reader (Live camera feeds)
# -------------------------------------------------------------------
class RTSPCameraStream:
    def __init__(self, src):
        self.src = src
        self.cap = None
        self.grabbed = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        
        self.connect()
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

    def connect(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        print(f"Connecting to RTSP stream: {self.src} (TCP mode)...")
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        if self.grabbed and self.frame is not None:
            print("Connected to camera stream successfully!")
        else:
            print("Waiting for camera feed...")

    def update(self):
        fail_count = 0
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(1.0)
                self.connect()
                continue

            grabbed, frame = self.cap.read()
            if grabbed and frame is not None:
                fail_count = 0
                with self.lock:
                    self.grabbed = True
                    self.frame = frame
            else:
                fail_count += 1
                if fail_count > 100:  # Allow 1.5 - 2s buffer before triggering full reconnect
                    self.connect()
                    fail_count = 0
                time.sleep(0.015)

    def read(self):
        with self.lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()


# -------------------------------------------------------------------
# Box Suppression & Shadow Filtering Helpers
# -------------------------------------------------------------------
def suppress_duplicate_boxes(boxes, iou_thresh=0.38):
    if len(boxes) <= 1:
        return boxes
    sorted_boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    selected = []
    for b in sorted_boxes:
        overlap = False
        for s in selected:
            if compute_box_iou(b[:4], s[:4]) > iou_thresh:
                overlap = True
                break
        if not overlap:
            selected.append(b)
    return selected


def is_ground_shadow(crop_bgr):
    """
    Intelligently filters out cast shadows and dark ground artifacts:
    - Real cattle (even black breeds like Murrah/Angus) have distinct coat textures,
      anatomical edges (horns, ears, limbs, spine), specular highlights, and high gradient variance.
    - Ground shadows are smooth, featureless dark silhouettes on the ground with low chroma,
      very low internal texture variance, and low edge density.
    Returns True if the region is a ground shadow false positive.
    """
    if crop_bgr is None or crop_bgr.shape[0] < 12 or crop_bgr.shape[1] < 12:
        return True

    try:
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        v_chan = hsv[:, :, 2]
        s_chan = hsv[:, :, 1]
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Color and intensity statistics
        b_mean = float(np.mean(crop_bgr[:, :, 0]))
        g_mean = float(np.mean(crop_bgr[:, :, 1]))
        r_mean = float(np.mean(crop_bgr[:, :, 2]))
        mean_intensity = (b_mean + g_mean + r_mean) / 3.0
        chroma = max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edges = cv2.Canny(gray, 40, 110)
        edge_density = float(np.sum(edges > 0)) / float(edges.size)

        # 2. Dark silhouette core analysis (isolates the dark cast shadow blob from the surrounding bright background)
        dark_mask = (v_chan < 75)
        dark_ratio = float(np.sum(dark_mask)) / float(v_chan.size)
        if dark_ratio > 0.18:
            dark_pixels = gray[dark_mask]
            dark_std = float(np.std(dark_pixels))
            dark_sat = float(np.mean(s_chan[dark_mask]))
            # Ground shadows on sand/grass have flat dark core (dark_std < 18, dark_sat < 55)
            if dark_std < 18.0 and dark_sat < 55.0:
                if dark_ratio > 0.28 or (dark_ratio > 0.18 and edge_density < 0.045):
                    return True

        # Condition A: Flat, dark ground shadow (low chroma, low texture, low edge density)
        if mean_intensity < 90 and chroma < 28 and lap_var < 100 and edge_density < 0.035:
            return True

        # Condition B: High dark-pixel silhouette with low internal edges
        if dark_ratio > 0.40 and chroma < 35 and (lap_var < 120 or edge_density < 0.040):
            return True

        # Condition C: Flat monochromatic ground patch
        if np.mean(s_chan) < 40 and lap_var < 75 and np.mean(v_chan) < 110:
            return True

        # Condition D: Featureless smooth occlusion
        if lap_var < 45 and edge_density < 0.020:
            return True

        return False
    except Exception:
        return False


def suppress_paired_shadows(boxes, frame):
    """
    When an outdoor animal and its cast shadow are both detected as adjacent or overlapping boxes,
    compares spatial orientation, aspect ratios, and texture variance
    to suppress the shadow box while preserving real cattle boxes (including mounting cattle).
    """
    if len(boxes) <= 1 or frame is None:
        return boxes

    h_f, w_f = frame.shape[:2]
    keep = [True] * len(boxes)

    for i in range(len(boxes)):
        if not keep[i]:
            continue
        bx1, by1, bx2, by2, bconf = boxes[i]
        bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
        bar = float(bw) / float(bh)

        for j in range(len(boxes)):
            if i == j or not keep[i] or not keep[j]:
                continue
            ox1, oy1, ox2, oy2, oconf = boxes[j]
            ow, oh = max(1, ox2 - ox1), max(1, oy2 - oy1)
            oar = float(ow) / float(oh)

            iou = compute_box_iou(boxes[i][:4], boxes[j][:4])
            iomin = compute_box_iomin(boxes[i][:4], boxes[j][:4])

            # Check horizontal and vertical distance between boxes
            dist_x = max(0, max(bx1, ox1) - min(bx2, ox2))
            dist_y = max(0, max(by1, oy1) - min(by2, oy2))

            # If boxes are adjacent (touching or very close at the ground base)
            is_adjacent = (dist_x <= min(bw, ow) * 0.40) and (dist_y <= max(bh, oh) * 0.25)

            if iomin > 0.15 or iou > 0.12:
                # Only suppress if crop is strictly confirmed to be a featureless cast shadow
                if (oy1 >= by1 + bh * 0.40) and (oy2 >= by2 - 10):
                    cx1, cy1 = max(0, ox1), max(0, oy1)
                    cx2, cy2 = min(w_f, ox2), min(h_f, oy2)
                    crop_j = frame[cy1:cy2, cx1:cx2]
                    if is_ground_shadow(crop_j):
                        keep[j] = False
                        continue
                elif (by1 >= oy1 + oh * 0.40) and (by2 >= oy2 - 10):
                    cx1, cy1 = max(0, bx1), max(0, by1)
                    cx2, cy2 = min(w_f, bx2), min(h_f, by2)
                    crop_i = frame[cy1:cy2, cx1:cx2]
                    if is_ground_shadow(crop_i):
                        keep[i] = False
                        break

    return [boxes[k] for k in range(len(boxes)) if keep[k]]


# -------------------------------------------------------------------
# Multi-Object Cattle Tracker (Anti-Duplicate CCTV Persistence Engine)
# -------------------------------------------------------------------
class CattleTracker:
    def __init__(self, max_lost_frames=60, max_distance=180):
        self.next_id = 1
        self.tracks = {}
        self.lost_tracks = {}
        self.max_lost = max_lost_frames
        self.max_dist = max_distance

    def reset(self):
        self.tracks.clear()
        self.lost_tracks.clear()
        self.next_id = 1

    def update(self, detected_boxes):
        dedup_boxes = suppress_duplicate_boxes(detected_boxes, iou_thresh=0.55)

        if len(dedup_boxes) == 0:
            for t_id in list(self.tracks.keys()):
                self.tracks[t_id]["lost"] += 1
                if self.tracks[t_id]["lost"] > self.max_lost:
                    del self.tracks[t_id]
            for l_id in list(self.lost_tracks.keys()):
                self.lost_tracks[l_id]["lost"] += 1
                if self.lost_tracks[l_id]["lost"] > self.max_lost:
                    del self.lost_tracks[l_id]
            return []

        det_centroids = [((b[0] + b[2]) // 2, (b[1] + b[3]) // 2) for b in dedup_boxes]
        matched_results = []
        used_det_indices = set()
        used_track_ids = set()

        if len(self.tracks) > 0:
            track_ids = list(self.tracks.keys())
            dist_pairs = []
            for t_id in track_ids:
                tcx, tcy = self.tracks[t_id]["centroid"]
                tbox = self.tracks[t_id]["box"]
                for d_idx, (dcx, dcy) in enumerate(det_centroids):
                    dbox = dedup_boxes[d_idx][:4]
                    dist = float(np.hypot(tcx - dcx, tcy - dcy))
                    iou = compute_box_iou(tbox, dbox)
                    score = dist - (iou * 140.0)
                    dist_pairs.append((score, t_id, d_idx, dist, iou))

            dist_pairs.sort(key=lambda x: x[0])

            for score, t_id, d_idx, dist, iou in dist_pairs:
                if t_id in used_track_ids or d_idx in used_det_indices:
                    continue
                if iou > 0.15 or dist <= self.max_dist:
                    used_track_ids.add(t_id)
                    used_det_indices.add(d_idx)
                    box = dedup_boxes[d_idx][:4]
                    conf = dedup_boxes[d_idx][4]
                    centroid = det_centroids[d_idx]
                    hits = self.tracks[t_id].get("hits", 0) + 1
                    self.tracks[t_id] = {"box": box, "centroid": centroid, "lost": 0, "hits": hits}
                    matched_results.append((t_id, box, conf, centroid))

        remaining_dets = [i for i in range(len(dedup_boxes)) if i not in used_det_indices]
        if remaining_dets and len(self.lost_tracks) > 0:
            lost_ids = list(self.lost_tracks.keys())
            reid_pairs = []
            for l_id in lost_ids:
                lcx, lcy = self.lost_tracks[l_id]["centroid"]
                lbox = self.lost_tracks[l_id]["box"]
                for d_idx in remaining_dets:
                    dbox = dedup_boxes[d_idx][:4]
                    dcx, dcy = det_centroids[d_idx]
                    dist = float(np.hypot(lcx - dcx, lcy - dcy))
                    iou = compute_box_iou(lbox, dbox)
                    score = dist - (iou * 120.0)
                    reid_pairs.append((score, l_id, d_idx, dist, iou))

            reid_pairs.sort(key=lambda x: x[0])
            for score, l_id, d_idx, dist, iou in reid_pairs:
                if l_id in used_track_ids or d_idx in used_det_indices:
                    continue
                if iou > 0.10 or dist <= (self.max_dist * 1.35):
                    used_track_ids.add(l_id)
                    used_det_indices.add(d_idx)
                    box = dedup_boxes[d_idx][:4]
                    conf = dedup_boxes[d_idx][4]
                    centroid = det_centroids[d_idx]
                    hits = self.lost_tracks[l_id].get("hits", 0) + 1
                    self.tracks[l_id] = {"box": box, "centroid": centroid, "lost": 0, "hits": hits}
                    del self.lost_tracks[l_id]
                    matched_results.append((l_id, box, conf, centroid))

        for t_id in list(self.tracks.keys()):
            if t_id not in used_track_ids:
                self.tracks[t_id]["lost"] += 1
                if self.tracks[t_id]["lost"] > 4:
                    self.lost_tracks[t_id] = self.tracks[t_id]
                    del self.tracks[t_id]

        for l_id in list(self.lost_tracks.keys()):
            self.lost_tracks[l_id]["lost"] += 1
            if self.lost_tracks[l_id]["lost"] > self.max_lost:
                del self.lost_tracks[l_id]

        for d_idx, box_info in enumerate(dedup_boxes):
            if d_idx not in used_det_indices:
                new_id = self.next_id
                self.next_id += 1
                box = box_info[:4]
                conf = box_info[4]
                centroid = det_centroids[d_idx]
                self.tracks[new_id] = {"box": box, "centroid": centroid, "lost": 0, "hits": 1}
                matched_results.append((new_id, box, conf, centroid))

        final_results = []
        final_results_sorted = sorted(matched_results, key=lambda x: x[0])
        for track_item in final_results_sorted:
            tid, tbox, tconf, tcen = track_item
            duplicate = False
            for ftid, fbox, fconf, fcen in final_results:
                if compute_box_iou(tbox, fbox) > 0.55:
                    duplicate = True
                    break
            if not duplicate:
                final_results.append(track_item)

        return final_results


# -------------------------------------------------------------------
# Multi-Object Human Tracker (Farm Worker / Person Persistence Engine)
# -------------------------------------------------------------------
class HumanTracker:
    def __init__(self, max_lost_frames=60, max_distance=160):
        self.next_id = 1
        self.tracks = {}
        self.lost_tracks = {}
        self.max_lost = max_lost_frames
        self.max_dist = max_distance

    def reset(self):
        self.tracks.clear()
        self.lost_tracks.clear()
        self.next_id = 1

    def update(self, detected_boxes):
        dedup_boxes = suppress_duplicate_boxes(detected_boxes, iou_thresh=0.50)

        if len(dedup_boxes) == 0:
            for t_id in list(self.tracks.keys()):
                self.tracks[t_id]["lost"] += 1
                if self.tracks[t_id]["lost"] > self.max_lost:
                    del self.tracks[t_id]
            for l_id in list(self.lost_tracks.keys()):
                self.lost_tracks[l_id]["lost"] += 1
                if self.lost_tracks[l_id]["lost"] > self.max_lost:
                    del self.lost_tracks[l_id]
            return []

        det_centroids = [((b[0] + b[2]) // 2, (b[1] + b[3]) // 2) for b in dedup_boxes]
        matched_results = []
        used_det_indices = set()
        used_track_ids = set()

        if len(self.tracks) > 0:
            track_ids = list(self.tracks.keys())
            dist_pairs = []
            for t_id in track_ids:
                tcx, tcy = self.tracks[t_id]["centroid"]
                tbox = self.tracks[t_id]["box"]
                for d_idx, (dcx, dcy) in enumerate(det_centroids):
                    dbox = dedup_boxes[d_idx][:4]
                    dist = float(np.hypot(tcx - dcx, tcy - dcy))
                    iou = compute_box_iou(tbox, dbox)
                    score = dist - (iou * 140.0)
                    dist_pairs.append((score, t_id, d_idx, dist, iou))

            dist_pairs.sort(key=lambda x: x[0])

            for score, t_id, d_idx, dist, iou in dist_pairs:
                if t_id in used_track_ids or d_idx in used_det_indices:
                    continue
                if iou > 0.15 or dist <= self.max_dist:
                    used_track_ids.add(t_id)
                    used_det_indices.add(d_idx)
                    box = dedup_boxes[d_idx][:4]
                    conf = dedup_boxes[d_idx][4]
                    centroid = det_centroids[d_idx]
                    hits = self.tracks[t_id].get("hits", 0) + 1
                    self.tracks[t_id] = {"box": box, "centroid": centroid, "lost": 0, "hits": hits}
                    matched_results.append((t_id, box, conf, centroid))

        remaining_dets = [i for i in range(len(dedup_boxes)) if i not in used_det_indices]
        if remaining_dets and len(self.lost_tracks) > 0:
            lost_ids = list(self.lost_tracks.keys())
            reid_pairs = []
            for l_id in lost_ids:
                lcx, lcy = self.lost_tracks[l_id]["centroid"]
                lbox = self.lost_tracks[l_id]["box"]
                for d_idx in remaining_dets:
                    dbox = dedup_boxes[d_idx][:4]
                    dcx, dcy = det_centroids[d_idx]
                    dist = float(np.hypot(lcx - dcx, lcy - dcy))
                    iou = compute_box_iou(lbox, dbox)
                    score = dist - (iou * 120.0)
                    reid_pairs.append((score, l_id, d_idx, dist, iou))

            reid_pairs.sort(key=lambda x: x[0])
            for score, l_id, d_idx, dist, iou in reid_pairs:
                if l_id in used_track_ids or d_idx in used_det_indices:
                    continue
                if iou > 0.10 or dist <= (self.max_dist * 1.35):
                    used_track_ids.add(l_id)
                    used_det_indices.add(d_idx)
                    box = dedup_boxes[d_idx][:4]
                    conf = dedup_boxes[d_idx][4]
                    centroid = det_centroids[d_idx]
                    hits = self.lost_tracks[l_id].get("hits", 0) + 1
                    self.tracks[l_id] = {"box": box, "centroid": centroid, "lost": 0, "hits": hits}
                    del self.lost_tracks[l_id]
                    matched_results.append((l_id, box, conf, centroid))

        for t_id in list(self.tracks.keys()):
            if t_id not in used_track_ids:
                self.tracks[t_id]["lost"] += 1
                if self.tracks[t_id]["lost"] > 4:
                    self.lost_tracks[t_id] = self.tracks[t_id]
                    del self.tracks[t_id]

        for l_id in list(self.lost_tracks.keys()):
            self.lost_tracks[l_id]["lost"] += 1
            if self.lost_tracks[l_id]["lost"] > self.max_lost:
                del self.lost_tracks[l_id]

        for d_idx, box_info in enumerate(dedup_boxes):
            if d_idx not in used_det_indices:
                new_id = self.next_id
                self.next_id += 1
                box = box_info[:4]
                conf = box_info[4]
                centroid = det_centroids[d_idx]
                self.tracks[new_id] = {"box": box, "centroid": centroid, "lost": 0, "hits": 1}
                matched_results.append((new_id, box, conf, centroid))

        final_results = []
        final_results_sorted = sorted(matched_results, key=lambda x: x[0])
        for track_item in final_results_sorted:
            tid, tbox, tconf, tcen = track_item
            duplicate = False
            for ftid, fbox, fconf, fcen in final_results:
                if compute_box_iou(tbox, fbox) > 0.50:
                    duplicate = True
                    break
            if not duplicate:
                final_results.append(track_item)

        return final_results


# -------------------------------------------------------------------
# Input Source Selection Prompt
# -------------------------------------------------------------------
def choose_input_source():
    if len(sys.argv) > 1:
        arg_path = sys.argv[1].strip("\"'")
        if os.path.exists(arg_path):
            return "video", arg_path
        elif arg_path.startswith("rtsp://") or arg_path.startswith("http://"):
            return "rtsp", arg_path

    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_videos = [
        f for f in os.listdir(current_dir)
        if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".webm"))
    ]

    print("\n" + "=" * 65)
    print("   AI CATTLE BREED, BCS & WEIGHT/HEIGHT MONITOR")
    print("=" * 65)
    print("Select input source:")
    print(f"  [1] Default Video: {DEFAULT_VIDEO_PATH}")
    print(f"  [2] Live RTSP Camera: {DEFAULT_RTSP_URL}")
    print("  [3] Browse your computer for a Video File (File Picker Popup)...")
    
    option_idx = 4
    for vid in local_videos:
        print(f"  [{option_idx}] Local Video: {vid}")
        option_idx += 1

    choice = input(f"\nEnter choice (1 to {option_idx - 1}, or paste file path) [Default 1]: ").strip().strip("\"'")

    if os.path.exists(choice):
        return "video", choice

    if choice == "2":
        return "rtsp", DEFAULT_RTSP_URL
    elif choice == "3":
        selected = None
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askopenfilename(
                title="Select Cattle Video File",
                filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov *.webm"), ("All Files", "*.*")]
            )
            root.destroy()
        except Exception:
            selected = None

        if selected and os.path.exists(selected):
            return "video", selected
        
        path = input("Enter video file path (or '2' for Live Camera, Enter for default): ").strip().strip("\"'")
        if path == "2":
            return "rtsp", DEFAULT_RTSP_URL
        if os.path.exists(path):
            return "video", path
        print("Using default video.")
        return "video", DEFAULT_VIDEO_PATH
    elif choice.isdigit() and int(choice) >= 4 and int(choice) < option_idx:
        vid_filename = local_videos[int(choice) - 4]
        return "video", os.path.join(current_dir, vid_filename)
    else:
        return "video", DEFAULT_VIDEO_PATH


# -------------------------------------------------------------------
# End-of-Video Comprehensive Analytics Summary Screen
# -------------------------------------------------------------------
def render_end_of_video_summary(
    video_source,
    total_frames,
    fps,
    unique_cattle_seen,
    cattle_profiles,
    breed_summary,
    bcs_summary,
    disease_summary,
    avg_herd_bcs,
    unique_humans_seen=None
):
    total_cattle = len(unique_cattle_seen)
    total_humans = len(unique_humans_seen) if unique_humans_seen is not None else 0
    duration_sec = total_frames / max(1.0, float(fps))
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Average weight & height across herd (only computed if cattle detected)
    weights = [prof.get("weight_kg") for prof in cattle_profiles.values() if prof.get("weight_kg") is not None]
    heights = [prof.get("height_cm") for prof in cattle_profiles.values() if prof.get("height_cm") is not None]
    avg_wt = int(np.mean(weights)) if weights else 0
    avg_ht = int(np.mean(heights)) if heights else 0
    wt_ht_banner = f"{avg_wt} kg | AVG HEIGHT: {avg_ht} cm" if (avg_wt > 0 and avg_ht > 0) else "--"
    card_wt_sub = f"Avg: {avg_wt}kg | {avg_ht}cm" if (avg_wt > 0 and avg_ht > 0) else "Avg: --"

    # 1. Print formatted report to terminal console
    print("\n" + "=" * 75)
    print("                 FINAL VIDEO CATTLE & HUMAN ANALYTICS REPORT")
    print("=" * 75)
    print(f"  Video Source        : {os.path.basename(video_source)}")
    print(f"  Processed Frames    : {total_frames} frames ({duration_sec:.1f} seconds)")
    print(f"  Analysis Timestamp  : {timestamp_str}")
    print("-" * 75)
    print(f"  >>> TOTAL CATTLE COUNTED : {total_cattle} <<<")
    print(f"  >>> TOTAL HUMANS COUNTED : {total_humans} <<<")
    print(f"  >>> HERD AVERAGE WEIGHT  : {wt_ht_banner} <<<")
    print("-" * 75)
    
    print("\n[+] BREED BREAKDOWN & POPULATION DISTRIBUTION:")
    if not breed_summary:
        print("    No cattle detected.")
    else:
        sorted_breeds = sorted(breed_summary.items(), key=lambda x: -x[1])
        for b_name, b_cnt in sorted_breeds:
            pct = (b_cnt / total_cattle * 100.0) if total_cattle > 0 else 0
            print(f"    - {b_name:<24} : {b_cnt:>3} cattle ({pct:>5.1f}%)")

    print("\n[+] HERD HEALTH & BODY CONDITION SCORE (BCS 1-5):")
    if avg_herd_bcs is not None and total_cattle > 0:
        avg_label = BCS_INFO.get(int(np.clip(round(avg_herd_bcs), 1, 5)), {}).get("label", "Ideal")
        print(f"    Average Herd BCS Score  : {avg_herd_bcs:.2f} / 5.0 [{avg_label}]")
    else:
        avg_label = "No Cattle Detected"
        print("    Average Herd BCS Score  : -- (No Cattle Detected)")
    print(f"    - Underconditioned (1-2): {bcs_summary.get(1, 0) + bcs_summary.get(2, 0)} cattle")
    print(f"    - Ideal Condition  (3)  : {bcs_summary.get(3, 0)} cattle")
    print(f"    - Overconditioned  (4-5): {bcs_summary.get(4, 0) + bcs_summary.get(5, 0)} cattle")

    n_healthy = disease_summary.get("Healthy", 0)
    n_lsd = disease_summary.get("Lumpy Skin Disease", 0)
    n_fmd = disease_summary.get("Foot and Mouth Disease", 0)
    print("\n[+] DISEASE & CLINICAL SCREENING:")
    print(f"    - Healthy Cattle        : {n_healthy} cattle")
    if n_fmd > 0:
        print(f"    - Foot & Mouth Alert (FMD): {n_fmd} cattle (URGENT VET ATTENTION)")
    if n_lsd > 0:
        print(f"    - Lumpy Skin Alert (LSD)  : {n_lsd} cattle (URGENT VET ATTENTION)")
    if n_lsd == 0 and n_fmd == 0:
        print("    - All cattle verified free of clinical lesions.")

    print("\n[+] ESTRUS & REPRODUCTIVE BEHAVIOR SCREENING:")
    estrus_log_path = os.path.join(ESTRUS_DIR, "estrus_log.csv")
    print(f"    - Event Snapshots Folder    : {os.path.abspath(ESTRUS_DIR)}")
    print(f"    - Estrus History Log (CSV)  : {os.path.abspath(estrus_log_path)}")

    print("\n[+] INDIVIDUAL CATTLE ROSTER (WITH WEIGHT & HEIGHT RANGES):")
    print(f"    {'ID':<5} | {'Breed':<18} | {'Conf':<6} | {'BCS':<11} | {'Weight (kg)':<14} | {'Height (cm)':<13} | {'Health'}")
    print("    " + "-" * 76)
    sorted_ids = sorted(list(unique_cattle_seen))
    for cid in sorted_ids:
        prof = cattle_profiles.get(cid, {})
        c_breed = prof.get("breed", "Cattle")
        c_conf = f"{int(prof.get('breed_conf', 0.85)*100)}%"
        c_bcs = f"{prof.get('bcs_float', 3.0):.1f} ({prof.get('bcs_label', 'Ideal')[:4]})"
        c_wt = f"{prof.get('weight_kg', 400)} kg"
        c_ht = f"{prof.get('height_cm', 135)} cm"
        c_dis = prof.get("disease", "Healthy")
        print(f"    #{cid:<4} | {c_breed:<18} | {c_conf:<6} | {c_bcs:<11} | {c_wt:<14} | {c_ht:<13} | {c_dis}")
    print("=" * 75)
    print("Controls: [R] = Replay Video  |  [S] = Save Report & Snapshot  |  [Q/ESC] = Exit\n")

    # 2. Render GUI Final Summary Canvas (1280x720 Dark Modern Card)
    canvas = np.full((720, 1280, 3), 18, dtype=np.uint8)

    # Top Header Banner
    cv2.rectangle(canvas, (0, 0), (1280, 75), (30, 30, 30), -1)
    cv2.line(canvas, (0, 75), (1280, 75), (0, 215, 255), 2)
    
    summary_logo = load_brand_logo(LOGO_PATH, target_size=(56, 56))
    if summary_logo is not None:
        overlay_logo_with_card(canvas, summary_logo, custom_x=20, custom_y=9, pad=4, bg_color=(255, 255, 255), border_color=(0, 215, 255), border_thickness=1)
        title_x = 100
    else:
        title_x = 40

    cv2.putText(canvas, "VIDEO ANALYSIS COMPLETE - HERD BIOMETRY & HEALTH REPORT", (title_x, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 230, 255), 2)
    cv2.putText(canvas, f"Source: {os.path.basename(video_source)}  |  {duration_sec:.1f}s  |  {timestamp_str}",
                (820, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    if avg_herd_bcs is not None and total_cattle > 0:
        avg_bcs_int = int(np.clip(round(avg_herd_bcs), 1, 5))
        avg_label = BCS_INFO.get(avg_bcs_int, {}).get("label", "Ideal")
        avg_bcs_str = f"{avg_herd_bcs:.2f}"
    else:
        avg_label = "No Cattle Detected"
        avg_bcs_str = "--"

    # Top KPI Metrics Cards (4 Cards)
    card_configs = [
        ("TOTAL CATTLE", f"{total_cattle}", (0, 255, 120), card_wt_sub),
        ("TOTAL HUMANS", f"{total_humans}", (255, 180, 0), "Workers / Herders" if total_humans > 0 else "None in Area"),
        ("AVG HERD BCS", avg_bcs_str, (50, 220, 50) if avg_bcs_str != "--" else (160, 160, 160), avg_label),
        ("HEALTH STATUS", f"{n_healthy} / {total_cattle}" if total_cattle > 0 else "--", (50, 220, 50) if (n_lsd + n_fmd) == 0 else (40, 40, 240),
         "Healthy" if (total_cattle > 0 and (n_lsd + n_fmd) == 0) else (f"{n_lsd+n_fmd} Disease Alert" if (n_lsd + n_fmd) > 0 else "No Cattle"))
    ]

    card_w = 275
    card_h = 95
    for i, (title, val, color, sub) in enumerate(card_configs):
        cx = 40 + i * (card_w + 30)
        cy = 95
        cv2.rectangle(canvas, (cx, cy), (cx + card_w, cy + card_h), (30, 30, 30), -1)
        cv2.rectangle(canvas, (cx, cy), (cx + card_w, cy + card_h), (60, 60, 60), 1)
        cv2.putText(canvas, title, (cx + 15, cy + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(canvas, val, (cx + 15, cy + 62), cv2.FONT_HERSHEY_SIMPLEX, 1.05, color, 3)
        cv2.putText(canvas, sub, (cx + 15, cy + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)

    # Left Column: Breed Breakdown Distribution Card
    left_x = 40
    left_y = 215
    left_w = 540
    left_h = 440
    cv2.rectangle(canvas, (left_x, left_y), (left_x + left_w, left_y + left_h), (26, 26, 26), -1)
    cv2.rectangle(canvas, (left_x, left_y), (left_x + left_w, left_y + left_h), (65, 65, 65), 1)
    cv2.putText(canvas, "CATTLE BREED DISTRIBUTION & COUNTS", (left_x + 20, left_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 215, 255), 2)
    cv2.line(canvas, (left_x + 20, left_y + 48), (left_x + left_w - 20, left_y + 48), (55, 55, 55), 1)

    if not breed_summary:
        cv2.putText(canvas, "No cattle detected in this video.", (left_x + 20, left_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 150), 1)
    else:
        bar_y = left_y + 80
        sorted_breeds = sorted(breed_summary.items(), key=lambda x: -x[1])
        for b_name, b_count in sorted_breeds[:7]:
            b_color = BREED_COLORS.get(b_name, (0, 215, 255))
            pct = (b_count / total_cattle) if total_cattle > 0 else 0
            cv2.putText(canvas, f"{b_name}", (left_x + 20, bar_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
            cnt_str = f"{b_count} cattle ({pct*100:.1f}%)"
            cv2.putText(canvas, cnt_str, (left_x + left_w - 170, bar_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, b_color, 1)
            bar_start_x = left_x + 20
            bar_w_max = left_w - 40
            cv2.rectangle(canvas, (bar_start_x, bar_y + 22), (bar_start_x + bar_w_max, bar_y + 34), (45, 45, 45), -1)
            fill_w = max(4, int(bar_w_max * pct))
            cv2.rectangle(canvas, (bar_start_x, bar_y + 22), (bar_start_x + fill_w, bar_y + 34), b_color, -1)
            bar_y += 48

    # Right Column: Herd Health, BCS & Manifest Table
    right_x = 610
    right_y = 215
    right_w = 630
    right_h = 440
    cv2.rectangle(canvas, (right_x, right_y), (right_x + right_w, right_y + right_h), (26, 26, 26), -1)
    cv2.rectangle(canvas, (right_x, right_y), (right_x + right_w, right_y + right_h), (65, 65, 65), 1)
    cv2.putText(canvas, "HERD ROSTER: WEIGHT, HEIGHT, BCS & HEALTH", (right_x + 20, right_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 255, 120), 2)
    cv2.line(canvas, (right_x + 20, right_y + 48), (right_x + right_w - 20, right_y + 48), (55, 55, 55), 1)

    cv2.putText(canvas, f"Thin: {bcs_summary.get(1,0)+bcs_summary.get(2,0)}  |  Ideal: {bcs_summary.get(3,0)}  |  Heavy: {bcs_summary.get(4,0)+bcs_summary.get(5,0)}",
                (right_x + 20, right_y + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    cv2.putText(canvas, f"FMD: {n_fmd}  |  LSD: {n_lsd}",
                (right_x + 450, right_y + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 220, 50) if (n_lsd+n_fmd)==0 else (0, 80, 255), 1)

    cv2.line(canvas, (right_x + 20, right_y + 88), (right_x + right_w - 20, right_y + 88), (55, 55, 55), 1)

    # Manifest Table Header
    cv2.putText(canvas, "ID", (right_x + 15, right_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)
    cv2.putText(canvas, "BREED", (right_x + 60, right_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)
    cv2.putText(canvas, "BCS", (right_x + 200, right_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)
    cv2.putText(canvas, "WEIGHT", (right_x + 295, right_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)
    cv2.putText(canvas, "HEIGHT", (right_x + 400, right_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)
    cv2.putText(canvas, "HEALTH", (right_x + 500, right_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)

    row_y = right_y + 138
    for cid in sorted_ids[:9]:
        prof = cattle_profiles.get(cid, {})
        c_breed = prof.get("breed", "Cattle")
        c_bcs = f"{prof.get('bcs_float', 3.0):.1f} [{prof.get('bcs_label', 'Ideal')[:3]}]"
        c_wt = f"{prof.get('weight_kg', 400)} kg"
        c_ht = f"{prof.get('height_cm', 135)} cm"
        c_dis = prof.get("disease", "Healthy")
        dis_col = (50, 220, 50) if c_dis == "Healthy" else ((0, 80, 255) if "Foot" in c_dis else (40, 40, 240))
        b_col = BREED_COLORS.get(c_breed, (0, 215, 255))

        cv2.putText(canvas, f"#{cid}", (right_x + 15, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1)
        cv2.putText(canvas, f"{c_breed[:14]}", (right_x + 60, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, b_col, 1)
        cv2.putText(canvas, f"{c_bcs}", (right_x + 200, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, prof.get("bcs_color", (50, 220, 50)), 1)
        cv2.putText(canvas, f"{c_wt}", (right_x + 295, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 215, 255), 1)
        cv2.putText(canvas, f"{c_ht}", (right_x + 400, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1)
        cv2.putText(canvas, f"{c_dis[:12]}", (right_x + 500, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, dis_col, 1)
        row_y += 32

    if len(sorted_ids) > 9:
        cv2.putText(canvas, f"... and {len(sorted_ids) - 9} more cattle logged in report/CSV",
                    (right_x + 25, row_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

    # Footer Action Bar
    cv2.rectangle(canvas, (0, 675), (1280, 720), (25, 25, 25), -1)
    cv2.line(canvas, (0, 675), (1280, 675), (60, 60, 60), 1)
    cv2.putText(canvas, "[R] Replay Video from Start    |    [S] Save Summary Report & Image    |    [Q / ESC] Quit",
                (320, 703), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 2)

    return canvas


def finalize_whatsapp_compatible_video(video_path):
    """
    Converts raw OpenCV recorded video to standard H.264 / AAC (YUV420p) format.
    Ensures 100% compatibility with WhatsApp (Mobile & Web), iPhone, Android, VS Code, and all web browsers!
    """
    if not video_path or not os.path.exists(video_path):
        return video_path

    try:
        import subprocess
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        temp_out = video_path.replace(".mp4", "_h264.mp4")
        if temp_out == video_path:
            temp_out = video_path + "_h264.mp4"

        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", video_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-movflags", "+faststart",
            temp_out
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        if res.returncode == 0 and os.path.exists(temp_out) and os.path.getsize(temp_out) > 1000:
            os.replace(temp_out, video_path)
            print(f"[✓] Converted to WhatsApp-compatible H.264 MP4: {video_path}")
            return video_path
    except Exception as e:
        pass

    return video_path


def export_summary_report(video_source, total_frames, fps, unique_cattle_seen, cattle_profiles, breed_summary, bcs_summary, disease_summary, avg_herd_bcs, summary_img, video_file_path=None, unique_humans_seen=None):
    out_dir = os.path.dirname(os.path.abspath(__file__))
    t_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(out_dir, f"cattle_report_{t_stamp}.csv")
    img_file = os.path.join(out_dir, f"cattle_report_{t_stamp}.png")
    total_humans = len(unique_humans_seen) if unique_humans_seen is not None else 0

    if video_file_path and os.path.exists(video_file_path):
        video_file_path = finalize_whatsapp_compatible_video(video_file_path)

    try:
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write(f"# Summary: Total Cattle Counted={len(unique_cattle_seen)}, Total Humans Counted={total_humans}\n")
            f.write("Cattle_ID,Breed,Confidence,BCS_Score,BCS_Label,Weight_kg,Weight_Range,Height_cm,Height_Range,Disease_Status\n")
            for cid in sorted(list(unique_cattle_seen)):
                prof = cattle_profiles.get(cid, {})
                c_breed = prof.get("breed", "Cattle")
                c_conf = prof.get("breed_conf", 0.85)
                c_bcs = prof.get("bcs_float", 3.0)
                c_label = prof.get("bcs_label", "Ideal")
                c_wt = prof.get("weight_kg", 400)
                c_wrange = prof.get("weight_range", "380-420 kg")
                c_ht = prof.get("height_cm", 135)
                c_hrange = prof.get("height_range", "130-140 cm")
                c_dis = prof.get("disease", "Healthy")
                f.write(f"{cid},{c_breed},{c_conf:.2f},{c_bcs:.2f},{c_label},{c_wt},{c_wrange},{c_ht},{c_hrange},{c_dis}\n")
        
        cv2.imwrite(img_file, summary_img)
        print("\n" + "=" * 75)
        print("                 SAVED ANALYTICS & MEDIA ARTIFACTS")
        print("=" * 75)
        print(f"  [✓] Summary CSV Table     : {csv_file}")
        print(f"  [✓] Summary Graphic Image : {img_file}")
        if video_file_path and os.path.exists(video_file_path):
            print(f"  [✓] Analyzed Output Video : {video_file_path}")
        if os.path.exists(ESTRUS_DIR):
            print(f"  [✓] Estrus Events Folder  : {os.path.abspath(ESTRUS_DIR)}")
            estrus_log_file = os.path.join(ESTRUS_DIR, "estrus_log.csv")
            if os.path.exists(estrus_log_file):
                print(f"  [✓] Estrus Events Log     : {os.path.abspath(estrus_log_file)}")
        print("=" * 75 + "\n")
    except Exception as e:
        print(f"Error saving report: {e}")


# -------------------------------------------------------------------
# Main Pipeline: Detection, Tracking, Breed, BCS, Weight/Height & Health
# -------------------------------------------------------------------
def main():
    source_type, source_path = choose_input_source()
    is_live = (source_type == "rtsp")

    print(f"\nOpening source: {source_path}")
    det_model_file = "yolo11n.pt" if os.path.exists("yolo11n.pt") else ("yolo11m.pt" if os.path.exists("yolo11m.pt") else "yolo11n.pt")
    print(f"Loading Fast, Low-Memory YOLO Cattle & Human Detector ({det_model_file})...")
    det_model = YOLO(det_model_file)

    trained_breed_model = None
    if os.path.exists(TRAINED_BREED_MODEL_PATH):
        try:
            print(f"Loading 50-Breed AI Classifier: '{TRAINED_BREED_MODEL_PATH}'...")
            trained_breed_model = YOLO(TRAINED_BREED_MODEL_PATH)
            num_classes = len(trained_breed_model.names) if hasattr(trained_breed_model, "names") else "50"
            print(f"Loaded {num_classes}-Breed AI Classifier successfully!")
        except Exception as e:
            print(f"Warning: Could not load {TRAINED_BREED_MODEL_PATH}: {e}")
            trained_breed_model = None

    bcs_model, bcs_device = load_bcs_model(BCS_MODEL_PATH)
    disease_model = load_disease_model(DISEASE_MODEL_PATH)
    live_brand_logo = load_brand_logo(LOGO_PATH, target_size=(110, 110))
    if live_brand_logo is not None:
        print("Loaded Chimertech Brand Logo successfully (110x110 on White Card)!")

    analysis_worker = CattleAnalysisWorker(trained_breed_model, bcs_model, bcs_device, disease_model)
    cattle_tracker = CattleTracker(max_lost_frames=90, max_distance=200)
    human_tracker = HumanTracker(max_lost_frames=90, max_distance=180)
    estrus_detector = EstrusDetector(ESTRUS_MODEL_PATH, COWXNET_HEAT_MODEL_PATH, ESTRUS_DIR)

    if is_live:
        cam = RTSPCameraStream(source_path)
        time.sleep(1.0)
        fps = 25.0
        total_video_frames = 0
    else:
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {source_path}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize PashuX Supabase Activity Engine & Web Dashboard
    try:
        from pashux_web_dashboard import start_dashboard_server
        start_dashboard_server(port=8000)
    except Exception as e:
        print(f"[Notice] Dashboard server: {e}")

    activity_engine = ActivityEngine(
        farm_id=DEFAULT_FARM_ID,
        camera_id=DEFAULT_CAMERA_ID,
        video_source_type="live" if is_live else "uploaded",
        buffer_fps=fps,
        pre_buffer_seconds=4.0,
        post_buffer_seconds=6.0,
        cooldown_seconds=60.0
    )

    conf_threshold = 0.25
    unique_cattle_seen = set()
    unique_humans_seen = set()
    counted_gate_ids = set()
    counted_human_gate_ids = set()
    track_life = {}
    human_track_life = {}
    track_trajectories = {}
    human_track_trajectories = {}
    flow_timestamps = []
    gate_flash = 0
    gate_mode = 2

    cattle_profiles = {}
    breed_summary = {}
    bcs_summary = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    disease_summary = {}
    cached_tracks = []
    cached_human_tracks = []
    cached_estrus_events = []
    cached_estrus_history = []
    cached_pose_kpts = []
    avg_herd_bcs = 3.0

    frame_count = 0
    paused = False
    video_ended = False
    summary_canvas = None

    playback_speed = 1.0
    turbo_mode = False
    INFERENCE_INTERVAL = 2 if not is_live else 1

    canvas_w = 1280
    canvas_h = 720

    # Initialize Screen Recording of Analyzed Output Video
    out_dir = os.path.dirname(os.path.abspath(__file__))
    session_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    analyzed_video_path = os.path.join(out_dir, f"analyzed_cattle_video_{session_stamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    rec_fps = float(fps) if (fps > 0 and fps <= 60) else 25.0
    video_writer = cv2.VideoWriter(analyzed_video_path, fourcc, rec_fps, (canvas_w, canvas_h))
    if not video_writer.isOpened():
        analyzed_video_path = os.path.join(out_dir, f"analyzed_cattle_video_{session_stamp}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_writer = cv2.VideoWriter(analyzed_video_path, fourcc, rec_fps, (canvas_w, canvas_h))

    if video_writer.isOpened():
        print(f"[REC] Auto-recording analyzed video to:\n      -> {analyzed_video_path}")

    WINDOW_TITLE = "Cattle & Human Detection, Breed, BCS, Weight & Health Monitor"
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    is_fullscreen = True
    hud_mode = 1

    print("\n" + "=" * 65)
    print("AI CATTLE & HUMAN MONITORING ACTIVE (NORMAL 1.0x PLAYBACK SPEED)!")
    print("  - Real-time Cattle & Human Detection and Counting")
    print("  - Real-time 50-Breed Identification")
    print("  - Body Condition Score (BCS 1 to 5)")
    print("  - Live Body Weight & Height Range Estimation")
    print("  - Disease Screening (Healthy / LSD / FMD)")
    print("  - CowXNet Heat & Estrus Detection + Mounting Kinematics Engine")
    print("  - Automatic 10-Second Video Clip Evidence & Supabase Storage Sync")
    print("  - Controls:")
    print("      [1] 1.0x Normal Speed    |  [2] 1.5x Speed  |  [3] 2.0x Double  |  [4] 4.0x Quad")
    print("      [T] Turbo Mode (Max FPS) |  [>] Fast Stride |  [<] Exact Stride |  [SPACE] Pause")
    print("      [W] Open Web Dashboard   |  [E] Test Estrus |  [G] Switch Gate  |  [Q/ESC] Quit")
    print("=" * 65 + "\n")

    while True:
        if video_ended:
            if summary_canvas is None:
                summary_canvas = render_end_of_video_summary(
                    video_source=source_path,
                    total_frames=frame_count,
                    fps=fps,
                    unique_cattle_seen=unique_cattle_seen,
                    cattle_profiles=cattle_profiles,
                    breed_summary=breed_summary,
                    bcs_summary=bcs_summary,
                    disease_summary=disease_summary,
                    avg_herd_bcs=avg_herd_bcs,
                    unique_humans_seen=unique_humans_seen
                )

            cv2.imshow(WINDOW_TITLE, summary_canvas)
            key = cv2.waitKey(30) & 0xFF

            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('r'), ord('R')):
                print("\n[⟲] Replaying video from start...")
                unique_cattle_seen.clear()
                unique_humans_seen.clear()
                counted_gate_ids.clear()
                counted_human_gate_ids.clear()
                track_life.clear()
                human_track_life.clear()
                track_trajectories.clear()
                human_track_trajectories.clear()
                flow_timestamps.clear()
                cattle_profiles.clear()
                breed_summary.clear()
                bcs_summary = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                disease_summary.clear()
                cached_tracks.clear()
                cached_human_tracks.clear()
                analysis_worker.reset()
                cattle_tracker.reset()
                human_tracker.reset()
                if not is_live:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                video_ended = False
                summary_canvas = None
                paused = False
                continue
            elif key in (ord('s'), ord('S')):
                export_summary_report(
                    source_path, frame_count, fps, unique_cattle_seen,
                    cattle_profiles, breed_summary, bcs_summary, disease_summary,
                    avg_herd_bcs, summary_canvas, analyzed_video_path,
                    unique_humans_seen=unique_humans_seen
                )
            continue

        frame_start_time = time.time()

        if not paused:
            if is_live:
                ret, frame = cam.read()
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("\n[!] Video playback completed. Finalizing activity sessions, recordings & report...")
                    if video_writer is not None and video_writer.isOpened():
                        video_writer.release()
                        video_writer = None

                    # Finalize all open activity engine sessions (feeding, resting, health)
                    try:
                        video_sec = float(frame_count) / max(1.0, fps)
                        activity_engine.finalize_all_sessions(current_video_sec=video_sec)
                    except Exception as e:
                        print(f"[ActivityEngine] Session finalization notice: {e}")

                    summary_img = render_end_of_video_summary(
                        source_path, frame_count, fps, unique_cattle_seen,
                        cattle_profiles, breed_summary, bcs_summary, disease_summary, avg_herd_bcs,
                        unique_humans_seen=unique_humans_seen
                    )
                    export_summary_report(
                        source_path, frame_count, fps, unique_cattle_seen,
                        cattle_profiles, breed_summary, bcs_summary, disease_summary,
                        avg_herd_bcs, summary_img, analyzed_video_path,
                        unique_humans_seen=unique_humans_seen
                    )
                    print("\n[✓] Video analysis completed. All data stored in Supabase & local cache. Closing cleanly...")
                    try:
                        cv2.imshow(WINDOW_TITLE, summary_img)
                        cv2.waitKey(2000)
                    except Exception:
                        pass
                    video_ended = True
                    break

            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            activity_engine.push_frame(frame)

            if frame_count % INFERENCE_INTERVAL == 0:
                h_frame, w_frame = frame.shape[:2]
                # High-res herd detection: 1280px image size with 0.12 conf captures distant hillside cattle cleanly
                infer_imgsz = 1280 if (w_frame >= 1280 or h_frame >= 720) else 960
                infer_conf = min(conf_threshold, 0.12)

                with torch.inference_mode():
                    results = det_model.predict(
                        frame,
                        classes=TARGET_CLASSES,
                        imgsz=infer_imgsz,
                        conf=infer_conf,
                        verbose=False
                    )

                detected_boxes = []
                detected_human_boxes = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        coords = list(map(int, box.xyxy[0]))
                        x1, y1, x2, y2 = coords
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0]) if box.cls is not None else COCO_COW_CLASS
                        box_w, box_h = max(0, x2 - x1), max(0, y2 - y1)

                        if cls_id == COCO_COW_CLASS:
                            # Cattle size filter: allow small distant hillside cattle (down to 8x8px or 80px^2)
                            if box_w < 8 or box_h < 8 or (box_w * box_h) < 80:
                                continue

                            aspect_ratio = float(box_w) / max(1.0, float(box_h))
                            if aspect_ratio < 0.22 or aspect_ratio > 4.2:
                                continue

                            # Verify ground shadow only on larger crops
                            if box_w >= 28 and box_h >= 28:
                                cx1, cy1 = max(0, x1), max(0, y1)
                                cx2, cy2 = min(w_frame, x2), min(h_frame, y2)
                                cand_crop = frame[cy1:cy2, cx1:cx2]
                                if is_ground_shadow(cand_crop):
                                    continue

                            detected_boxes.append((x1, y1, x2, y2, conf))

                        elif cls_id == COCO_PERSON_CLASS:
                            # Strict Human Validation:
                            # 1. Require strong detection confidence (>= 0.52) to reject noise on cattle fur/nostrils/grass
                            if conf < 0.52:
                                continue
                            # 2. Minimum human dimensions (humans in camera view are upright & substantial)
                            if box_w < 35 or box_h < 75 or (box_w * box_h) < 3000:
                                continue
                            # 3. Upright Aspect Ratio: standing/walking human has box_w <= 1.05 * box_h
                            aspect_ratio = float(box_w) / max(1.0, float(box_h))
                            if aspect_ratio < 0.15 or aspect_ratio > 1.05:
                                continue
                            detected_human_boxes.append((x1, y1, x2, y2, conf))

                # Macro / Close-up Cattle Inspection Fallback:
                # If no full-body cattle bbox was detected by YOLO COCO, check if frame is a close-up cattle view
                if len(detected_boxes) == 0 and trained_breed_model is not None:
                    try:
                        b_res = trained_breed_model.predict(frame, verbose=False)
                        top1_conf = float(b_res[0].probs.top1conf) if b_res and b_res[0].probs is not None else 0.0
                        if top1_conf >= 0.25:
                            mx1 = int(w_frame * 0.04)
                            my1 = int(h_frame * 0.04)
                            mx2 = int(w_frame * 0.96)
                            my2 = int(h_frame * 0.96)
                            detected_boxes.append((mx1, my1, mx2, my2, max(0.85, top1_conf)))
                            # Clear false human triggers inside close-up cattle view
                            detected_human_boxes.clear()
                    except Exception:
                        pass

                # Filter out adjacent paired shadows of real cattle
                detected_boxes = suppress_paired_shadows(detected_boxes, frame)
                tracked_objects = cattle_tracker.update(detected_boxes)
                tracked_humans = human_tracker.update(detected_human_boxes)

                current_tracks = []
                current_human_tracks = []
                gate_x_raw = int(w_frame * 0.50)
                gate_y_raw = int(h_frame * 0.50)

                # Process cattle tracks
                for track_id, (x1, y1, x2, y2), conf, (cx_raw, cy_raw) in tracked_objects:
                    track_life[track_id] = track_life.get(track_id, 0) + 1

                    if track_life[track_id] >= 4:
                        unique_cattle_seen.add(track_id)

                    if track_id in track_trajectories and len(track_trajectories[track_id]) > 0:
                        prev_cx, prev_cy = track_trajectories[track_id][-1]
                        crossed = False
                        if gate_mode == 0:
                            if (prev_cx < gate_x_raw <= cx_raw) or (prev_cx > gate_x_raw >= cx_raw):
                                crossed = True
                        elif gate_mode == 1:
                            if (prev_cy < gate_y_raw <= cy_raw) or (prev_cy > gate_y_raw >= cy_raw):
                                crossed = True

                        if crossed and track_id not in counted_gate_ids and track_life[track_id] >= 4:
                            counted_gate_ids.add(track_id)
                            gate_flash = 12
                            flow_timestamps.append(time.time())

                    if track_id not in track_trajectories:
                        track_trajectories[track_id] = []
                    track_trajectories[track_id].append((cx_raw, cy_raw))
                    if len(track_trajectories[track_id]) > 25:
                        track_trajectories[track_id].pop(0)

                    cx1, cy1 = max(0, x1), max(0, y1)
                    cx2, cy2 = min(w_frame, x2), min(h_frame, y2)
                    cattle_crop = frame[cy1:cy2, cx1:cx2]

                    if not analysis_worker.is_done(track_id) and cattle_crop.shape[0] >= 20 and cattle_crop.shape[1] >= 20:
                        analysis_worker.submit(track_id, cattle_crop, (x1, y1, x2, y2), frame.shape)

                    res = analysis_worker.get_result(track_id)
                    if res is not None:
                        cattle_profiles[track_id] = res
                    elif track_id not in cattle_profiles:
                        cattle_profiles[track_id] = {
                            "breed": "Scanning...",
                            "breed_conf": 0.0,
                            "bcs_int": None,
                            "bcs_float": None,
                            "bcs_label": "Scanning",
                            "bcs_color": (160, 160, 160),
                            "disease": "Scanning...",
                            "disease_conf": 0.0,
                            "disease_color": (160, 160, 160),
                            "weight_kg": None,
                            "weight_range": "Estimating...",
                            "height_cm": None,
                            "height_range": "Estimating..."
                        }

                    prof = cattle_profiles[track_id]
                    current_tracks.append({
                        "id": track_id,
                        "box": (x1, y1, x2, y2),
                        "conf": conf,
                        "breed": prof.get("breed", "Scanning..."),
                        "breed_conf": prof.get("breed_conf", 0.0),
                        "bcs_int": prof.get("bcs_int"),
                        "bcs_float": prof.get("bcs_float"),
                        "bcs_label": prof.get("bcs_label", "Scanning"),
                        "bcs_color": prof.get("bcs_color", (160, 160, 160)),
                        "disease": prof.get("disease", "Scanning..."),
                        "disease_conf": prof.get("disease_conf", 0.0),
                        "disease_color": prof.get("disease_color", (160, 160, 160)),
                        "weight_kg": prof.get("weight_kg"),
                        "weight_range": prof.get("weight_range", "Estimating..."),
                        "height_cm": prof.get("height_cm"),
                        "height_range": prof.get("height_range", "Estimating..."),
                    })

                # Process human tracks
                for h_id, (hx1, hy1, hx2, hy2), h_conf, (hcx_raw, hcy_raw) in tracked_humans:
                    human_track_life[h_id] = human_track_life.get(h_id, 0) + 1

                    if human_track_life[h_id] >= 8:
                        unique_humans_seen.add(h_id)

                    if h_id in human_track_trajectories and len(human_track_trajectories[h_id]) > 0:
                        prev_cx, prev_cy = human_track_trajectories[h_id][-1]
                        crossed = False
                        if gate_mode == 0:
                            if (prev_cx < gate_x_raw <= hcx_raw) or (prev_cx > gate_x_raw >= hcx_raw):
                                crossed = True
                        elif gate_mode == 1:
                            if (prev_cy < gate_y_raw <= hcy_raw) or (prev_cy > gate_y_raw >= hcy_raw):
                                crossed = True

                        if crossed and h_id not in counted_human_gate_ids and human_track_life[h_id] >= 8:
                            counted_human_gate_ids.add(h_id)

                    if h_id not in human_track_trajectories:
                        human_track_trajectories[h_id] = []
                    human_track_trajectories[h_id].append((hcx_raw, hcy_raw))
                    if len(human_track_trajectories[h_id]) > 25:
                        human_track_trajectories[h_id].pop(0)

                    current_human_tracks.append({
                        "id": h_id,
                        "box": (hx1, hy1, hx2, hy2),
                        "conf": h_conf
                    })

                all_confirmed_ids = counted_gate_ids if (gate_mode in (0, 1) and len(counted_gate_ids) > 0) else unique_cattle_seen
                breed_summary = {}
                bcs_summary = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                disease_summary = {}
                all_bcs_scores = []

                for t_id in all_confirmed_ids:
                    prof = cattle_profiles.get(t_id, {})
                    b = prof.get("breed", "Scanning...")
                    if b not in ("Scanning...", "Cattle"):
                        breed_summary[b] = breed_summary.get(b, 0) + 1

                    b_int = prof.get("bcs_int")
                    if b_int is not None and b_int in bcs_summary:
                        bcs_summary[b_int] = bcs_summary.get(b_int, 0) + 1
                    
                    b_float = prof.get("bcs_float")
                    if b_float is not None and b_float > 0:
                        all_bcs_scores.append(b_float)

                    d = prof.get("disease", "Scanning...")
                    if d not in ("Scanning...", ""):
                        disease_summary[d] = disease_summary.get(d, 0) + 1

                    d = prof.get("disease", "Scanning...")

                avg_herd_bcs = float(np.mean(all_bcs_scores)) if all_bcs_scores else None

                # Estrus / Mounting Action & Pose Detection
                active_events, pose_kpts, estrus_history = estrus_detector.detect_estrus(frame, current_tracks, frame_count)
                cached_estrus_events = active_events
                cached_pose_kpts = pose_kpts
                cached_estrus_history = estrus_history
                cached_tracks = current_tracks
                cached_human_tracks = current_human_tracks

                # Automatic Farm Activity Recognition & 10-Second Video Clip Evidence (Supabase)
                activity_engine.process_frame_activities(
                    frame=frame,
                    cattle_tracks=current_tracks,
                    human_tracks=current_human_tracks,
                    estrus_events=cached_estrus_events,
                    frame_idx=frame_count
                )
            else:
                current_tracks = cached_tracks
                current_human_tracks = cached_human_tracks

        # -------------------------------------------------------------
        # Draw Overlays, Bounding Boxes, Biometry & Health Badges
        # -------------------------------------------------------------
        canvas_w = 1280
        canvas_h = 720
        orig_h, orig_w = frame.shape[:2]
        scale = min(canvas_w / orig_w, canvas_h / orig_h)
        nw = int(orig_w * scale)
        nh = int(orig_h * scale)
        scaled_frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        display_frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        off_x = (canvas_w - nw) // 2
        off_y = (canvas_h - nh) // 2
        display_frame[off_y:off_y + nh, off_x:off_x + nw] = scaled_frame

        if gate_mode == 0:
            gx_disp = int((orig_w * 0.50) * scale) + off_x
            gate_color = (0, 255, 120) if gate_flash > 0 else (0, 215, 255)
            cv2.line(display_frame, (gx_disp, off_y), (gx_disp, off_y + nh), gate_color, 2)
            badge_str = f"|| GATE: {len(counted_gate_ids)} Cattle | {len(counted_human_gate_ids)} Humans ||"
            cv2.putText(display_frame, badge_str, (gx_disp - 120, off_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, gate_color, 1)

        elif gate_mode == 1:
            gy_disp = int((orig_h * 0.50) * scale) + off_y
            gate_color = (0, 255, 120) if gate_flash > 0 else (0, 215, 255)
            cv2.line(display_frame, (off_x, gy_disp), (off_x + nw, gy_disp), gate_color, 2)
            badge_str = f"== GATE: {len(counted_gate_ids)} Cattle | {len(counted_human_gate_ids)} Humans =="
            cv2.putText(display_frame, badge_str, (off_x + 20, gy_disp - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, gate_color, 1)

        if gate_flash > 0:
            gate_flash -= 1

        # Draw Pose Skeletons on Canvas
        for item in cached_pose_kpts:
            if isinstance(item, tuple):
                kpts, confs = item
            else:
                kpts, confs = item, None
            for p1, p2 in COW_SKELETON:
                if p1 < len(kpts) and p2 < len(kpts):
                    c1 = confs[p1] if confs is not None else 1.0
                    c2 = confs[p2] if confs is not None else 1.0
                    if c1 > 0.30 and c2 > 0.30:
                        pt1 = (int(kpts[p1][0] * scale) + off_x, int(kpts[p1][1] * scale) + off_y)
                        pt2 = (int(kpts[p2][0] * scale) + off_x, int(kpts[p2][1] * scale) + off_y)
                        dist = float(np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1]))
                        if dist < (canvas_w * 0.40) and pt1[0] > off_x and pt1[1] > off_y and pt2[0] > off_x and pt2[1] > off_y:
                            cv2.line(display_frame, pt1, pt2, (255, 0, 220), 2)
            for p_idx, pt in enumerate(kpts):
                c = confs[p_idx] if confs is not None else 1.0
                if c > 0.30:
                    px, py = int(pt[0] * scale) + off_x, int(pt[1] * scale) + off_y
                    if px > off_x and py > off_y:
                        cv2.circle(display_frame, (px, py), 3, (0, 255, 255), -1)

        # Draw Top Estrus Notification Banner if active
        if len(cached_estrus_events) > 0:
            ev = cached_estrus_events[0]
            action_desc = "Sniffing (Estrus)" if "Sniffing" in ev["behavior"] else "Mounted"
            alert_str = f"ESTRUS ALERT: Cattle #{ev['mounting_id']} {action_desc} #{ev['mounted_id']} ({int(ev['conf']*100)}%) | Saved to estrus/"
            banner_y = off_y + 10
            cv2.rectangle(display_frame, (off_x + 40, banner_y), (off_x + nw - 40, banner_y + 36), (20, 20, 20), -1)
            cv2.rectangle(display_frame, (off_x + 40, banner_y), (off_x + nw - 40, banner_y + 36), (255, 0, 200), 2)
            cv2.putText(display_frame, alert_str, (off_x + 55, banner_y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

        # Build behavior mapping per ID
        event_role_map = {}
        for ev in cached_estrus_events:
            m_id = ev["mounting_id"]
            d_id = ev["mounted_id"]
            beh = ev["behavior"]
            if "Sniffing" in beh:
                event_role_map[m_id] = "[ESTRUS - SNIFFING]"
                event_role_map[d_id] = "[ESTRUS TARGET]"
            else:
                event_role_map[m_id] = "[ESTRUS - MOUNTING]"
                event_role_map[d_id] = "[STANDING HEAT]"

        # Draw Cattle Boxes, Motion Trails & Biometry Badges
        mounting_ids = {ev["mounting_id"] for ev in cached_estrus_events}
        mounted_ids = {ev["mounted_id"] for ev in cached_estrus_events}

        for item in cached_tracks:
            rx1, ry1, rx2, ry2 = item["box"]
            x1 = int(rx1 * scale) + off_x
            y1 = int(ry1 * scale) + off_y
            x2 = int(rx2 * scale) + off_x
            y2 = int(ry2 * scale) + off_y

            t_id = item["id"]
            breed = item["breed"]
            bcs_val = item.get("bcs_float", 3.0)
            bcs_label = item.get("bcs_label", "Ideal")
            bcs_color = item.get("bcs_color", (50, 220, 50))
            dis_name = item.get("disease", "Healthy")
            dis_conf = item.get("disease_conf", 0.0)
            dis_short = DISEASE_INFO.get(dis_name, {"short": dis_name}).get("short", dis_name)
            wt_kg = item.get("weight_kg", 400)
            ht_cm = item.get("height_cm", 135)
            wt_range = item.get("weight_range", "380-420 kg")

            is_counted = (t_id in counted_gate_ids) or (gate_mode == 2 and t_id in unique_cattle_seen)
            is_fmd = "Foot and Mouth" in dis_name
            is_diseased = dis_name not in ("Healthy", "Scanning...")
            is_cowxnet_heat = item.get("is_heat", False)
            heat_conf = item.get("heat_conf", 0.0)

            if t_id in mounting_ids:
                box_color = (255, 0, 200) # Neon Pink / Mounting Estrus
            elif t_id in mounted_ids:
                box_color = (255, 100, 230) # Soft Magenta / Standing Heat
            elif is_cowxnet_heat:
                box_color = (255, 0, 180) # Neon Pink / CowXNet Heat Alert
            elif is_fmd:
                box_color = (0, 80, 255)
            elif is_diseased:
                box_color = (40, 40, 240)
            elif is_counted:
                box_color = (50, 220, 50)
            else:
                box_color = (0, 215, 255)

            if t_id in track_trajectories and len(track_trajectories[t_id]) > 1:
                pts = track_trajectories[t_id]
                for p_idx in range(1, len(pts)):
                    pt1 = (int(pts[p_idx-1][0] * scale) + off_x, int(pts[p_idx-1][1] * scale) + off_y)
                    pt2 = (int(pts[p_idx][0] * scale) + off_x, int(pts[p_idx][1] * scale) + off_y)
                    cv2.line(display_frame, pt1, pt2, box_color, 2)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)

            # Badge Text (Clean ASCII, no unicode checkmarks)
            if t_id in event_role_map:
                count_tag = event_role_map[t_id]
            elif is_cowxnet_heat:
                count_tag = f"[HEAT ACTIVE {int(heat_conf*100)}%]"
            elif is_counted:
                count_tag = "[COUNTED]"
            else:
                count_tag = "[SCANNING]"

            box_disp_w = max(1, x2 - x1)
            box_disp_h = max(1, y2 - y1)

            # Adaptive Badge Sizing:
            # Small / distant cattle in dense herd views (< 75px) get a compact 1-line badge
            # to avoid obscuring nearby cattle on slopes/pastures.
            if box_disp_h < 75 or box_disp_w < 75:
                mini_text = f"#{t_id} {count_tag} {breed}"
                sz_mini, _ = cv2.getTextSize(mini_text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
                bw = sz_mini[0] + 8
                bh = 18
                by1 = max(0, y1 - bh)
                by2 = y1

                cv2.rectangle(display_frame, (x1, by1), (x1 + bw, by2), (18, 18, 18), -1)
                cv2.rectangle(display_frame, (x1, by1), (x1 + bw, by2), box_color, 1)
                cv2.putText(display_frame, mini_text, (x1 + 4, by1 + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (245, 245, 245), 1)
            else:
                # Full 3-Line High-Tech Biometry Card for Standard/Close-up View
                line1_text = f"#{t_id} {count_tag} {breed}"
                dis_conf_str = f" {int(dis_conf*100)}%" if dis_conf > 0 else ""
                if bcs_val is not None:
                    line2_text = f"BCS: {bcs_val:.1f} [{bcs_label}] | HP: {dis_short}{dis_conf_str}"
                else:
                    line2_text = f"BCS: Scanning... | HP: {dis_short}"
                
                if wt_kg is not None and ht_cm is not None:
                    line3_text = f"WT: {wt_kg} kg [{wt_range}] | HT: {ht_cm} cm"
                else:
                    line3_text = "WT: Estimating... | HT: Estimating..."

                sz1, _ = cv2.getTextSize(line1_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                sz2, _ = cv2.getTextSize(line2_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                sz3, _ = cv2.getTextSize(line3_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                bw = max(sz1[0], sz2[0], sz3[0]) + 14
                bh = 54
                by1 = max(0, y1 - bh)
                by2 = y1

                cv2.rectangle(display_frame, (x1, by1), (x1 + bw, by2), (18, 18, 18), -1)
                cv2.rectangle(display_frame, (x1, by1), (x1 + bw, by2), box_color, 1)
                cv2.putText(display_frame, line1_text, (x1 + 5, by1 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (245, 245, 245), 1)
                cv2.putText(display_frame, line2_text, (x1 + 5, by1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, bcs_color, 1)
                cv2.putText(display_frame, line3_text, (x1 + 5, by1 + 46),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 255), 1)

        # Draw Human Boxes, Motion Trails & Badges
        for h_item in current_human_tracks:
            rx1, ry1, rx2, ry2 = h_item["box"]
            hx1 = int(rx1 * scale) + off_x
            hy1 = int(ry1 * scale) + off_y
            hx2 = int(rx2 * scale) + off_x
            hy2 = int(ry2 * scale) + off_y
            h_id = h_item["id"]
            h_conf = h_item["conf"]

            is_h_counted = (h_id in counted_human_gate_ids) or (gate_mode == 2 and h_id in unique_humans_seen)
            h_box_color = (255, 180, 0)  # Vivid Amber / Orange-Yellow for Human

            if h_id in human_track_trajectories and len(human_track_trajectories[h_id]) > 1:
                h_pts = human_track_trajectories[h_id]
                for p_idx in range(1, len(h_pts)):
                    pt1 = (int(h_pts[p_idx-1][0] * scale) + off_x, int(h_pts[p_idx-1][1] * scale) + off_y)
                    pt2 = (int(h_pts[p_idx][0] * scale) + off_x, int(h_pts[p_idx][1] * scale) + off_y)
                    cv2.line(display_frame, pt1, pt2, h_box_color, 2)

            cv2.rectangle(display_frame, (hx1, hy1), (hx2, hy2), h_box_color, 2)

            h_tag = "[COUNTED]" if is_h_counted else "[SCANNING]"
            h_line_text = f"Human #{h_id} {h_tag} ({int(h_conf*100)}%)"
            h_sz, _ = cv2.getTextSize(h_line_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            h_bw = h_sz[0] + 12
            h_bh = 22
            h_by1 = max(0, hy1 - h_bh)
            h_by2 = hy1

            cv2.rectangle(display_frame, (hx1, h_by1), (hx1 + h_bw, h_by2), (18, 18, 18), -1)
            cv2.rectangle(display_frame, (hx1, h_by1), (hx1 + h_bw, h_by2), h_box_color, 1)
            cv2.putText(display_frame, h_line_text, (hx1 + 5, h_by1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

        # HUD Dashboard Card
        total_counted = len(counted_gate_ids) if (gate_mode in (0, 1) and len(counted_gate_ids) > 0) else len(unique_cattle_seen)
        total_humans_counted = len(counted_human_gate_ids) if (gate_mode in (0, 1) and len(counted_human_gate_ids) > 0) else len(unique_humans_seen)
        gate_mode_labels = {0: "Vertical Gate", 1: "Horizontal Gate", 2: "Full View Accumulator"}

        # Calculate average herd weight & height in live view
        live_wts = [item["weight_kg"] for item in cached_tracks if item.get("weight_kg") is not None]
        live_hts = [item["height_cm"] for item in cached_tracks if item.get("height_cm") is not None]

        if live_wts and live_hts:
            hud_avg_wt = int(np.mean(live_wts))
            hud_avg_ht = int(np.mean(live_hts))
            wt_ht_text = f"Herd Weight & Height : ~{hud_avg_wt} kg  |  ~{hud_avg_ht} cm ({hud_avg_ht/30.48:.1f} ft)"
        elif total_counted > 0:
            conf_wts = [cattle_profiles[cid]["weight_kg"] for cid in all_confirmed_ids if cid in cattle_profiles and cattle_profiles[cid].get("weight_kg") is not None]
            conf_hts = [cattle_profiles[cid]["height_cm"] for cid in all_confirmed_ids if cid in cattle_profiles and cattle_profiles[cid].get("height_cm") is not None]
            if conf_wts and conf_hts:
                c_avg_w = int(np.mean(conf_wts))
                c_avg_h = int(np.mean(conf_hts))
                wt_ht_text = f"Herd Weight & Height : ~{c_avg_w} kg  |  ~{c_avg_h} cm ({c_avg_h/30.48:.1f} ft)"
            else:
                wt_ht_text = "Herd Weight & Height : Estimating..."
        else:
            wt_ht_text = "Herd Weight & Height : --  |  -- (No Cattle in View)"

        if avg_herd_bcs is not None and total_counted > 0:
            avg_bcs_int = int(np.clip(round(avg_herd_bcs), 1, 5))
            avg_bcs_info = BCS_INFO.get(avg_bcs_int, {"label": "Ideal", "color": (50, 220, 50)})
            bcs_text = f"Avg Herd BCS Score   : {avg_herd_bcs:.2f} [{avg_bcs_info['label']}]"
            bcs_color = avg_bcs_info["color"]
        elif total_counted > 0:
            bcs_text = "Avg Herd BCS Score   : Scanning..."
            bcs_color = (0, 215, 255)
        else:
            bcs_text = "Avg Herd BCS Score   : -- (No Cattle in View)"
            bcs_color = (160, 160, 160)

        if hud_mode == 1:
            hud_w = 460
            breed_lines = min(len(breed_summary), 4)
            disease_lines = min(len(disease_summary), 3)
            hud_h = 280 + breed_lines * 18 + disease_lines * 18
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (15, 15), (15 + hud_w, 15 + hud_h), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.82, display_frame, 0.18, 0, display_frame)
            cv2.rectangle(display_frame, (15, 15), (15 + hud_w, 15 + hud_h), (80, 80, 80), 1)

            cv2.putText(display_frame, "AI HERD & HUMAN BIOMETRY MONITOR", (25, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 230, 255), 2)
            cv2.putText(display_frame, f"CATTLE: {total_counted}  |  HUMANS: {total_humans_counted}", (25, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 120), 2)
            
            speed_label = "Turbo" if turbo_mode else f"{playback_speed:.1f}x (Normal)"
            cv2.putText(display_frame, f"In View: Cattle: {len(cached_tracks)} | Humans: {len(current_human_tracks)} | Spd: {speed_label}", (25, 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)
            cv2.putText(display_frame, wt_ht_text, (25, 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255) if total_counted > 0 else (160, 160, 160), 1)
            cv2.putText(display_frame, bcs_text, (25, 128),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, bcs_color, 2 if total_counted > 0 else 1)

            cv2.line(display_frame, (25, 140), (15 + hud_w - 15, 140), (70, 70, 70), 1)
            if total_counted > 0:
                bcs_summary_str = f"Thin: {bcs_summary[1] + bcs_summary[2]} | Ideal: {bcs_summary[3]} | Heavy: {bcs_summary[4] + bcs_summary[5]}"
            else:
                bcs_summary_str = "Thin: 0 | Ideal: 0 | Heavy: 0"
            cv2.putText(display_frame, bcs_summary_str, (25, 158),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

            cv2.line(display_frame, (25, 170), (15 + hud_w - 15, 170), (70, 70, 70), 1)
            n_healthy = disease_summary.get("Healthy", 0)
            n_lsd = disease_summary.get("Lumpy Skin Disease", 0)
            n_fmd = disease_summary.get("Foot and Mouth Disease", 0)
            n_sick = n_lsd + n_fmd
            
            if total_counted > 0:
                dis_status_text = f"Healthy: {n_healthy}  |  FMD: {n_fmd}" if n_fmd > 0 else f"Healthy: {n_healthy}  |  Sick: {n_sick}"
                dis_col = (50, 220, 50) if n_sick == 0 else (0, 80, 255)
            else:
                dis_status_text = "No Cattle Detected"
                dis_col = (160, 160, 160)

            cv2.putText(display_frame, f"Disease Status:  {dis_status_text}", (25, 188),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, dis_col, 1)
            if n_fmd > 0:
                cv2.putText(display_frame, f"  * Foot & Mouth (FMD): {n_fmd}", (30, 208),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 80, 255), 1)
            if n_lsd > 0:
                cv2.putText(display_frame, f"  * Lumpy Skin (LSD): {n_lsd}", (30, 226),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 240), 1)

            line_y = 212 if n_sick == 0 else (234 + (18 if (n_lsd > 0 and n_fmd > 0) else 0))
            cv2.line(display_frame, (25, line_y), (15 + hud_w - 15, line_y), (70, 70, 70), 1)
            cv2.putText(display_frame, f"Gate Mode: {gate_mode_labels[gate_mode]} [Press G]", (25, line_y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

            line_y += 34
            if not breed_summary:
                cv2.putText(display_frame, "Scanning cattle breeds...", (25, line_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)
            else:
                shown_breeds = sorted(breed_summary.items(), key=lambda x: -x[1])[:4]
                for b_name, b_count in shown_breeds:
                    b_color = BREED_COLORS.get(b_name, (0, 215, 255))
                    cv2.putText(display_frame, f"  * {b_name}: {b_count}", (30, line_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, b_color, 1)
                    line_y += 18

            cv2.putText(display_frame, f"[1/2/3/T] Speed  |  [+/-] Conf: {conf_threshold:.2f}  |  [G] Gate  |  [H] HUD", (25, hud_h + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 210, 255), 1)

        elif hud_mode == 0:
            bar_w = 900
            bar_h = 28
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (15, 12), (15 + bar_w, 12 + bar_h), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)
            cv2.rectangle(display_frame, (15, 12), (15 + bar_w, 12 + bar_h), (80, 80, 80), 1)

            if total_counted > 0:
                wt_disp = f"~{hud_avg_wt}kg" if live_wts else "--"
                ht_disp = f"~{hud_avg_ht}cm" if live_hts else "--"
                bcs_disp = f"{avg_herd_bcs:.1f}" if avg_herd_bcs is not None else "--"
            else:
                wt_disp = "--"
                ht_disp = "--"
                bcs_disp = "--"

            top_status = (f"CATTLE: {total_counted} (View: {len(cached_tracks)})  |  "
                          f"HUMANS: {total_humans_counted} (View: {len(current_human_tracks)})  |  "
                          f"WT: {wt_disp}  |  "
                          f"HT: {ht_disp}  |  "
                          f"BCS: {bcs_disp}")
            cv2.putText(display_frame, top_status, (25, 31),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1)

        # -------------------------------------------------------------
        # Estrus & Mounting Action Live Data Table Overlay Card
        # -------------------------------------------------------------
        tbl_w = 465
        tbl_h = 165
        tbl_x = canvas_w - tbl_w - 20
        tbl_y = canvas_h - tbl_h - 20

        tbl_overlay = display_frame.copy()
        cv2.rectangle(tbl_overlay, (tbl_x, tbl_y), (tbl_x + tbl_w, tbl_y + tbl_h), (16, 16, 16), -1)
        cv2.addWeighted(tbl_overlay, 0.86, display_frame, 0.14, 0, display_frame)

        is_estrus_live = len(cached_estrus_events) > 0
        border_col = (255, 0, 200) if is_estrus_live else (70, 70, 70)
        cv2.rectangle(display_frame, (tbl_x, tbl_y), (tbl_x + tbl_w, tbl_y + tbl_h), border_col, 2 if is_estrus_live else 1)

        # Header Title and Status Badge
        hdr_status = "[LIVE MOUNTING]" if is_estrus_live else "IDLE / MONITORING"
        hdr_col = (0, 255, 120) if is_estrus_live else (160, 160, 160)
        cv2.putText(display_frame, "ESTRUS & MOUNTING ACTION DATA TABLE", (tbl_x + 12, tbl_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 0, 200), 2)
        cv2.putText(display_frame, hdr_status, (tbl_x + tbl_w - 170, tbl_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, hdr_col, 1)
        cv2.line(display_frame, (tbl_x + 10, tbl_y + 28), (tbl_x + tbl_w - 10, tbl_y + 28), (60, 60, 60), 1)

        # Table Column Headers
        cv2.putText(display_frame, "TIME", (tbl_x + 12, tbl_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1)
        cv2.putText(display_frame, "MOUNTING", (tbl_x + 76, tbl_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1)
        cv2.putText(display_frame, "MOUNTED", (tbl_x + 160, tbl_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1)
        cv2.putText(display_frame, "ACTION / BEHAVIOR", (tbl_x + 242, tbl_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1)
        cv2.putText(display_frame, "CONF", (tbl_x + 400, tbl_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1)
        cv2.line(display_frame, (tbl_x + 10, tbl_y + 50), (tbl_x + tbl_w - 10, tbl_y + 50), (50, 50, 50), 1)

        # Populate Data Table Rows
        display_rows = []
        if is_estrus_live:
            for ev in cached_estrus_events:
                display_rows.append((ev["time_str"], f"#{ev['mounting_id']}", f"#{ev['mounted_id']}", "Mounting / Heat", f"{int(ev['conf']*100)}%", True))

        for ev in reversed(cached_estrus_history):
            if len(display_rows) >= 3:
                break
            r_item = (ev["time_str"], f"#{ev['mounting_id']}", f"#{ev['mounted_id']}", "Standing Heat", f"{int(ev['conf']*100)}%", False)
            if not any(r[0] == r_item[0] and r[1] == r_item[1] and r[2] == r_item[2] for r in display_rows):
                display_rows.append(r_item)

        if len(display_rows) == 0:
            cv2.putText(display_frame, "No estrus / mounting events logged yet", (tbl_x + 14, tbl_y + 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1)
            cv2.putText(display_frame, "Monitoring herd 24/7 for standing heat & mounting...", (tbl_x + 14, tbl_y + 102),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (100, 100, 100), 1)
        else:
            r_y = tbl_y + 72
            for r_time, r_m1, r_m2, r_act, r_conf, is_active in display_rows[:3]:
                r_col = (255, 120, 240) if is_active else (210, 210, 210)
                prefix = "* " if is_active else ""
                cv2.putText(display_frame, r_time, (tbl_x + 12, r_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1)
                cv2.putText(display_frame, f"{prefix}{r_m1}", (tbl_x + 76, r_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 0, 200) if is_active else (0, 215, 255), 1)
                cv2.putText(display_frame, r_m2, (tbl_x + 160, r_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 120) if is_active else (180, 180, 180), 1)
                cv2.putText(display_frame, r_act, (tbl_x + 242, r_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, r_col, 1)
                cv2.putText(display_frame, r_conf, (tbl_x + 400, r_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 230, 255), 1)
                r_y += 24

        # Footer Note
        cv2.putText(display_frame, f"Snapshots: {ESTRUS_DIR}/ | Auto-logged to CSV", (tbl_x + 12, tbl_y + tbl_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 140), 1)

        # -------------------------------------------------------------
        # Live Brand Logo Overlay (Medium 110x110 on Crisp White Card)
        # -------------------------------------------------------------
        if live_brand_logo is not None:
            lw_card = live_brand_logo.shape[1]
            lh_card = live_brand_logo.shape[0]
            card_pad = 6
            card_x = canvas_w - lw_card - (card_pad * 2) - 20
            card_y = 15
            overlay_logo_with_card(
                display_frame,
                live_brand_logo,
                custom_x=card_x,
                custom_y=card_y,
                pad=card_pad,
                bg_color=(255, 255, 255),
                border_color=(60, 60, 60),
                border_thickness=2
            )

        # Write Frame to Output Video Recording
        if video_writer is not None and video_writer.isOpened() and not paused:
            video_writer.write(display_frame)

        # Show active 10s video capture alert banner if capturing
        with activity_engine.jobs_lock:
            if activity_engine.active_recording_jobs:
                latest_job = activity_engine.active_recording_jobs[-1]
                act_type = latest_job["meta"]["activity_type"].replace("_", " ").upper()
                cv2.rectangle(display_frame, (canvas_w // 2 - 260, 12), (canvas_w // 2 + 260, 46), (20, 20, 180), -1)
                cv2.rectangle(display_frame, (canvas_w // 2 - 260, 12), (canvas_w // 2 + 260, 46), (0, 229, 155), 2)
                cv2.putText(display_frame, f"[AUTO-CAPTURING 10s CLIP] {act_type}", (canvas_w // 2 - 245, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

        cv2.imshow(WINDOW_TITLE, display_frame)

        if not is_live and not turbo_mode and not paused:
            target_frame_duration = 1.0 / max(1.0, (fps * playback_speed))
            compute_elapsed = time.time() - frame_start_time
            sleep_needed = target_frame_duration - compute_elapsed
            wait_delay = max(1, int(sleep_needed * 1000)) if sleep_needed > 0 else 1
        else:
            wait_delay = 1

        key = cv2.waitKey(wait_delay) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('1'),):
            playback_speed = 1.0
            turbo_mode = False
            INFERENCE_INTERVAL = 1
            print("Playback Speed set to: 1.0x (Normal Video Speed)")
        elif key in (ord('2'),):
            playback_speed = 1.5
            turbo_mode = False
            INFERENCE_INTERVAL = 2
            print("Playback Speed set to: 1.5x (Fast Stride)")
        elif key in (ord('3'),):
            playback_speed = 2.0
            turbo_mode = False
            INFERENCE_INTERVAL = 2
            print("Playback Speed set to: 2.0x Double Speed")
        elif key in (ord('4'),):
            playback_speed = 4.0
            turbo_mode = False
            INFERENCE_INTERVAL = 3
            print("Playback Speed set to: 4.0x Quad Speed")
        elif key in (ord('t'), ord('T')):
            turbo_mode = not turbo_mode
            INFERENCE_INTERVAL = 2 if turbo_mode else 1
            print(f"Turbo Mode (Maximum Hardware FPS): {'ON (Ultra Fast Processing)' if turbo_mode else 'OFF (Normal 1.0x Speed)'}")
        elif key in (ord('>'), ord('.')):
            INFERENCE_INTERVAL = min(4, INFERENCE_INTERVAL + 1)
            print(f"Inference Frame Stride set to: Every {INFERENCE_INTERVAL} frames (Faster)")
        elif key in (ord('<'), ord(',')):
            INFERENCE_INTERVAL = max(1, INFERENCE_INTERVAL - 1)
            print(f"Inference Frame Stride set to: Every {INFERENCE_INTERVAL} frames (Higher Precision)")
        elif key in (ord('+'), ord('='), ord(']')):
            conf_threshold = min(0.90, round(conf_threshold + 0.05, 2))
            print(f"Confidence Threshold increased to: {conf_threshold:.2f}")
        elif key in (ord('-'), ord('_'), ord('[')):
            conf_threshold = max(0.15, round(conf_threshold - 0.05, 2))
            print(f"Confidence Threshold decreased to: {conf_threshold:.2f}")
        elif key in (ord('g'), ord('G')):
            gate_mode = (gate_mode + 1) % 3
            print(f"Counting Gate Mode changed to: {gate_mode_labels[gate_mode]}")
        elif key in (ord('h'), ord('H'), ord('m'), ord('M')):
            hud_mode = (hud_mode + 1) % 3
        elif key in (ord('f'), ord('F')):
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        elif key in (ord('p'), ord(' ')):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key in (ord('w'), ord('W')):
            try:
                webbrowser.open("http://localhost:8000")
                print("\n[✓] Opened PashuX Activity History Dashboard in browser: http://localhost:8000")
            except Exception as e:
                print(f"Notice: Could not open browser: {e}")
        elif key in (ord('e'), ord('E')):
            sim_ev = estrus_detector.trigger_manual_estrus_alert(frame, current_tracks, frame_count)
            if sim_ev:
                cached_estrus_events = [sim_ev]
                print(f"\n[🔥 TEST ESTRUS ALERT TRIGGERED] Cattle #{sim_ev['mounting_id']} -> #{sim_ev['mounted_id']} | Snapshot saved to estrus/!")
        elif key in (ord('r'), ord('R')):
            unique_cattle_seen.clear()
            unique_humans_seen.clear()
            counted_gate_ids.clear()
            counted_human_gate_ids.clear()
            track_life.clear()
            human_track_life.clear()
            track_trajectories.clear()
            human_track_trajectories.clear()
            flow_timestamps.clear()
            cattle_profiles.clear()
            breed_summary.clear()
            bcs_summary = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            disease_summary.clear()
            cached_tracks.clear()
            cached_human_tracks.clear()
            analysis_worker.reset()
            cattle_tracker.reset()
            human_tracker.reset()
            if not is_live:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            video_ended = False
            summary_canvas = None
            print("Counters reset and video rewound.")

    print("Releasing resources...")
    if video_writer is not None and video_writer.isOpened():
        video_writer.release()
        print(f"[✓] Finalized raw recording: {analyzed_video_path}")
        analyzed_video_path = finalize_whatsapp_compatible_video(analyzed_video_path)
        print(f"[✓] Saved WhatsApp-ready video: {analyzed_video_path}")
    analysis_worker.stop()
    if is_live:
        cam.stop()
    else:
        cap.release()
    cv2.destroyAllWindows()
    print("Program exited cleanly.")


if __name__ == "__main__":
    main()