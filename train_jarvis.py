"""
Train 50-Breed Indian Cattle Classifier on JarvisLabs.ai (or any cloud GPU)
Using YOLOv11 Classification (Ultralytics)

Instructions on JarvisLabs:
1. Open Terminal in JupyterLab.
2. Run: pip install ultralytics kaggle kagglehub
3. Run: python train_jarvis.py
"""

import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

# -------------------------------------------------------------
# 1. Dataset Configuration
# -------------------------------------------------------------
# Atharva Darpude's Indian Cattle Dataset slug on Kaggle
DATASET_SLUG = "atharvadarpude/indian-cattle-image-dataset"
LOCAL_EXTRACT_DIR = Path("./cattle_dataset_raw")
SPLIT_DATASET_DIR = Path("./cattle_dataset_split")
FINAL_MODEL_NAME = "cattle_breed_model.pt"

# Training parameters
MODEL_BACKBONE = "yolo11s-cls.pt"  # yolo11n-cls (fastest) or yolo11s-cls (higher accuracy for 50 classes)
EPOCHS = 25
IMG_SIZE = 224
BATCH_SIZE = 32  # 32 works great on GPUs like RTX 4090 / A5000 / RTX 5000


def download_dataset():
    """Download and extract dataset via KaggleHub or Kaggle API."""
    print("=" * 70)
    print("STEP 1: Downloading Indian Cattle Image Dataset from Kaggle...")
    print("=" * 70)
    
    # Try using kagglehub first
    try:
        import kagglehub
        path = kagglehub.dataset_download(DATASET_SLUG)
        print(f"Downloaded via kagglehub to: {path}")
        return Path(path)
    except Exception as e:
        print(f"Kagglehub download notice: {e}")
        print("Checking if dataset was downloaded manually or via Kaggle CLI...")
        
        # Check current working directory for unzipped folders
        possible_paths = [
            Path("cattle"),
            Path("Indian_Cattle_Dataset"),
            Path("images"),
            Path(".")
        ]
        for p in possible_paths:
            if p.exists():
                subdirs = [d for d in p.iterdir() if d.is_dir() and d.name not in ["cattle_dataset_split", "__pycache__", "runs", ".ipynb_checkpoints"]]
                if len(subdirs) >= 20: # Found breed directories
                    print(f"Found breed folders in: {p.resolve()}")
                    return p.resolve()
        
        raise RuntimeError(
            "Could not automatically download dataset.\n"
            "Please ensure Kaggle API is configured, or download manually using:\n"
            f"kaggle datasets download -d {DATASET_SLUG} --unzip"
        )


def find_breed_root(raw_path: Path):
    """Find the directory that contains the 50 breed subfolders."""
    # Check if raw_path itself directly contains breed folders
    subdirs = [d for d in raw_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    # If there's an 'images' or 'cattle' folder inside, check that
    for candidate in ["images", "cattle", "Indian_Cattle_Dataset"]:
        cand_path = raw_path / candidate
        if cand_path.exists() and cand_path.is_dir():
            nested_subdirs = [d for d in cand_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if len(nested_subdirs) > len(subdirs):
                return cand_path

    return raw_path


def prepare_train_val_split(raw_dir: Path, split_dir: Path, train_ratio: float = 0.8):
    """
    Split the dataset into standard YOLO classification structure:
      split_dir/
        train/
          Gir/
          Sahiwal/
          ...
        val/
          Gir/
          Sahiwal/
          ...
    """
    print("\n" + "=" * 70)
    print("STEP 2: Preparing 80/20 Train/Validation Split for 50 Breeds...")
    print("=" * 70)

    breed_root = find_breed_root(raw_dir)
    breed_folders = [d for d in breed_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    print(f"Detected {len(breed_folders)} breed classes in '{breed_root.name}'.")
    
    train_dir = split_dir / "train"
    val_dir = split_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    total_train = 0
    total_val = 0

    for breed_folder in breed_folders:
        breed_name = breed_folder.name
        
        # Get all images
        images = [f for f in breed_folder.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
        if not images:
            continue

        random.seed(42)
        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # Destination folders
        target_train_breed = train_dir / breed_name
        target_val_breed = val_dir / breed_name
        target_train_breed.mkdir(exist_ok=True)
        target_val_breed.mkdir(exist_ok=True)

        for img in train_imgs:
            dest = target_train_breed / img.name
            if not dest.exists():
                try:
                    os.link(img, dest)
                except Exception:
                    shutil.copy2(img, dest)
            total_train += 1

        for img in val_imgs:
            dest = target_val_breed / img.name
            if not dest.exists():
                try:
                    os.link(img, dest)
                except Exception:
                    shutil.copy2(img, dest)
            total_val += 1

    print(f"Data preparation complete:")
    print(f" - Training images:   {total_train}")
    print(f" - Validation images: {total_val}")
    print(f" - Ready in folder:   {split_dir.resolve()}")
    return split_dir


def train_model(data_dir: Path):
    """Train YOLOv11 classification model."""
    print("\n" + "=" * 70)
    print(f"STEP 3: Training YOLOv11 Classification ({MODEL_BACKBONE}) on GPU...")
    print(f"Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | Image Size: {IMG_SIZE}")
    print("=" * 70)

    model = YOLO(MODEL_BACKBONE)

    results = model.train(
        data=str(data_dir.resolve()),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=0,          # GPU 0 on JarvisLabs
        workers=4,
        project="cattle_project",
        name="run_50_breeds",
        exist_ok=True
    )

    best_weights = Path("cattle_project") / "run_50_breeds" / "weights" / "best.pt"
    if best_weights.exists():
        shutil.copy(best_weights, FINAL_MODEL_NAME)
        print("\n" + "=" * 70)
        print(f"SUCCESS! Best trained model saved to: {Path(FINAL_MODEL_NAME).resolve()}")
        print(f"Right-click '{FINAL_MODEL_NAME}' in JarvisLabs file explorer and click DOWNLOAD.")
        print("=" * 70)
    else:
        print(f"Training finished. Check {best_weights} for model weights.")


if __name__ == "__main__":
    # 1. Download
    raw_data_path = download_dataset()
    
    # 2. Split into train/val
    prepared_data_path = prepare_train_val_split(raw_data_path, SPLIT_DATASET_DIR)
    
    # 3. Train
    train_model(prepared_data_path)
