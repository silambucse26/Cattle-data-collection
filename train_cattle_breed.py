"""
Cattle Breed Classifier Trainer
Trains a lightweight, high-performance YOLOv11 Classification model
using the Kaggle Cattle Breed dataset.

Can be run locally or pasted into Google Colab / Kaggle Notebook for free GPU training.
"""

import os
import shutil
import kagglehub
from ultralytics import YOLO

DATASET_NAME = "atharvadarpude/indian-cattle-image-dataset"
OUTPUT_MODEL_NAME = "cattle_breed_model.pt"

def download_and_inspect_dataset():
    print("=" * 65)
    print(f"1. Downloading dataset '{DATASET_NAME}' via kagglehub...")
    print("=" * 65)
    dataset_path = kagglehub.dataset_download(DATASET_NAME)
    print(f"Dataset downloaded to: {dataset_path}")

    # Inspect directory contents
    contents = os.listdir(dataset_path)
    print(f"Top-level contents: {contents}")

    return dataset_path


def train_breed_model(dataset_path, epochs=15, imgsz=224):
    print("\n" + "=" * 65)
    print("2. Starting YOLOv11 Classification Training...")
    print(f"   Target epochs: {epochs}, Image size: {imgsz}")
    print("=" * 65)

    # Use pretrained YOLOv11 nano classification backbone
    model = YOLO("yolo11n-cls.pt")

    # Train classifier
    results = model.train(
        data=dataset_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        workers=2,
        project="cattle_training",
        name="breed_run",
        exist_ok=True
    )

    best_weights = os.path.join("cattle_training", "breed_run", "weights", "best.pt")
    if os.path.exists(best_weights):
        shutil.copy(best_weights, OUTPUT_MODEL_NAME)
        print(f"\nTraining Complete! Best model saved to: {OUTPUT_MODEL_NAME}")
        print("Your camera/video script will now automatically use this trained AI model!")
    else:
        print("\nTraining completed. Check cattle_training/breed_run for weights.")


if __name__ == "__main__":
    try:
        path = download_and_inspect_dataset()
        train_breed_model(path, epochs=15)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTIP: Since this Kaggle dataset is 5.86 GB, it is highly recommended to")
        print("run this training script on Google Colab or a Kaggle Notebook with a FREE T4 GPU.")
        print("Once trained, simply download 'best.pt' and place it here as 'cattle_breed_model.pt'.")
