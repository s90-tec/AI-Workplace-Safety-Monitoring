import glob
import os
import torch
from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_WEIGHTS_PATTERN = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "runs",
    "**",
    "ppe_safety_model_v*",
    "weights",
    "best.pt",
)


def main():
    print("================================")
    print("AI WORKPLACE SAFETY MODEL TRAINING")
    print("================================")

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    batch_size = 16 if torch.cuda.is_available() else 8
    worker_count = 4 if torch.cuda.is_available() else 2
    print(f"Hardware Acceleration: {device_name} (device={device})")
    print(f"Training batch size: {batch_size}; data workers: {worker_count}")

    checkpoints = sorted(
        glob.glob(RUN_WEIGHTS_PATTERN, recursive=True),
        key=os.path.getmtime,
        reverse=True,
    )
    starting_weights = checkpoints[0] if checkpoints else "yolo11n.pt"
    print(f"Loading starting weights: {starting_weights}...")
    model = YOLO(starting_weights)

    print("Starting model training on PPE safety dataset...")
    results = model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=batch_size,
        device=device,
        workers=worker_count,
        name="ppe_safety_model_v4",
        project="runs/detect/runs"
    )

    print()
    print("================================")
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("================================")

if __name__ == "__main__":
    main()