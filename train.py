import os
import torch
from ultralytics import YOLO

def main():
    print("================================")
    print("AI WORKPLACE SAFETY MODEL TRAINING")
    print("================================")

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Hardware Acceleration: {device_name} (device={device})")

    print("Loading starting weights: yolo11n.pt...")
    model = YOLO("yolo11n.pt")

    print("Starting model training on PPE safety dataset...")
    results = model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=device,
        workers=2,
        name="ppe_safety_model_v3",
        project="runs/detect/runs"
    )

    print()
    print("================================")
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("================================")

if __name__ == "__main__":
    main()