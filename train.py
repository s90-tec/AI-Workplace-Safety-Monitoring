from ultralytics import YOLO

print("Loading YOLO model...")

model = YOLO("yolo11n.pt")

print("Starting V3 training...")

results = model.train(
    data="dataset/data.yaml",
    epochs=80,
    imgsz=640,
    batch=8,
    name="ppe_safety_model_v3",
    project="runs"
)

print()
print("================================")
print("V3 TRAINING COMPLETED")
print("================================")