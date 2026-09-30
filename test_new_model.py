from ultralytics import YOLO


MODEL_PATH = "runs/detect/runs/ppe_safety_model_v2/weights/best.pt"
IMAGE_PATH = "test.jpg"

print("Loading new PPE model...")

model = YOLO(MODEL_PATH)

print("Running detection...")

results = model(IMAGE_PATH, conf=0.10)

for result in results:
    result.save(filename="new_model_test.jpg")

    print("\nDetected objects:")

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        print(f"{class_name}: {confidence:.2f}")

print("\nTest completed.")
print("Result saved as: new_model_test.jpg")
