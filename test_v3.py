from ultralytics import YOLO

MODEL_PATH = "runs/detect/runs/ppe_safety_model_v3/weights/best.pt"

model = YOLO(MODEL_PATH)

results = model("new_test.jpg", conf=0.10)

for result in results:
    result.save(filename="v3_test.jpg")

    print()
    print("Detected objects:")
    print("------------------")

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        print(f"{class_name}: {confidence:.2f}")

print()
print("Saved: v3_test.jpg")