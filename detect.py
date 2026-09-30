from ultralytics import YOLO


model = YOLO(
    "runs/detect/runs/ppe_safety_model_v3/weights/best.pt"
)

image_path = "new_test.jpg"

results = model(image_path, conf=0.10)

# Display results
for result in results:
	result.show()

print("PPE detection completed successfully.")
