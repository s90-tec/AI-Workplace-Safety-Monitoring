from ultralytics import YOLO


MODEL_PATH = "runs/detect/runs/ppe_safety_model-2/weights/best.pt"

model = YOLO(MODEL_PATH)

image_path = "dataset/images/test"

results = model(
	image_path,
	conf=0.10,
)

for result in results:
	result.show()

print("Test completed.")
