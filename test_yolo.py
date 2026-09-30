from ultralytics import YOLO


model = YOLO("yolo11n.pt")

results = model("test.jpg")

for result in results:
    result.show()

print("YOLO test completed successfully.")