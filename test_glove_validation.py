from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = "runs/detect/runs/ppe_safety_model_v2/weights/best.pt"

model = YOLO(MODEL_PATH)

val_images = []

for label_file in Path("dataset/labels/val").glob("*.txt"):
    lines = label_file.read_text().splitlines()

    # Class 3 = Glove
    if any(line.strip().startswith("3 ") for line in lines):
        image_name = label_file.stem

        for extension in [".jpg", ".jpeg", ".png"]:
            image_path = Path("dataset/images/val") / (image_name + extension)

            if image_path.exists():
                val_images.append(image_path)
                break

print(f"Glove validation images found: {len(val_images)}")
print()

for image_path in val_images:
    print("=" * 50)
    print(f"Testing: {image_path.name}")

    results = model(str(image_path), conf=0.01)

    detected_glove = False

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        print(f"{class_name}: {confidence:.3f}")

        if class_name == "Glove":
            detected_glove = True

    if detected_glove:
        print(">>> Glove detected")
    else:
        print(">>> Glove NOT detected")

    output_name = f"glove_check_{image_path.stem}.jpg"
    results[0].save(filename=output_name)

    print(f"Saved: {output_name}")

print()
print("Glove validation testing completed.")