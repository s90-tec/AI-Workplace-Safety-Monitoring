from pathlib import Path
import shutil

train_labels = Path("dataset/labels/train")
train_images = Path("dataset/images/train")

val_labels = Path("dataset/labels/val")
val_images = Path("dataset/images/val")

val_labels.mkdir(parents=True, exist_ok=True)
val_images.mkdir(parents=True, exist_ok=True)

glove_files = []

# Find images containing Glove annotations
for label_file in sorted(train_labels.glob("*.txt")):

    lines = label_file.read_text().splitlines()

    if any(
        line.strip() and line.split()[0] == "3"
        for line in lines
    ):
        glove_files.append(label_file)

# Select 4 images
selected = glove_files[:4]

print("Selected Glove validation images:")
print()

for label_file in selected:

    print(label_file.name)

    # Move label
    shutil.move(
        str(label_file),
        str(val_labels / label_file.name)
    )

    # Find matching image
    image_file = None

    for extension in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:

        candidate = train_images / (label_file.stem + extension)

        if candidate.exists():
            image_file = candidate
            break

    if image_file is None:
        print("❌ Image not found:", label_file.stem)
        continue

    # Move image
    shutil.move(
        str(image_file),
        str(val_images / image_file.name)
    )

    print("  Image moved:", image_file.name)

print()
print("✅ Glove validation split completed.")
print("Glove images moved to validation:", len(selected))