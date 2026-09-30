from pathlib import Path
import shutil
import random

SOURCE_IMAGES = Path("glove_converted/images")
SOURCE_LABELS = Path("glove_converted/labels")

TRAIN_IMAGES = Path("dataset/images/train")
TRAIN_LABELS = Path("dataset/labels/train")

VAL_IMAGES = Path("dataset/images/val")
VAL_LABELS = Path("dataset/labels/val")

TRAIN_IMAGES.mkdir(parents=True, exist_ok=True)
TRAIN_LABELS.mkdir(parents=True, exist_ok=True)
VAL_IMAGES.mkdir(parents=True, exist_ok=True)
VAL_LABELS.mkdir(parents=True, exist_ok=True)

random.seed(42)

image_extensions = {".jpg", ".jpeg", ".png"}

pairs = []

for image in SOURCE_IMAGES.iterdir():
    if image.suffix.lower() not in image_extensions:
        continue

    label = SOURCE_LABELS / f"{image.stem}.txt"

    if label.exists():
        pairs.append((image, label))

print(f"Valid image/label pairs found: {len(pairs)}")

random.shuffle(pairs)

split_index = int(len(pairs) * 0.80)

train_pairs = pairs[:split_index]
val_pairs = pairs[split_index:]

print(f"New Glove training images: {len(train_pairs)}")
print(f"New Glove validation images: {len(val_pairs)}")

def copy_pair(image, label, destination_images, destination_labels, index):
    new_name = f"newglove_{index:05d}{image.suffix.lower()}"

    shutil.copy2(
        image,
        destination_images / new_name
    )

    shutil.copy2(
        label,
        destination_labels / f"newglove_{index:05d}.txt"
    )

for i, (image, label) in enumerate(train_pairs, start=1):
    copy_pair(
        image,
        label,
        TRAIN_IMAGES,
        TRAIN_LABELS,
        i
    )

offset = len(train_pairs)

for i, (image, label) in enumerate(val_pairs, start=1):
    copy_pair(
        image,
        label,
        VAL_IMAGES,
        VAL_LABELS,
        offset + i
    )

print()
print("=" * 50)
print("GLOVE DATA MERGE COMPLETED")
print("=" * 50)
print(f"Added to training   : {len(train_pairs)}")
print(f"Added to validation : {len(val_pairs)}")