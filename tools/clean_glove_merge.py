from pathlib import Path
import hashlib
import shutil
import random

DATASET = Path("dataset")

TRAIN_IMAGES = DATASET / "images" / "train"
TRAIN_LABELS = DATASET / "labels" / "train"

VAL_IMAGES = DATASET / "images" / "val"
VAL_LABELS = DATASET / "labels" / "val"

SOURCE_IMAGES = Path("glove_converted/images")
SOURCE_LABELS = Path("glove_converted/labels")


def file_hash(path):
    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


print("Removing previous newglove files...")

removed = 0

for image_folder, label_folder in [
    (TRAIN_IMAGES, TRAIN_LABELS),
    (VAL_IMAGES, VAL_LABELS)
]:
    for image in image_folder.glob("newglove_*"):
        image.unlink()
        removed += 1

    for label in label_folder.glob("newglove_*.txt"):
        label.unlink()

print(f"Removed: {removed} old newglove images")


print()
print("Building hash list of original dataset...")

existing_hashes = set()

for folder in [TRAIN_IMAGES, VAL_IMAGES]:
    for image in folder.iterdir():
        if image.is_file():
            existing_hashes.add(file_hash(image))

print(f"Original unique images: {len(existing_hashes)}")


print()
print("Finding unique Glove images...")

pairs = []
source_hashes = set()

for image in SOURCE_IMAGES.iterdir():

    if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

    label = SOURCE_LABELS / f"{image.stem}.txt"

    if not label.exists():
        continue

    image_hash = file_hash(image)

    # Skip images already present in original dataset
    if image_hash in existing_hashes:
        continue

    # Skip duplicates inside new Glove dataset
    if image_hash in source_hashes:
        continue

    source_hashes.add(image_hash)
    pairs.append((image, label))


print(f"Unique new Glove images: {len(pairs)}")


random.seed(42)
random.shuffle(pairs)

split = int(len(pairs) * 0.80)

train_pairs = pairs[:split]
val_pairs = pairs[split:]

print(f"New Glove training images: {len(train_pairs)}")
print(f"New Glove validation images: {len(val_pairs)}")


def copy_pair(image, label, image_folder, label_folder, number):

    name = f"newglove_{number:05d}"

    shutil.copy2(
        image,
        image_folder / f"{name}{image.suffix.lower()}"
    )

    shutil.copy2(
        label,
        label_folder / f"{name}.txt"
    )


for i, (image, label) in enumerate(train_pairs, 1):
    copy_pair(
        image,
        label,
        TRAIN_IMAGES,
        TRAIN_LABELS,
        i
    )


for i, (image, label) in enumerate(val_pairs, 1):
    copy_pair(
        image,
        label,
        VAL_IMAGES,
        VAL_LABELS,
        len(train_pairs) + i
    )


print()
print("=" * 50)
print("CLEAN GLOVE MERGE COMPLETE")
print("=" * 50)
print(f"Training added   : {len(train_pairs)}")
print(f"Validation added : {len(val_pairs)}")
print("Original dataset was preserved.")