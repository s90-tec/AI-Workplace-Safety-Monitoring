from pathlib import Path
import hashlib

TRAIN_IMAGES = Path("dataset/images/train")
TRAIN_LABELS = Path("dataset/labels/train")

VAL_IMAGES = Path("dataset/images/val")
VAL_LABELS = Path("dataset/labels/val")


def image_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# Hash all ORIGINAL images first
original_hashes = {}

for folder in [TRAIN_IMAGES, VAL_IMAGES]:
    for image in folder.iterdir():
        if image.is_file() and not image.stem.startswith("newglove_"):
            original_hashes[image_hash(image)] = image


removed = 0

# Remove duplicate newglove images and their labels
for folder, label_folder in [
    (TRAIN_IMAGES, TRAIN_LABELS),
    (VAL_IMAGES, VAL_LABELS)
]:
    for image in list(folder.glob("newglove_*")):

        if not image.is_file():
            continue

        if image_hash(image) in original_hashes:

            label = label_folder / f"{image.stem}.txt"

            image.unlink()

            if label.exists():
                label.unlink()

            removed += 1

print()
print("=" * 50)
print("DUPLICATE GLOVE CLEANUP COMPLETE")
print("=" * 50)
print(f"Duplicate image/label pairs removed: {removed}")