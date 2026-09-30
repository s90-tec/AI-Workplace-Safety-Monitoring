from pathlib import Path
import shutil

SOURCE = Path("glove_new_images")
OUTPUT = Path("glove_converted")

# Construction-PPE class
SOURCE_GLOVE_ID = "1"

# Your project class
TARGET_GLOVE_ID = "3"

OUTPUT_IMAGES = OUTPUT / "images"
OUTPUT_LABELS = OUTPUT / "labels"

OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

copied = 0

for label_file in SOURCE.rglob("*.txt"):

    lines = label_file.read_text(encoding="utf-8").splitlines()

    glove_lines = []

    for line in lines:
        parts = line.strip().split()

        if not parts:
            continue

        # Keep only Gloves
        if parts[0] == SOURCE_GLOVE_ID:
            parts[0] = TARGET_GLOVE_ID
            glove_lines.append(" ".join(parts))

    # Skip images that don't contain gloves
    if not glove_lines:
        continue

    # Find corresponding image
    image_found = None

    for extension in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        possible_image = label_file.parent.parent / "images" / label_file.name.replace(
            label_file.suffix, extension
        )

        if possible_image.exists():
            image_found = possible_image
            break

    # If folder structure is different, search by filename
    if image_found is None:
        for image in SOURCE.rglob(label_file.stem + ".*"):
            if image.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                image_found = image
                break

    if image_found is None:
        print(f"Image not found for: {label_file.name}")
        continue

    # Copy image
    output_image = OUTPUT_IMAGES / image_found.name
    shutil.copy2(image_found, output_image)

    # Save converted label
    output_label = OUTPUT_LABELS / f"{label_file.stem}.txt"
    output_label.write_text(
        "\n".join(glove_lines) + "\n",
        encoding="utf-8"
    )

    copied += 1
    print(f"[{copied}] {image_found.name}")

print()
print("=" * 50)
print("GLOVE EXTRACTION COMPLETE")
print("=" * 50)
print(f"Images extracted : {copied}")
print(f"Output folder    : {OUTPUT.resolve()}")
print()
print("Construction-PPE class 1 → Your project class 3")