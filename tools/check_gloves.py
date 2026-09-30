from pathlib import Path

labels_folder = Path("dataset/labels/train")
images_folder = Path("dataset/images/train")

glove_files = []

for label_file in labels_folder.glob("*.txt"):

    for line in label_file.read_text().splitlines():

        if not line.strip():
            continue

        class_id = int(line.split()[0])

        if class_id == 3:
            glove_files.append(label_file)
            break


print("=" * 60)
print("GLove annotation files found:", len(glove_files))
print("=" * 60)

for number, label_file in enumerate(glove_files, start=1):

    image_file = None

    for extension in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:

        candidate = images_folder / (label_file.stem + extension)

        if candidate.exists():
            image_file = candidate
            break

    print()
    print(f"{number}.")
    print("Label:", label_file)
    print("Image:", image_file)