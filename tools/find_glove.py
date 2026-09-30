from pathlib import Path

labels_folder = Path("dataset/labels/train")
images_folder = Path("dataset/images/train")

glove_class_id = 3

found_label = None

for label_file in labels_folder.glob("*.txt"):

    lines = label_file.read_text().splitlines()

    for line in lines:

        if not line.strip():
            continue

        class_id = int(line.split()[0])

        if class_id == glove_class_id:
            found_label = label_file
            break

    if found_label:
        break


if found_label is None:

    print("❌ No Glove annotation found.")

else:

    print("✅ Glove annotation found!")
    print()
    print("Label file:")
    print(found_label)

    image_file = None

    for extension in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:

        possible_image = (
            images_folder /
            (found_label.stem + extension)
        )

        if possible_image.exists():

            image_file = possible_image
            break


    print()
    print("Image file:")

    if image_file:
        print(image_file)

    else:
        print("❌ Matching image not found.")