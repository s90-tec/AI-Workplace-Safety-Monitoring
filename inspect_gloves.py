from pathlib import Path
from PIL import Image, ImageDraw
import sys

split = sys.argv[1] if len(sys.argv) > 1 else "train"

if split not in {"train", "val"}:
    raise SystemExit("Choose a split: train or val")

labels_folder = Path("dataset/labels") / split
images_folder = Path("dataset/images") / split
glove_class_id = 3
items = []

for label_file in sorted(labels_folder.glob("*.txt")):
    glove_boxes = []

    for line in label_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        parts = line.split()
        class_id = int(parts[0])

        if class_id == glove_class_id:
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            glove_boxes.append((x_center, y_center, width, height))

    if not glove_boxes:
        continue

    image_file = next(
        (
            candidate
            for candidate in images_folder.iterdir()
            if candidate.is_file()
            and candidate.stem == label_file.stem
            and candidate.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ),
        None,
    )
    if image_file is None:
        raise FileNotFoundError(f"No image found for Glove label: {label_file}")
    items.append((image_file, glove_boxes))


print(f"Glove images found in {split}:", len(items))

thumb_width = 320
thumb_height = 320
label_height = 28
columns = 4
page_size = 40
rows_per_page = page_size // columns
base_name = (
    "glove_dataset_review"
    if split == "train"
    else "glove_validation_review"
)
page_count = max(1, (len(items) + page_size - 1) // page_size)

for page_index in range(page_count):
    page_items = items[
        page_index * page_size:(page_index + 1) * page_size
    ]
    sheet = Image.new(
        "RGB",
        (
            columns * thumb_width,
            32 + rows_per_page * (thumb_height + label_height),
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    start_number = page_index * page_size + 1
    end_number = page_index * page_size + len(page_items)
    draw.text(
        (8, 8),
        f"{split.upper()} Glove labels {start_number}-{end_number} of {len(items)}",
        fill="black",
    )

    for page_item_index, (image_file, glove_boxes) in enumerate(page_items):
        image = Image.open(image_file).convert("RGB")
        original_width, original_height = image.size
        image.thumbnail((thumb_width - 10, thumb_height - 10))

        column = page_item_index % columns
        row = page_item_index // columns
        x_offset = column * thumb_width + (thumb_width - image.width) // 2
        y_offset = 32 + row * (thumb_height + label_height) + 5
        sheet.paste(image, (x_offset, y_offset))
        scale_x = image.width / original_width
        scale_y = image.height / original_height

        for x_center, y_center, width, height in glove_boxes:
            x1 = int((x_center - width / 2) * original_width * scale_x)
            y1 = int((y_center - height / 2) * original_height * scale_y)
            x2 = int((x_center + width / 2) * original_width * scale_x)
            y2 = int((y_center + height / 2) * original_height * scale_y)
            x1 += x_offset
            x2 += x_offset
            y1 += y_offset
            y2 += y_offset
            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
            draw.text((x1, max(y_offset, y1 - 18)), "GLOVE", fill="red")

        filename = image_file.name
        if len(filename) > 34:
            filename = filename[:31] + "..."
        number = page_index * page_size + page_item_index + 1
        text_y = 32 + row * (thumb_height + label_height) + thumb_height + 5
        draw.text(
            (column * thumb_width + 5, text_y),
            f"{number}. {filename}",
            fill="black",
        )

    suffix = "" if page_index == 0 else f"_{page_index + 1:03d}"
    output = Path(f"{base_name}{suffix}.jpg")
    sheet.save(output, quality=95)
    print(f"Review image created: {output}")