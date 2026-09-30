from pathlib import Path

val_labels = Path("dataset/labels/val")

count = 0

for label_file in val_labels.glob("*.txt"):

    for line in label_file.read_text().splitlines():

        if not line.strip():
            continue

        class_id = line.split()[0]

        if class_id == "3":
            count += 1

print("Validation Glove annotations:", count)