from pathlib import Path
import hashlib
import math

DATASET = Path("dataset")

splits = ["train", "val", "test"]
class_names = [
	"Boots",
	"Ear-protection",
	"Glass",
	"Glove",
	"Helmet",
	"Mask",
	"Person",
	"Vest",
]
box_boundary_tolerance = 1e-6

print("\n===== DATASET VALIDATION =====\n")

all_valid = True
image_hashes = {}

for split in splits:
	class_counts = [0] * len(class_names)
	image_dir = DATASET / "images" / split
	label_dir = DATASET / "labels" / split

	images = [
		path for path in image_dir.iterdir()
		if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
	]

	labels = list(label_dir.glob("*.txt"))

	for image in images:
		digest = hashlib.sha256(image.read_bytes()).hexdigest()
		previous_image = image_hashes.get(digest)
		if previous_image is not None:
			print(
				f"Duplicate image content: {previous_image} and "
				f"{image_dir / image.name}"
			)
			all_valid = False
		else:
			image_hashes[digest] = image_dir / image.name

	image_names = {x.stem for x in images}
	label_names = {x.stem for x in labels}

	missing_labels = image_names - label_names
	missing_images = label_names - image_names

	print(f"{split.upper()}")
	print(f"Images : {len(images)}")
	print(f"Labels : {len(labels)}")

	if missing_labels:
		print("Missing labels:")
		for name in sorted(missing_labels):
			print("  ", name)
		all_valid = False

	if missing_images:
		print("Missing images:")
		for name in sorted(missing_images):
			print("  ", name)
		all_valid = False

	# Check class IDs
	for label_file in labels:
		with open(label_file, "r", encoding="utf-8") as f:
			seen_annotations = set()
			for line_number, line in enumerate(f, start=1):
				line = line.strip()

				if not line:
					continue

				if line in seen_annotations:
					print(
						f"Duplicate annotation in "
						f"{label_file}:{line_number}"
					)
					all_valid = False
				seen_annotations.add(line)

				parts = line.split()

				try:
					if len(parts) != 5:
						raise ValueError("expected five YOLO label values")

					class_id = int(parts[0])
					coordinates = [float(value) for value in parts[1:]]
				except ValueError:
					print(
						f"Invalid annotation in {label_file}:"
						f"{line_number}: {line}"
					)
					all_valid = False
					continue

				if class_id < 0 or class_id >= len(class_names):
					print(
						f"Invalid class ID {class_id} "
						f"in {label_file}:{line_number}"
					)
					all_valid = False
				else:
					class_counts[class_id] += 1

				if not all(
					math.isfinite(value) and 0 <= value <= 1
					for value in coordinates
				):
					print(
						f"Invalid normalized coordinates in "
						f"{label_file}:{line_number}"
					)
					all_valid = False
				elif coordinates[2] <= 0 or coordinates[3] <= 0:
					print(
						f"Non-positive box size in "
						f"{label_file}:{line_number}"
					)
					all_valid = False
				else:
					x_center, y_center, width, height = coordinates
					if (
						x_center - width / 2 < -box_boundary_tolerance
						or y_center - height / 2 < -box_boundary_tolerance
						or x_center + width / 2 > 1 + box_boundary_tolerance
						or y_center + height / 2 > 1 + box_boundary_tolerance
					):
						print(
							f"Bounding box extends outside image in "
							f"{label_file}:{line_number}"
						)
						all_valid = False

	print()
	print(
		"Class boxes: "
		+ ", ".join(
			f"{name}={count}"
			for name, count in zip(class_names, class_counts)
		)
	)
	absent_classes = [
		name for name, count in zip(class_names, class_counts)
		if count == 0
	]
	if absent_classes:
		print(
			"Coverage warning (not a malformed-label error): "
			"No boxes for "
			+ ", ".join(absent_classes)
		)
	print()

if all_valid:
	print("PASS: DATASET VALIDATION")
	print("All class IDs and normalized bounding boxes are valid.")
	print("Image/label filenames match.")
	print("\nYou can proceed to training.")
else:
	print("FAIL: DATASET VALIDATION")
	print("Fix the problems above before training.")
