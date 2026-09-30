def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_area(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    return width * height


def overlap_ratio(ppe_box, person_box):
    """
    Calculates how much of the PPE bounding box
    overlaps with the person's bounding box.
    """

    ppe_area = box_area(ppe_box)

    if ppe_area == 0:
        return 0

    intersection = intersection_area(
        ppe_box,
        person_box
    )

    return intersection / ppe_area


def check_worker_ppe(person_box, ppe_detections):

    detected_ppe = set()

    for item in ppe_detections:

        class_name = item["class_name"]
        ppe_box = item["box"]

        ratio = overlap_ratio(
            ppe_box,
            person_box
        )

        # PPE must overlap sufficiently with the worker
        if ratio >= 0.20:
            detected_ppe.add(class_name)

    required_ppe = {
        "Helmet",
        "Vest",
        "Glove"
    }

    missing = sorted(
        required_ppe - detected_ppe
    )

    if missing:
        status = "VIOLATION"
    else:
        status = "SAFE"

    return status, missing