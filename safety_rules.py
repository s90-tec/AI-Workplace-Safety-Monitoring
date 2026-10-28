def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union (IoU)
    between two bounding boxes.

    Box format:
    [x1, y1, x2, y2]
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width *
        intersection_height
    )

    area_a = max(
        0,
        ax2 - ax1
    ) * max(
        0,
        ay2 - ay1
    )

    area_b = max(
        0,
        bx2 - bx1
    ) * max(
        0,
        by2 - by1
    )

    union_area = (
        area_a +
        area_b -
        intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def box_center(box):
    """
    Return center point of a bounding box.
    """

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def point_inside_box(point, box):
    """
    Check whether a point is inside a box.
    """

    px, py = point

    x1, y1, x2, y2 = box

    return (
        x1 <= px <= x2
        and
        y1 <= py <= y2
    )


def normalize_ppe_name(name):
    normalized_name = "".join(
        character
        for character in name.casefold()
        if character.isalnum()
    )
    aliases = {
        "boots": "Boots",
        "earprotection": "Ear-protection",
        "glass": "Glass",
        "goggle": "Glass",
        "goggles": "Glass",
        "glove": "Glove",
        "helmet": "Helmet",
        "mask": "Mask",
        "vest": "Vest",
    }
    return aliases.get(normalized_name, name)


def calculate_overlap_ratio(ppe_box, person_box):
    """
    Calculate how much of the PPE box
    overlaps the person box.
    """

    px1, py1, px2, py2 = person_box

    ppe_x1, ppe_y1, ppe_x2, ppe_y2 = ppe_box

    intersection_x1 = max(
        px1,
        ppe_x1
    )

    intersection_y1 = max(
        py1,
        ppe_y1
    )

    intersection_x2 = min(
        px2,
        ppe_x2
    )

    intersection_y2 = min(
        py2,
        ppe_y2
    )

    intersection_width = max(
        0,
        intersection_x2 -
        intersection_x1
    )

    intersection_height = max(
        0,
        intersection_y2 -
        intersection_y1
    )

    intersection_area = (
        intersection_width *
        intersection_height
    )

    ppe_width = max(
        0,
        ppe_x2 -
        ppe_x1
    )

    ppe_height = max(
        0,
        ppe_y2 -
        ppe_y1
    )

    ppe_area = (
        ppe_width *
        ppe_height
    )

    if ppe_area <= 0:
        return 0.0

    return intersection_area / ppe_area


def check_worker_ppe(
    person_box,
    ppe_detections,
    required_ppe=None
):
    """
    Determine which required PPE items
    belong to a particular worker.

    Required PPE (default):
        Helmet
        Vest
        Glove
    """
    status, missing, _ = get_worker_ppe_details(
        person_box,
        ppe_detections,
        required_ppe
    )
    return status, missing


def get_worker_ppe_details(
    person_box,
    ppe_detections,
    required_ppe=None
):
    """
    Determine which required PPE items
    belong to a particular worker and return full details:
    status, missing_list, detected_list
    """

    detected_ppe = set()

    for item in ppe_detections:

        class_name = normalize_ppe_name(item["class_name"])

        ppe_box = item["box"]


        # ------------------------------------------------
        # METHOD 1:
        # PPE center inside person box
        # ------------------------------------------------

        center = box_center(
            ppe_box
        )

        center_inside = point_inside_box(
            center,
            person_box
        )


        # ------------------------------------------------
        # METHOD 2:
        # PPE overlap with person
        # ------------------------------------------------

        overlap_ratio = calculate_overlap_ratio(
            ppe_box,
            person_box
        )


        # ------------------------------------------------
        # ACCEPT PPE IF EITHER CONDITION IS TRUE
        # ------------------------------------------------

        if (
            center_inside
            or
            overlap_ratio >= 0.20
        ):

            detected_ppe.add(
                class_name
            )


    # ----------------------------------------------------
    # REQUIRED PPE
    # ----------------------------------------------------

    if required_ppe is None:
        required_set = {
            "Helmet",
            "Vest",
            "Glove"
        }
    else:
        required_set = {
            normalize_ppe_name(name)
            for name in required_ppe
        }


    # ----------------------------------------------------
    # FIND MISSING PPE
    # ----------------------------------------------------

    missing = sorted(
        required_set -
        detected_ppe
    )


    # ----------------------------------------------------
    # STATUS
    # ----------------------------------------------------

    if missing:
        status = "VIOLATION"
    else:
        status = "SAFE"


    return status, missing, sorted(detected_ppe)