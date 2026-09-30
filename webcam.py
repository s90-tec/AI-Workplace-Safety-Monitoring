import cv2
import os
from datetime import datetime
from ultralytics import YOLO

from safety_rules import check_worker_ppe
from database import create_database, save_violation


# ==============================
# MODEL PATH
# ==============================

MODEL_PATH = "runs/detect/runs/ppe_safety_model_v3/weights/best.pt"


# ==============================
# LOAD MODEL
# ==============================

model = YOLO(MODEL_PATH)

create_database()


# ==============================
# CREATE ALERTS FOLDER
# ==============================

os.makedirs("alerts", exist_ok=True)


# ==============================
# OPEN WEBCAM
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not access the camera.")
    raise SystemExit


# ==============================
# ALERT SETTINGS
# ==============================

last_alert_time = 0
alert_cooldown = 5


# ==============================
# MAIN LOOP
# ==============================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame.")
        break


    # ==============================
    # YOLO DETECTION
    # ==============================

    result = model(frame, conf=0.10)[0]

    people = []
    ppe_objects = []


    # ==============================
    # PROCESS DETECTIONS
    # ==============================

    for box in result.boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        detection = {
            "class_name": class_name,
            "box": [x1, y1, x2, y2]
        }


        # Detect person
        if class_name == "Person":

            people.append(detection)


        # Detect PPE
        elif class_name in {
            "Helmet",
            "Vest",
            "Glove",
            "Boots",
            "Ear-protection",
            "Glass",
            "Mask"
        }:

            ppe_objects.append(detection)


    # ==============================
    # DRAW YOLO DETECTIONS
    # ==============================

    annotated_frame = result.plot()


    # ==============================
    # CHECK EACH WORKER
    # ==============================

    for index, person in enumerate(people):

        person_box = person["box"]

        status, missing = check_worker_ppe(
            person_box,
            ppe_objects
        )

        x1, y1, x2, y2 = map(
            int,
            person_box
        )


        # Green = SAFE
        # Red = VIOLATION

        color = (
            (0, 255, 0)
            if status == "SAFE"
            else (0, 0, 255)
        )


        # ==============================
        # DISPLAY WORKER STATUS
        # ==============================

        cv2.putText(
            annotated_frame,
            f"Worker {index + 1}: {status}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )


        # ==============================
        # DISPLAY MISSING PPE
        # ==============================

        if missing:

            text = "Missing: " + ", ".join(missing)

            cv2.putText(
                annotated_frame,
                text,
                (
                    x1,
                    min(
                        annotated_frame.shape[0] - 10,
                        y2 + 25
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )


            # ==============================
            # SAVE VIOLATION
            # ==============================

            current_time = datetime.now().timestamp()

            if current_time - last_alert_time >= alert_cooldown:

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )

                filename = (
                    f"alerts/violation_{timestamp}.jpg"
                )


                # Save evidence image

                cv2.imwrite(
                    filename,
                    annotated_frame
                )


                # Save violation to database

                save_violation(
                    worker_id=index + 1,
                    missing_ppe=missing,
                    image_path=filename
                )


                print(
                    f"⚠️ Violation saved: {filename}"
                )

                print(
                    f"📋 Database updated for Worker {index + 1}"
                )


                last_alert_time = current_time


    # ==============================
    # SHOW CAMERA WINDOW
    # ==============================

    cv2.imshow(
        "AI Workplace Safety Monitoring",
        annotated_frame
    )


    # ==============================
    # PRESS Q TO EXIT
    # ==============================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==============================
# RELEASE CAMERA
# ==============================

cap.release()

cv2.destroyAllWindows()