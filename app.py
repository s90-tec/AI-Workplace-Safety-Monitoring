from flask import Flask, render_template, Response, send_from_directory
from ultralytics import YOLO

import cv2
import os
import time

from safety_rules import check_worker_ppe
from database import create_database, save_violation, get_violations


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "runs",
    "ppe_safety_model_v3",
    "weights",
    "best.pt"
)

ALERT_FOLDER = os.path.join(
    BASE_DIR,
    "alerts"
)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

os.makedirs(ALERT_FOLDER, exist_ok=True)


# ============================================================
# DATABASE
# ============================================================

create_database()


# ============================================================
# YOLO MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
else:
    print("Camera opened successfully.")


# ============================================================
# VIOLATION COOLDOWN
# ============================================================

# Stores the last alert time for each worker
last_alert_times = {}

ALERT_COOLDOWN = 5


# ============================================================
# LIVE VIDEO GENERATOR
# ============================================================

def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera frame.")
            break


        # ----------------------------------------------------
        # YOLO TRACKING
        # ----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            conf=0.10,
            tracker="bytetrack.yaml"
        )

        result = results[0]


        # ----------------------------------------------------
        # DETECTION STORAGE
        # ----------------------------------------------------

        people = []
        ppe_objects = []


        # ----------------------------------------------------
        # GET TRACKING IDs
        # ----------------------------------------------------

        track_ids = result.boxes.id

        if track_ids is not None:
            track_ids = track_ids.cpu().numpy()


        # ----------------------------------------------------
        # PROCESS DETECTIONS
        # ----------------------------------------------------

        for index, box in enumerate(result.boxes):

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )


            # Get tracking ID
            if track_ids is not None:
                track_id = int(track_ids[index])
            else:
                track_id = index + 1


            detection = {
                "class_name": class_name,
                "box": [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2)
                ],
                "track_id": track_id
            }


            # ------------------------------------------------
            # PERSON
            # ------------------------------------------------

            if class_name == "Person":

                people.append(detection)


            # ------------------------------------------------
            # PPE
            # ------------------------------------------------

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


        # ----------------------------------------------------
        # CREATE ANNOTATED FRAME
        # ----------------------------------------------------

        annotated_frame = result.plot()


        # ----------------------------------------------------
        # WORKER PROCESSING
        # ----------------------------------------------------

        for worker_index, person in enumerate(people):

            person_box = person["box"]

            worker_id = person["track_id"]


            # ------------------------------------------------
            # SAFETY CHECK
            # ------------------------------------------------

            status, missing = check_worker_ppe(
                person_box,
                ppe_objects
            )


            # ------------------------------------------------
            # PERSON BOX
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                person_box
            )


            # ------------------------------------------------
            # STATUS COLOR
            # ------------------------------------------------

            if status == "SAFE":

                color = (0, 255, 0)

            else:

                color = (0, 0, 255)


            # ------------------------------------------------
            # WORKER STATUS
            # ------------------------------------------------

            cv2.putText(
                annotated_frame,
                f"Worker {worker_id}: {status}",
                (
                    x1,
                    max(30, y1 - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )


            # ------------------------------------------------
            # MISSING PPE
            # ------------------------------------------------

            if missing:

                missing_text = (
                    "Missing: "
                    + ", ".join(missing)
                )


                cv2.putText(
                    annotated_frame,
                    missing_text,
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


                # ------------------------------------------------
                # ALERT COOLDOWN FOR EACH WORKER
                # ------------------------------------------------

                current_time = time.time()

                last_time = last_alert_times.get(
                    worker_id,
                    0
                )


                if (
                    current_time - last_time
                    >= ALERT_COOLDOWN
                ):

                    # ------------------------------------------------
                    # CREATE TIMESTAMP
                    # ------------------------------------------------

                    timestamp = time.strftime(
                        "%Y%m%d_%H%M%S"
                    )


                    filename = (
                        f"violation_worker_"
                        f"{worker_id}_"
                        f"{timestamp}.jpg"
                    )


                    image_path = os.path.join(
                        ALERT_FOLDER,
                        filename
                    )


                    # ------------------------------------------------
                    # SAVE EVIDENCE IMAGE
                    # ------------------------------------------------

                    saved = cv2.imwrite(
                        image_path,
                        annotated_frame
                    )


                    if saved:

                        print()
                        print(
                            "================================"
                        )
                        print(
                            "VIOLATION SAVED"
                        )
                        print(
                            f"Worker: {worker_id}"
                        )
                        print(
                            f"Missing: {missing}"
                        )
                        print(
                            f"Image: {image_path}"
                        )


                        # ------------------------------------------------
                        # SAVE DATABASE RECORD
                        # ------------------------------------------------

                        save_violation(
                            worker_id=worker_id,
                            missing_ppe=missing,
                            image_path=os.path.join(
                                "alerts",
                                filename
                            )
                        )


                        print(
                            "Database updated successfully."
                        )
                        print(
                            "================================"
                        )
                        print()


                        # Update worker cooldown
                        last_alert_times[
                            worker_id
                        ] = current_time


        # ----------------------------------------------------
        # ENCODE FRAME FOR BROWSER
        # ----------------------------------------------------

        success, buffer = cv2.imencode(
            ".jpg",
            annotated_frame
        )


        if not success:
            continue


        frame_bytes = buffer.tobytes()


        # ----------------------------------------------------
        # SEND FRAME TO FLASK
        # ----------------------------------------------------

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    violations = get_violations()


    helmet_count = 0
    vest_count = 0
    glove_count = 0


    for violation in violations:

        missing_ppe = violation[3]


        if "Helmet" in missing_ppe:
            helmet_count += 1


        if "Vest" in missing_ppe:
            vest_count += 1


        if "Glove" in missing_ppe:
            glove_count += 1


    statistics = {

        "total_violations":
            len(violations),

        "helmet_violations":
            helmet_count,

        "vest_violations":
            vest_count,

        "glove_violations":
            glove_count
    }


    return render_template(
        "dashboard.html",
        violations=violations,
        statistics=statistics
    )


# ============================================================
# LIVE VIDEO ROUTE
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# ALERT IMAGE ROUTE
# ============================================================

@app.route("/alerts/<filename>")
def alert_image(filename):

    return send_from_directory(
        ALERT_FOLDER,
        filename
    )


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    print()
    print("================================")
    print("AI WORKPLACE SAFETY MONITORING")
    print("================================")
    print("Dashboard:")
    print("http://127.0.0.1:5000")
    print()
    print("Multi-worker tracking enabled.")
    print("Press CTRL+C to stop.")
    print()

    app.run(
        debug=False,
        threaded=True
    )