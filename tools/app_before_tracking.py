from flask import Flask, render_template, send_from_directory, Response
from database import get_violations, create_database, save_violation
from safety_rules import check_worker_ppe
from ultralytics import YOLO

import cv2
import os
import time
from datetime import datetime


# ==========================================
# FLASK
# ==========================================

app = Flask(__name__)


# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "runs",
    "ppe_safety_model-2",
    "weights",
    "best.pt"
)

ALERT_FOLDER = os.path.join(
    BASE_DIR,
    "alerts"
)


# ==========================================
# CREATE ALERT FOLDER
# ==========================================

os.makedirs(
    ALERT_FOLDER,
    exist_ok=True
)


# ==========================================
# DATABASE
# ==========================================

create_database()


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")


# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print("ERROR: Camera could not be opened.")

else:

    print("Camera opened successfully.")


# ==========================================
# ALERT CONTROL
# ==========================================

last_alert_time = 0

ALERT_COOLDOWN = 5


# ==========================================
# DASHBOARD
# ==========================================

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


# ==========================================
# LIVE FRAME GENERATOR
# ==========================================

def generate_frames():

    global last_alert_time


    while True:


        success, frame = camera.read()


        if not success:

            print(
                "ERROR: Could not read camera frame."
            )

            break


        # ==================================
        # YOLO
        # ==================================

        result = model(

            frame,

            conf=0.10,

            verbose=False

        )[0]


        # ==================================
        # DETECTION LISTS
        # ==================================

        people = []

        ppe_objects = []


        # ==================================
        # READ DETECTIONS
        # ==================================

        for box in result.boxes:


            class_id = int(
                box.cls[0]
            )


            class_name = model.names[
                class_id
            ]


            x1, y1, x2, y2 = (

                box.xyxy[0]
                .cpu()
                .numpy()

            )


            detection = {

                "class_name":
                    class_name,

                "box":
                    [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2)
                    ]

            }


            if class_name == "Person":

                people.append(
                    detection
                )


            elif class_name in {

                "Helmet",
                "Vest",
                "Glove",
                "Boots",
                "Ear-protection",
                "Glass",
                "Mask"

            }:

                ppe_objects.append(
                    detection
                )


        # ==================================
        # DEBUG INFORMATION
        # ==================================

        if people:

            print(
                "Person detected:",
                len(people)
            )

            print(
                "PPE detected:",
                [
                    item["class_name"]
                    for item in ppe_objects
                ]
            )


        # ==================================
        # DRAW YOLO
        # ==================================

        annotated_frame = result.plot()


        # ==================================
        # CHECK WORKERS
        # ==================================

        for index, person in enumerate(
            people
        ):


            person_box = person["box"]


            status, missing = (
                check_worker_ppe(

                    person_box,

                    ppe_objects

                )
            )


            # ==================================
            # WORKER BOX
            # ==================================

            x1, y1, x2, y2 = map(
                int,
                person_box
            )


            # ==================================
            # STATUS COLOR
            # ==================================

            if status == "SAFE":

                color = (
                    0,
                    255,
                    0
                )

            else:

                color = (
                    0,
                    0,
                    255
                )


            # ==================================
            # STATUS
            # ==================================

            cv2.putText(

                annotated_frame,

                f"Worker {index + 1}: {status}",

                (
                    x1,
                    max(
                        30,
                        y1 - 10
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                color,

                2

            )


            # ==================================
            # VIOLATION
            # ==================================

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

                    (
                        0,
                        0,
                        255
                    ),

                    2

                )


                # ==================================
                # CHECK COOLDOWN
                # ==================================

                current_time = time.time()


                if (

                    current_time
                    - last_alert_time

                    >= ALERT_COOLDOWN

                ):


                    # ==================================
                    # FILE NAME
                    # ==================================

                    timestamp = (

                        datetime.now()
                        .strftime(
                            "%Y%m%d_%H%M%S"
                        )

                    )


                    filename = (

                        "violation_"

                        + timestamp

                        + ".jpg"

                    )


                    image_path = os.path.join(

                        ALERT_FOLDER,

                        filename

                    )


                    # ==================================
                    # SAVE IMAGE
                    # ==================================

                    saved = cv2.imwrite(

                        image_path,

                        annotated_frame

                    )


                    if saved:

                        print(
                            "================================"
                        )

                        print(
                            "VIOLATION SAVED"
                        )

                        print(
                            "Worker:",
                            index + 1
                        )

                        print(
                            "Missing:",
                            ", ".join(missing)
                        )

                        print(
                            "Image:",
                            image_path
                        )

                        print(
                            "================================"
                        )


                        # ==================================
                        # DATABASE
                        # ==================================

                        save_violation(

                            worker_id=index + 1,

                            missing_ppe=missing,

                            image_path=(
                                "alerts/"
                                + filename
                            )

                        )


                        print(
                            "Database updated successfully."
                        )


                        last_alert_time = (
                            current_time
                        )


                    else:

                        print(
                            "ERROR: Image could not be saved."
                        )


        # ==================================
        # ENCODE IMAGE
        # ==================================

        success, buffer = cv2.imencode(

            ".jpg",

            annotated_frame

        )


        if not success:

            continue


        frame_bytes = (
            buffer.tobytes()
        )


        # ==================================
        # SEND TO BROWSER
        # ==================================

        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n\r\n"

            + frame_bytes

            + b"\r\n"

        )


# ==========================================
# VIDEO FEED
# ==========================================

@app.route("/video_feed")
def video_feed():

    return Response(

        generate_frames(),

        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )

    )


# ==========================================
# ALERT IMAGE
# ==========================================

@app.route("/alerts/<filename>")
def alert_image(filename):

    return send_from_directory(

        ALERT_FOLDER,

        filename

    )


# ==========================================
# START APPLICATION
# ==========================================

if __name__ == "__main__":

    print("")
    print("======================================")
    print(" AI WORKPLACE SAFETY MONITORING")
    print("======================================")
    print("")
    print("Dashboard:")
    print("http://127.0.0.1:5000")
    print("")
    print("Alerts folder:")
    print(ALERT_FOLDER)
    print("")

    app.run(

        debug=False,

        threaded=True

    )