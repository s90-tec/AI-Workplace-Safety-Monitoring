from flask import Flask, render_template, send_from_directory, Response
from database import get_violations
from ultralytics import YOLO
import cv2
import os


app = Flask(__name__)


# ==========================================
# PATHS
# ==========================================

MODEL_PATH = (
    "runs/detect/runs/ppe_safety_model-2/weights/best.pt"
)

ALERT_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "alerts"
)


# ==========================================
# LOAD YOLO MODEL
# ==========================================

model = YOLO(MODEL_PATH)


# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(0)


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
# LIVE CAMERA GENERATOR
# ==========================================

def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break


        # YOLO detection

        results = model(
            frame,
            conf=0.10
        )


        # Draw YOLO detections

        annotated_frame = results[0].plot()


        # Convert frame to JPEG

        ret, buffer = cv2.imencode(
            ".jpg",
            annotated_frame
        )


        frame = buffer.tobytes()


        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )


# ==========================================
# LIVE VIDEO ROUTE
# ==========================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
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
# RUN FLASK
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        threaded=True
    )