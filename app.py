from flask import Flask, render_template, Response, send_from_directory, request, jsonify
from ultralytics import YOLO

import cv2
import os
import time
import glob
import threading
import base64
import numpy as np

from safety_rules import check_worker_ppe, get_worker_ppe_details
from database import create_database, save_violation, get_violations


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Resolve the most recently trained V3 checkpoint across the existing run paths.
RUN_WEIGHTS_PATTERN = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "runs",
    "**",
    "ppe_safety_model_v*",
    "weights",
    "best.pt",
)
CANDIDATE_PATHS = sorted(
    glob.glob(RUN_WEIGHTS_PATTERN, recursive=True),
    key=os.path.getmtime,
    reverse=True,
)
CANDIDATE_PATHS.extend([
    os.path.join(BASE_DIR, "runs", "ppe_safety_model_v3", "weights", "best.pt"),
    os.path.join(BASE_DIR, "yolo11n.pt"),
])

MODEL_PATH = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), CANDIDATE_PATHS[0])

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

camera = None
camera_lock = threading.Lock()
monitoring_active = False
monitoring_generation = 0


# ============================================================
# VIOLATION COOLDOWN
# ============================================================

# Stores the last alert time for each worker
last_alert_times = {}

ALERT_COOLDOWN = 5


# ============================================================
# LIVE VIDEO GENERATOR
# ============================================================

def generate_frames(generation):
    global camera, monitoring_active

    while True:

        with camera_lock:
            if (
                not monitoring_active
                or generation != monitoring_generation
                or camera is None
            ):
                break

            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read camera frame.")
                camera.release()
                camera = None
                monitoring_active = False
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

        # Infer worker from detected PPE if explicit Person box is missing
        if len(people) == 0 and len(ppe_objects) > 0:
            frame_h, frame_w = frame.shape[:2]
            vests = [p for p in ppe_objects if p["class_name"] == "Vest"]
            helmets = [p for p in ppe_objects if p["class_name"] == "Helmet"]
            if vests:
                for idx_v, v in enumerate(vests):
                    vx1, vy1, vx2, vy2 = v["box"]
                    vh = vy2 - vy1
                    vw = vx2 - vx1
                    inferred_box = [
                        max(0.0, vx1 - vw * 0.45),
                        max(0.0, vy1 - vh * 0.55),
                        min(float(frame_w), vx2 + vw * 0.45),
                        min(float(frame_h), vy2 + vh * 1.6)
                    ]
                    people.append({"class_name": "Person", "box": inferred_box, "track_id": idx_v + 1})
            elif helmets:
                for idx_h, h in enumerate(helmets):
                    hx1, hy1, hx2, hy2 = h["box"]
                    hh = hy2 - hy1
                    hw = hx2 - hx1
                    inferred_box = [
                        max(0.0, hx1 - hw * 0.9),
                        hy1,
                        min(float(frame_w), hx2 + hw * 0.9),
                        min(float(frame_h), hy1 + hh * 5.5)
                    ]
                    people.append({"class_name": "Person", "box": inferred_box, "track_id": idx_h + 1})

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

        with camera_lock:
            if not monitoring_active or generation != monitoring_generation:
                break

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
    with camera_lock:
        if not monitoring_active or camera is None or not camera.isOpened():
            return jsonify({
                "success": False,
                "error": "Live monitoring is stopped. Start monitoring before opening the video feed."
            }), 409
        generation = monitoring_generation

    return Response(
        generate_frames(generation),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


@app.route("/api/monitoring/status", methods=["GET"])
def live_monitoring_status():
    with camera_lock:
        active = monitoring_active and camera is not None and camera.isOpened()
    return jsonify({"active": active})


@app.route("/api/monitoring/start", methods=["POST"])
def start_live_monitoring():
    global camera, monitoring_active, monitoring_generation

    with camera_lock:
        if camera is None or not camera.isOpened():
            new_camera = cv2.VideoCapture(0)
            if not new_camera.isOpened():
                new_camera.release()
                monitoring_active = False
                return jsonify({
                    "success": False,
                    "active": False,
                    "error": "Could not open camera. Check that a camera is connected and available."
                }), 503
            camera = new_camera

        monitoring_active = True
        monitoring_generation += 1

    print("Live monitoring started.")
    return jsonify({"success": True, "active": True})


@app.route("/api/monitoring/stop", methods=["POST"])
def stop_live_monitoring():
    global camera, monitoring_active, monitoring_generation

    with camera_lock:
        monitoring_active = False
        monitoring_generation += 1
        if camera is not None:
            camera.release()
            camera = None

    print("Live monitoring stopped.")
    return jsonify({"success": True, "active": False})


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
# PHOTO SAFETY ANALYSIS LOGIC
# ============================================================

PPE_COLORS = {
    "Helmet": (255, 200, 0),          # Cyan / Blue
    "Vest": (0, 165, 255),            # Vibrant Orange
    "Glove": (255, 0, 255),           # Magenta
    "Boots": (0, 220, 255),           # Bright Yellow
    "Ear-protection": (230, 130, 255),# Violet
    "Glass": (255, 190, 80),          # Sky Blue
    "Mask": (80, 240, 120),           # Mint / Emerald
}


def analyze_photo_safety(img_bgr, required_ppe=None, conf_threshold=0.15, save_to_db=False):
    """
    Run YOLO PPE detection and safety rules on an uploaded image,
    annotate with worker and PPE boxes, and optionally record violations.
    """
    if required_ppe is None:
        required_ppe = ["Helmet", "Vest", "Glove"]

    annotated = img_bgr.copy()
    results = model(img_bgr, conf=conf_threshold, verbose=False)
    result = results[0]

    people = []
    ppe_objects = []
    ppe_counts = {}

    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        coords = [float(v) for v in box.xyxy[0].cpu().numpy()]

        item = {
            "class_name": cls_name,
            "box": coords,
            "conf": conf
        }

        if cls_name == "Person":
            people.append(item)
        else:
            ppe_objects.append(item)
            ppe_counts[cls_name] = ppe_counts.get(cls_name, 0) + 1

    # Smart worker inference: if explicit Person box is missing but PPE was detected,
    # infer worker boundary from the detected safety equipment (e.g. Vest or Helmet)
    if len(people) == 0 and len(ppe_objects) > 0:
        img_h, img_w = img_bgr.shape[:2]
        vests = [p for p in ppe_objects if p["class_name"] == "Vest"]
        helmets = [p for p in ppe_objects if p["class_name"] == "Helmet"]

        if vests:
            for v in vests:
                vx1, vy1, vx2, vy2 = v["box"]
                vh = vy2 - vy1
                vw = vx2 - vx1
                inferred_box = [
                    max(0.0, vx1 - vw * 0.45),
                    max(0.0, vy1 - vh * 0.55),
                    min(float(img_w), vx2 + vw * 0.45),
                    min(float(img_h), vy2 + vh * 1.6)
                ]
                people.append({"class_name": "Person", "box": inferred_box, "conf": v["conf"], "inferred": True})
        elif helmets:
            for h in helmets:
                hx1, hy1, hx2, hy2 = h["box"]
                hh = hy2 - hy1
                hw = hx2 - hx1
                inferred_box = [
                    max(0.0, hx1 - hw * 0.9),
                    hy1,
                    min(float(img_w), hx2 + hw * 0.9),
                    min(float(img_h), hy1 + hh * 5.5)
                ]
                people.append({"class_name": "Person", "box": inferred_box, "conf": h["conf"], "inferred": True})
        else:
            all_boxes = [p["box"] for p in ppe_objects]
            min_x = min(b[0] for b in all_boxes)
            min_y = min(b[1] for b in all_boxes)
            max_x = max(b[2] for b in all_boxes)
            max_y = max(b[3] for b in all_boxes)
            people.append({"class_name": "Person", "box": [min_x, min_y, max_x, max_y], "conf": 0.5, "inferred": True})

    # Draw PPE bounding boxes first
    for item in ppe_objects:
        cls_name = item["class_name"]
        x1, y1, x2, y2 = map(int, item["box"])
        color = PPE_COLORS.get(cls_name, (200, 200, 200))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        lbl = f"{cls_name} {item['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        badge_y1 = max(0, y1 - th - 6)
        badge_y2 = max(th + 6, y1)
        cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + 6, badge_y2), color, -1)
        cv2.putText(
            annotated,
            lbl,
            (x1 + 3, max(th + 2, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    worker_results = []
    safe_count = 0
    violation_count = 0
    saved_count = 0

    for idx, person in enumerate(people):
        worker_id = idx + 1
        person_box = person["box"]
        status, missing, detected_ppe = get_worker_ppe_details(
            person_box,
            ppe_objects,
            required_ppe
        )

        x1, y1, x2, y2 = map(int, person_box)

        if status == "SAFE":
            safe_count += 1
            box_color = (34, 197, 94)    # Green
        else:
            violation_count += 1
            box_color = (0, 0, 235)      # Red

        # Worker bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)

        # Worker status banner
        worker_label = f"Worker {worker_id}: {status}"
        (w_w, w_h), _ = cv2.getTextSize(worker_label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        top_y = max(w_h + 8, y1)
        cv2.rectangle(annotated, (x1, top_y - w_h - 8), (x1 + w_w + 12, top_y), box_color, -1)
        cv2.putText(
            annotated,
            worker_label,
            (x1 + 6, top_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Missing alert label below box
        if missing:
            missing_text = "Missing: " + ", ".join(missing)
            (m_w, m_h), _ = cv2.getTextSize(missing_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            bot_y = min(annotated.shape[0] - 6, y2 + m_h + 10)
            cv2.rectangle(annotated, (x1, bot_y - m_h - 6), (x1 + m_w + 10, bot_y), (0, 0, 180), -1)
            cv2.putText(
                annotated,
                missing_text,
                (x1 + 5, bot_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Record violation in DB if requested
            if save_to_db:
                ts = time.strftime("%Y%m%d_%H%M%S")
                filename = f"upload_violation_w{worker_id}_{ts}_{idx}.jpg"
                image_path = os.path.join(ALERT_FOLDER, filename)
                cv2.imwrite(image_path, annotated)
                save_violation(
                    worker_id=worker_id,
                    missing_ppe=missing,
                    image_path=os.path.join("alerts", filename)
                )
                saved_count += 1

        worker_results.append({
            "worker_id": worker_id,
            "status": status,
            "missing": missing,
            "detected_ppe": detected_ppe,
            "box": [x1, y1, x2, y2]
        })

    # Encode annotated image to JPEG -> Base64
    _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    b64_image = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

    if len(people) == 0:
        overall_status = "NO WORKERS DETECTED"
    elif violation_count > 0:
        overall_status = "VIOLATION DETECTED"
    else:
        overall_status = "ALL WORKERS SAFE"

    return {
        "success": True,
        "annotated_image": b64_image,
        "summary": {
            "overall_status": overall_status,
            "total_workers": len(people),
            "safe_workers": safe_count,
            "violation_workers": violation_count,
            "saved_violations": saved_count,
            "required_ppe": list(required_ppe)
        },
        "workers": worker_results,
        "ppe_counts": ppe_counts
    }


# ============================================================
# PHOTO CHECK ROUTE
# ============================================================

@app.route("/check_photo", methods=["POST"])
def check_photo():
    try:
        # Check if user uploaded a file
        file = request.files.get("photo") or request.files.get("file")
        sample_name = request.form.get("sample")
        
        img_bgr = None

        if file and file.filename != "":
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif sample_name:
            sample_path = os.path.join(BASE_DIR, "static", "samples", os.path.basename(sample_name))
            if os.path.exists(sample_path):
                img_bgr = cv2.imread(sample_path)
            else:
                return jsonify({"success": False, "error": f"Sample image {sample_name} not found."}), 404
        elif request.is_json:
            data = request.get_json()
            if "sample" in data:
                sample_path = os.path.join(BASE_DIR, "static", "samples", os.path.basename(data["sample"]))
                if os.path.exists(sample_path):
                    img_bgr = cv2.imread(sample_path)
            elif "image_base64" in data:
                b64_str = data["image_base64"]
                if "," in b64_str:
                    b64_str = b64_str.split(",")[1]
                img_data = base64.b64decode(b64_str)
                file_bytes = np.frombuffer(img_data, np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({"success": False, "error": "No valid image provided. Please upload a photo."}), 400

        # Parse required PPE
        req_ppe_raw = request.form.get("required_ppe")
        if req_ppe_raw:
            if isinstance(req_ppe_raw, str):
                required_ppe = [p.strip() for p in req_ppe_raw.split(",") if p.strip()]
            else:
                required_ppe = list(req_ppe_raw)
        else:
            required_ppe = ["Helmet", "Vest", "Glove"]

        # Parse confidence
        try:
            conf = float(request.form.get("confidence", 0.15))
            conf = max(0.05, min(0.95, conf))
        except (ValueError, TypeError):
            conf = 0.15

        # Parse save_to_db flag
        save_to_db = str(request.form.get("save_to_db", "false")).lower() in ["true", "1", "yes"]

        result = analyze_photo_safety(
            img_bgr,
            required_ppe=required_ppe,
            conf_threshold=conf,
            save_to_db=save_to_db
        )

        return jsonify(result)

    except Exception as e:
        print(f"Error checking photo: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# SAMPLES LIST ROUTE
# ============================================================

@app.route("/api/samples", methods=["GET"])
def get_samples():
    samples_dir = os.path.join(BASE_DIR, "static", "samples")
    if not os.path.exists(samples_dir):
        return jsonify({"samples": []})

    samples = [
        {
            "filename": "sample_site_inspection.jpg",
            "title": "Site Inspection",
            "description": "Multi-worker construction site with helmets, vests & boots"
        },
        {
            "filename": "sample_worker_missing_helmet.jpg",
            "title": "Missing Helmet Violation",
            "description": "Worker with safety vest and boots but missing helmet"
        },
        {
            "filename": "sample_worker_helmet_vest.jpg",
            "title": "Helmet & Vest Compliance",
            "description": "Worker wearing helmet and high-visibility safety vest"
        }
    ]

    return jsonify({"samples": samples})


# ============================================================
# DYNAMIC STATS ROUTE
# ============================================================

@app.route("/api/stats", methods=["GET"])
def get_stats():
    violations = get_violations()
    helmet_count = sum(1 for v in violations if "Helmet" in v[3])
    vest_count = sum(1 for v in violations if "Vest" in v[3])
    glove_count = sum(1 for v in violations if "Glove" in v[3])

    return jsonify({
        "total_violations": len(violations),
        "helmet_violations": helmet_count,
        "vest_violations": vest_count,
        "glove_violations": glove_count,
        "recent_violations": [
            {
                "id": v[0],
                "timestamp": v[1],
                "worker_id": v[2],
                "missing_ppe": v[3],
                "image_path": v[4]
            }
            for v in violations[:10]
        ]
    })


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    print()
    print("================================")
    print("AI WORKPLACE SAFETY MONITORING")
    print("================================")
    print("Dashboard:")
    print("http://127.0.0.1:5001")
    print()
    print("Multi-worker tracking enabled.")
    print("Press CTRL+C to stop.")
    print()

    app.run(
        debug=False,
        threaded=True,
        port=5001
    )