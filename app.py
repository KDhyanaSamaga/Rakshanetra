import pathlib
# (Optional) Fix for checkpoints saved on POSIX when loading on Windows
# Uncomment only if you get path-related errors when loading the model.
# pathlib.PosixPath = pathlib.WindowsPath

from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import os
import numpy as np
import base64
import time
import requests
from werkzeug.utils import secure_filename

# ==========================
# CONFIGURATION
# ==========================

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = os.path.join("static", "outputs")
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

# Your Telegram details
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

#TELEGRAM_BOT_TOKEN = "8007824138:AAF2KQNqwplfzuac8kqHmSjM_nhRk1egKYU"
#TELEGRAM_CHAT_ID  = "6360921508"

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================
# LOAD YOLO11 MODEL (NEW_MODEL.pt)
# ==========================

# Path to your trained YOLO11 weights
model_path = r"D:\UI\yolov5\NEW_MODEL.pt"  # keep your path as is

# Use the Ultralytics YOLO class instead of torch.hub
model = YOLO(model_path)
# If you want to force CPU:
# model.to('cpu')


# ==========================
# HELPER FUNCTIONS
# ==========================

def allowed_file(filename):
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""
    return ext in ALLOWED_IMAGE_EXTENSIONS.union(ALLOWED_VIDEO_EXTENSIONS)


def is_image_file(filename):
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""
    return ext in ALLOWED_IMAGE_EXTENSIONS


def get_class_name(cls_idx: int) -> str:
    """Get readable class name from model.names."""
    names = model.names  # dict {id: name}
    try:
        return str(names[int(cls_idx)])
    except Exception:
        return str(int(cls_idx))


def send_telegram_alert(detections, output_path):
    """
    Send Telegram alert WITH IMAGE/VIDEO ONLY IF class is:
    Elephant, Leopard or Wild Boar.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    if not detections:
        return

    # Allowed alert classes (case insensitive, include typo variants)
    ALERT_CLASSES = {
        "elephant",
        "leopard", "leopord", "leapord",
        "wildboar", "wild boar", "wild_bore", "wildbore", "wild bore"
    }

    # Check if any detection belongs to alert classes
    alert_species = []
    for det in detections:
        cls = det["class"].strip().lower()
        if cls in ALERT_CLASSES:
            alert_species.append(det["class"])

    if not alert_species:
        return  # No Elephant, No Leopard, No Wild Boar → No Telegram Alert

    # Remove leading slash from URL/path to get local FS path
    file_path = output_path.lstrip("/")

    # Create caption
    caption = (
        "🚨 *Wildlife Alert Detected!* 🚨\n\n"
        f"Detected: {', '.join(sorted(set(alert_species)))}\n"
        "Stay safe and alert!"
    )

    # Determine if it's image or video
    files = None
    data = None
    telegram_url = None

    if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
        telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        files = {"photo": open(file_path, "rb")}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "Markdown"}

    elif file_path.lower().endswith(".mp4"):
        telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
        files = {"video": open(file_path, "rb")}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "Markdown"}

    else:
        return  # Unsupported file type

    try:
        requests.post(telegram_url, data=data, files=files)
    except Exception as e:
        print("Failed to send Telegram media:", e)
    finally:
        # Close file handle if opened
        if files:
            for f in files.values():
                try:
                    f.close()
                except Exception:
                    pass


def draw_detections_on_image(image_bgr, result):
    """
    Draw bounding boxes and labels on a BGR image using Ultralytics YOLO result.
    'result' is a single ultralytics.engine.results.Results object.
    """
    annotated = image_bgr.copy()
    detections = []

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections, annotated

    xyxy = boxes.xyxy.cpu().numpy()   # [N,4]
    confs = boxes.conf.cpu().numpy()  # [N]
    clss = boxes.cls.cpu().numpy()    # [N]

    for (x1, y1, x2, y2), conf, cls_idx in zip(xyxy, confs, clss):
        cls_idx = int(cls_idx)
        class_name = get_class_name(cls_idx)

        detections.append(
            {
                "class": class_name,
                "confidence": round(float(conf), 3),
                "box": [
                    round(float(x1), 1),
                    round(float(y1), 1),
                    round(float(x2), 1),
                    round(float(y2), 1),
                ],
            }
        )

        # Draw box
        x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(annotated, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)

        label = f"{class_name} {conf:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1_i, max(0, y1_i - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return detections, annotated


def run_inference_on_image(image_bgr, output_filename):
    """Run YOLO11 on a single BGR image and save annotated result."""
    # Ultralytics returns a list of Results -> take [0]
    result = model(image_bgr, verbose=False)[0]
    detections, annotated_bgr = draw_detections_on_image(image_bgr, result)

    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_path, annotated_bgr)
    return detections, output_path


def run_inference_on_video(video_path, output_filename):
    """Run YOLO11 on every frame of a video and save annotated video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open uploaded video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25  # fallback

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

    all_detections = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Run YOLO on BGR frame
        result = model(frame, verbose=False)[0]

        # Draw boxes
        detections, annotated_bgr = draw_detections_on_image(frame, result)
        all_detections.extend(detections)

        # Ensure size matches writer
        if annotated_bgr.shape[1] != width or annotated_bgr.shape[0] != height:
            annotated_bgr = cv2.resize(annotated_bgr, (width, height))

        writer.write(annotated_bgr)

    cap.release()
    writer.release()

    # De-duplicate detections
    unique_detections = {}
    for det in all_detections:
        key = (det["class"], tuple(det["box"]))
        if key not in unique_detections:
            unique_detections[key] = det

    return list(unique_detections.values()), output_path


def decode_webcam_image(data_url):
    """Decode a base64 dataURL from the browser to a BGR image."""
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    img_bytes = base64.b64decode(encoded)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


# ==========================
# ROUTES
# ==========================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    file = request.files.get("file")
    webcam_image = request.form.get("webcam_image")

    if not file and not webcam_image:
        return jsonify({"status": "error", "message": "No file or webcam image provided"}), 400

    timestamp = int(time.time() * 1000)

    detections = []
    output_rel_path = ""
    media_type = "image"
    output_path = ""

    try:
        # From webcam snapshot
        if webcam_image:
            image_bgr = decode_webcam_image(webcam_image)
            if image_bgr is None:
                return jsonify({"status": "error", "message": "Invalid webcam image"}), 400

            output_filename = f"webcam_{timestamp}.jpg"
            detections, output_path = run_inference_on_image(image_bgr, output_filename)
            output_rel_path = "/" + output_path.replace("\\", "/")
            media_type = "image"

        # From uploaded file
        elif file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            ext = filename.rsplit(".", 1)[1].lower()
            temp_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{filename}")
            file.save(temp_path)

            if ext in ALLOWED_IMAGE_EXTENSIONS:
                image_bgr = cv2.imread(temp_path)
                if image_bgr is None:
                    return jsonify({"status": "error", "message": "Could not read image file"}), 400

                output_filename = f"img_{timestamp}.jpg"
                detections, output_path = run_inference_on_image(image_bgr, output_filename)
                media_type = "image"
            else:
                output_filename = f"vid_{timestamp}.mp4"
                detections, output_path = run_inference_on_video(temp_path, output_filename)
                media_type = "video"

            output_rel_path = "/" + output_path.replace("\\", "/")

        else:
            return jsonify({"status": "error", "message": "File type not allowed"}), 400

        # Send Telegram if anything detected (uses local path, not URL)
        if output_path:
            send_telegram_alert(detections, output_path)

        return jsonify(
            {
                "status": "success",
                "type": media_type,
                "output_url": output_rel_path,
                "detections": detections,
            }
        )
    except Exception as e:
        print("Error during detection:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    # Use debug=False in production
    app.run(host="0.0.0.0", port=5000, debug=True)
