from flask import Flask, render_template, Response, jsonify, request,Blueprint
import cv2
import time
import random
import requests
import base64
import numpy as np
from datetime import datetime

# ===== IMPORT YOUR EXISTING MODULES =====
from App.routes.yolo_detector import detect_objects
from App.routes.gemini_explainer import explain

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ================= CAMERA SETUP =================
camera = cv2.VideoCapture(0)
latest_objects = []

def generate_frames():
    global latest_objects

    while True:
        success, frame = camera.read()
        if not success:
            break

        # YOLO detection
        latest_objects, frame = detect_objects(frame)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes +
            b"\r\n"
        )

# ================= WEB PAGES =================
main_bp = Blueprint("main", __name__)

@main_bp.route("/")
def home():
    return render_template("index.html")

@main_bp.route("/about")
def about():
    return render_template("About.html")

@main_bp.route("/contact", methods=["GET", "POST"])
def contact():
    return render_template("Contact.html")


# ================= CAMERA STREAM =================

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ================= SIMPLE SYSTEM MESSAGE =================

@app.route("/api/system-message")
def system_message():
    if latest_objects:
        return jsonify({
            "status": "success",
            "message": f"I can see: {', '.join(latest_objects)}",
            "timestamp": datetime.now().isoformat()
        })
    else:
        return jsonify({
            "status": "success",
            "message": "No objects detected",
            "timestamp": datetime.now().isoformat()
        })

# ================= GEMINI FRAME ANALYSIS =================

GEMINI_API_KEY = "AIzaSyBfSvWYmVOREYjYAZu0G0ai5iLbsrB_COo"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-2.0-flash:generateContent"
)

@main_bp.route("/api/analyze-frame", methods=["POST"])
def analyze_frame():
    try:
        data = request.get_json()
        image_data = data.get("image_data")

        if not image_data:
            return jsonify({"status": "error", "error": "No image"}), 400

        # Decode base64 image
        image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        np_img = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"status": "error", "error": "Invalid image"}), 400

        # 🔥 YOLO DETECTION
        detected_objects = detect_objects(frame)

        if len(detected_objects) > 0:
            message = "I can see " + ", ".join(detected_objects)
        else:
            message = "I do not see any objects."
        
        return jsonify({
            "status": "ok",
            "audioDescription": message,
            "detectedObjects": detected_objects
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# ================= API TEST =================

@main_bp.route("/api/test-connection")
def test_connection():
    try:
        r = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}",
            timeout=5
        )

        if r.status_code == 200:
            return jsonify({
                "status": "key_valid",
                "message": "API key is valid",
                "timestamp": datetime.now().isoformat()
            })
        elif r.status_code == 429:
            return jsonify({
                "status": "quota_exceeded",
                "message": "Quota exceeded today",
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "status": "key_issue",
                "message": f"Status code: {r.status_code}"
            }), 500

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ================= RUN =================

if __name__ == "__main__":
    app.run(debug=True)