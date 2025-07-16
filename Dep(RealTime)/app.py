import os
import time
from datetime import datetime
from flask import (
    Flask, request, redirect, url_for, render_template,
    send_from_directory, Response, jsonify, copy_current_request_context
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models

from model import build_model
from ultralytics import YOLO

# ─── Config ─────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    SQLALCHEMY_DATABASE_URI=f"sqlite:///{os.path.join(BASE,'predictions.db')}",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
)
db = SQLAlchemy(app)

# ─── DB Model ───────────────────────────────────────────────
class Prediction(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    filename  = db.Column(db.String(256), nullable=False)
    verdict   = db.Column(db.String(32),  nullable=False)
    timestamp = db.Column(db.DateTime,    default=datetime.utcnow)

with app.app_context():
    db.create_all()

# ─── Load AI-vs-Real model ───────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = {
    'weights': models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    'finetuned_layers': 'all',
    'lr': 0.001,
    'momentum': 0.9,
    'optimizer': 'Adam'
}
model, _ = build_model(config, device, return_optimizer=True)
state_dict = torch.load("effbest_model_ai_detection.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device).eval()

# ─── Load YOLOv8 paper detection model ──────────────────────
paper_model = YOLO("paperDetection.pt")

# ─── Preprocessing ──────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ─── Real-time detection globals ────────────────────────────
CONF_THRESH     = 0.85
last_screenshot = 0.0

def generate_frames():
    global last_screenshot

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = paper_model(frame)[0]
        detected_paper = False

        if len(results.boxes) > 0:
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < 0.75:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f"Paper {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                cropped = frame[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue

                pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                inp = preprocess(pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    prob = model(inp).item()

                is_ai = prob > CONF_THRESH
                label = "AI-generated" if is_ai else "Real"
                color = (0, 0, 255) if is_ai else (0, 255, 0)
                text = f"{label} {prob:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                text_x = x2 - text_w
                text_y = y1 + text_h + 5
                cv2.putText(frame, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


                if is_ai and (time.time() - last_screenshot) >= 10.0:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"screenshot_{ts}.jpg"
                    path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
                    cv2.imwrite(path, frame)

                    try:
                        with app.app_context():
                            rec = Prediction(filename=fname, verdict=label)
                            db.session.add(rec)
                            db.session.commit()
                        last_screenshot = time.time()
                    except Exception as e:
                        app.logger.error("Auto-screenshot DB write failed: %s", e)

                detected_paper = True

        if not detected_paper:
            cv2.putText(frame, "No paper detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        ret, buf = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    cap.release()


# ─── Routes ─────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return "No file provided", 400

        fname = secure_filename(file.filename)
        path  = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        file.save(path)

        img = Image.open(path).convert("RGB")
        inp = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = model(inp).item()
        verdict = "AI-generated" if prob > CONF_THRESH else "Real"

        rec = Prediction(filename=fname, verdict=verdict)
        db.session.add(rec)
        db.session.commit()
        return redirect(url_for("result", pred_id=rec.id))

    return render_template("upload.html")


@app.route("/result/<int:pred_id>")
def result(pred_id):
    rec = Prediction.query.get_or_404(pred_id)
    return render_template("result.html", record=rec)


@app.route("/history")
def history():
    recs = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    return render_template("history.html", records=recs)


@app.route("/camera")
def camera():
    return render_template("camera.html")


@app.route("/video_feed")
def video_feed():
    @copy_current_request_context
    def inner():
        yield from generate_frames()
    return Response(inner(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/last_detection")
def last_detection():
    rec = (
        Prediction.query
        .filter_by(verdict="AI-generated")
        .order_by(Prediction.timestamp.desc())
        .first()
    )
    if not rec:
        return jsonify(detected=False)
    return jsonify(
        detected=True,
        label=rec.verdict,
        filename=rec.filename,
        timestamp=rec.timestamp.isoformat()
    )


@app.route("/manual_snapshot", methods=["POST"])
def manual_snapshot():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return jsonify(status="error", message="Cannot open camera"), 500
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return jsonify(status="error", message="Cannot read frame"), 500

    # Just classify full frame (optionally adapt for paper detection)
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inp = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = model(inp).item()

    label = "AI-generated" if prob > CONF_THRESH else "Real"
    color = (0, 0, 255) if label == "AI-generated" else (0, 255, 0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"manual_{ts}.jpg"
    path = os.path.join(app.config["UPLOAD_FOLDER"], fname)

    cv2.putText(frame, f"{label} {prob:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imwrite(path, frame)

    rec = Prediction(filename=fname, verdict=label)
    db.session.add(rec)
    db.session.commit()
    return jsonify(status="ok", filename=fname)


@app.route("/delete/<int:rec_id>", methods=["POST"])
def delete(rec_id):
    rec = Prediction.query.get_or_404(rec_id)
    try:
        os.remove(os.path.join(app.config["UPLOAD_FOLDER"], rec.filename))
    except OSError:
        pass

    db.session.delete(rec)
    db.session.commit()
    return redirect(url_for("history"))


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
