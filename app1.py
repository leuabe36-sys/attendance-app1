from flask import Flask, request, jsonify, Response, redirect, session, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os
import base64
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta, timezone
import requests as http_requests
import smtplib
import secrets
from email.mime.text import MIMEText

app = Flask(__name__)

# SECRET_KEY must be set as a real environment variable on Vercel.
# No hardcoded fallback — a guessable fallback would let anyone forge sessions.
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

# =========================================================
# EMAIL (SMTP) CONFIG — for school registration verification
# =========================================================
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("SMTP_FROM", SMTP_USER)
APP_BASE_URL = os.environ.get("APP_BASE_URL", "")

def email_is_configured():
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASSWORD)

def send_email(to_email, subject, body):
    if not email_is_configured():
        print(f"[email_disabled] Would send to {to_email}: {subject}\n{body}", flush=True)
        return False, "Email is not configured on this server (missing SMTP_HOST/SMTP_USER/SMTP_PASSWORD)."
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = SMTP_FROM
        msg["To"] = to_email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, [to_email], msg.as_string())
        print(f"[email_sent] To {to_email}: {subject}", flush=True)
        return True, ""
    except Exception as e:
        print("Email send failed:", repr(e), flush=True)
        return False, str(e)

# DATABASE_URL must be set as a real environment variable on Vercel.
# No hardcoded fallback — the previous fallback embedded a live-looking DB password in source.
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# =========================================================
# SUPABASE STORAGE CONFIG
# =========================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET = "student-images"

def supabase_is_configured():
    return bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)

def supabase_upload(filename, image_bytes, content_type="image/jpeg"):
    if not supabase_is_configured():
        print("Supabase upload skipped: SUPABASE_URL/SUPABASE_SERVICE_KEY not set", flush=True)
        return None
    try:
        url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{filename}"
        headers = {
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "apikey": SUPABASE_SERVICE_KEY,
            "Content-Type": content_type,
            "x-upsert": "true"
        }
        resp = http_requests.put(url, headers=headers, data=image_bytes, timeout=30)
        if resp.status_code in (200, 201):
            return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"

        resp2 = http_requests.post(url, headers=headers, data=image_bytes, timeout=30)
        if resp2.status_code in (200, 201):
            return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"
        return None
    except Exception as e:
        print("Supabase upload exception:", e)
        return None

def supabase_delete(filename):
    if not supabase_is_configured():
        return
    try:
        url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{filename}"
        headers = {"Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        http_requests.delete(url, headers=headers, timeout=10)
    except Exception as e:
        print("Supabase delete exception:", e)

def supabase_public_url(filename):
    if not filename:
        return ""
    if filename.startswith("http"):
        return filename
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"

def generate_video_thumbnail(video_bytes, ext):
    """
    Extracts a frame using OpenCV. Runs strictly inside Vercel's temporary directory write space (/tmp).
    """
    import tempfile
    tmp_path = None
    try:
        suffix = f".{ext}" if ext else ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, dir="/tmp", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return None

        frame = None
        for frame_idx in (5, 0):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frame = candidate
                break
        cap.release()

        if frame is None:
            return None

        h, w = frame.shape[:2]
        max_dim = 480
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return None
        return buf.tobytes()
    except Exception as e:
        print("Video thumbnail generation error:", e, flush=True)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def maybe_generate_and_upload_thumb(storage_name, file_bytes, ext):
    video_exts = ("mp4", "mov", "avi", "webm", "mkv")
    if (ext or "").lower() not in video_exts:
        return ""
    thumb_bytes = generate_video_thumbnail(file_bytes, ext)
    if not thumb_bytes:
        return ""
    thumb_name = f"{storage_name}.thumb.jpg"
    thumb_url = supabase_upload(thumb_name, thumb_bytes, "image/jpeg")
    return thumb_url or ""

# =========================================================
# FACE PROCESSING STUBS (Vercel-compatible cloud approach)
# =========================================================
def _get_face_embedding(rgb_image):
    """
    MediaPipe modules cannot boot on Vercel due to strict runtime sizing limitations.
    Calculate face landmark arrays using front-end client browser JavaScript tools
    instead and transmit matrices directly via JSON data payloads.
    """
    print("MediaPipe visual alignment operations safely bypassed for serverless performance.")
    return None

def _compare_embeddings(known_embedding, unknown_embedding, tolerance=0.6):
    dist = np.linalg.norm(known_embedding - unknown_embedding)
    return dist < tolerance, dist

# =========================================================
# FLASK ROUTING HANDLERS WITH DYNAMIC TIMESTAMPS
# =========================================================
@app.route('/upload', methods=['POST'])
def handle_upload():
    if not supabase_is_configured():
        return jsonify({"error": "Server is missing SUPABASE_URL/SUPABASE_SERVICE_KEY configuration"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part found in request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    base_name = secure_filename(file.filename)
    name_part, ext_part = os.path.splitext(base_name)
    ext = ext_part.lstrip('.')

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    timestamped_filename = f"{name_part}_{timestamp}.{ext}"
    file_bytes = file.read()

    public_url = supabase_upload(timestamped_filename, file_bytes, file.content_type)
    if not public_url:
        return jsonify({"error": "Asset cloud upload interaction failed"}), 500

    thumb_url = maybe_generate_and_upload_thumb(f"{name_part}_{timestamp}", file_bytes, ext)

    return jsonify({
        "status": "success",
        "filename": timestamped_filename,
        "url": public_url,
        "thumbnail_url": thumb_url
    })

@app.route('/')
def index():
    return jsonify({
        "status": "active",
        "message": "Attendance system backend API operating smoothly via Vercel Serverless Function."
    })
