from flask import Flask, request, render_template
from ultralytics import YOLO
import numpy as np
from geopy.geocoders import Nominatim
import os, uuid, cv2, time
import json
from datetime import datetime, timedelta
import sqlite3

app = Flask(__name__)
# Disable static file caching during development so browsers pick up CSS changes immediately
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# Cache-bust static assets in development by adding a version query param
app.config['STATIC_VERSION'] = int(time.time())
@app.context_processor
def inject_static_version():
    return dict(static_version=app.config['STATIC_VERSION'])

# Use absolute paths to ensure it works on Windows and Vercel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Explicitly load the exported ONNX model (not .pt)
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "pothole_guard.onnx")
model = YOLO(ONNX_MODEL_PATH, task="detect")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
VIDEO_FRAME_STRIDE = int(os.environ.get("VIDEO_FRAME_STRIDE", "60"))
MAX_VIDEO_FRAMES = int(os.environ.get("MAX_VIDEO_FRAMES", "10"))
VIDEO_MAX_WIDTH = int(os.environ.get("VIDEO_MAX_WIDTH", "960"))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

geolocator = Nominatim(user_agent="air_pothole_v8_final")

# Database setup
DB_PATH = os.path.join(BASE_DIR, 'pothole_history.db')

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            potholes INTEGER NOT NULL,
            address TEXT,
            lat TEXT,
            lon TEXT,
            date TEXT NOT NULL,
            images TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    
    print(f" Database initialized at: {DB_PATH}")

# Initialize database after function definition is available
init_db()

def cleanup_old_history():
    """Clean up history older than 30 days"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Delete records older than 30 days
        thirty_days_ago = datetime.now() - timedelta(days=30)
        cursor.execute("DELETE FROM history WHERE created_at < ?", (thirty_days_ago,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            print(f" History cleaned: {deleted_count} old items removed")
        
        # Also clean up old result images
        cleanup_old_images()
        
    except Exception as e:
        print(f" History cleanup error: {e}")

def cleanup_old_images():
    """Clean up result images older than 30 days"""
    try:
        thirty_days_ago = time.time() - (30 * 24 * 60 * 60)  # 30 days in seconds
        
        for filename in os.listdir(RESULT_FOLDER):
            file_path = os.path.join(RESULT_FOLDER, filename)
            if os.path.isfile(file_path):
                file_time = os.path.getctime(file_path)
                if file_time < thirty_days_ago:
                    os.remove(file_path)
                    print(f" Deleted old result: {filename}")
                    
    except Exception as e:
        print(f" Image cleanup error: {e}")

def save_to_history(data):
    """Save detection data to SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Insert new record
        cursor.execute('''
            INSERT INTO history (potholes, address, lat, lon, date, images)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            data['potholes'],
            data['address'],
            data['lat'],
            data['lon'],
            data['date'],
            json.dumps(data['images'])
        ))
        
        conn.commit()
        conn.close()
        
        print(f" Saved to history: {data['potholes']} potholes detected")
        
    except Exception as e:
        print(f" History save error: {e}")

def get_history():
    """Get filtered history (last 30 days)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get records from last 30 days
        thirty_days_ago = datetime.now() - timedelta(days=30)
        cursor.execute('''
            SELECT potholes, address, lat, lon, date, images
            FROM history 
            WHERE created_at >= ?
            ORDER BY created_at DESC
        ''', (thirty_days_ago,))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'potholes': row[0],
                'address': row[1],
                'lat': row[2],
                'lon': row[3],
                'date': row[4],
                'images': json.loads(row[5]) if row[5] else []
            })
        
        conn.close()
        return history
        
    except Exception as e:
        print(f" History load error: {e}")
        return []

@app.route("/")
def home(): return render_template("index.html")

@app.route("/history")
def history(): 
    return render_template("history.html")

@app.route("/api/history", methods=["GET"])
def api_get_history():
    """Get history data for frontend"""
    history = get_history()
    return {"history": history}

@app.route("/api/history/cleanup", methods=["POST"])
def api_cleanup_history():
    """Manual cleanup endpoint"""
    cleanup_old_history()
    return {"status": "success", "message": "History cleaned up"}

@app.route("/complaint")
def complaint(): return render_template("complaint.html")

@app.route("/last_result")
def last_result(): return render_template("last_result.html")


@app.route("/api/detect_single", methods=["POST"])
def detect_single():
    file = request.files.get("image")
    if not file: return {"error": "No file"}, 400
    
    unique_id = uuid.uuid4().hex
    ext = file.filename.split('.')[-1].lower()
    path = os.path.normpath(os.path.join(UPLOAD_FOLDER, f"{unique_id}.{ext}"))
    file.save(path)
    
    results = []
    total_potholes = 0
    video_exts = {"mp4", "mov", "avi", "webm", "mkv"}
    
    if ext in video_exts:
        cap = cv2.VideoCapture(path)
        frame_idx = 0
        processed_frames = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % VIDEO_FRAME_STRIDE == 0:
                # Downscale large frames to reduce inference and save time.
                h, w = frame.shape[:2]
                if w > VIDEO_MAX_WIDTH:
                    new_h = int(h * (VIDEO_MAX_WIDTH / w))
                    frame = cv2.resize(frame, (VIDEO_MAX_WIDTH, new_h), interpolation=cv2.INTER_AREA)
                try:
                    res = model(frame, conf=0.25)
                    r0 = res[0]
                    num = len(r0.boxes)
                    total_potholes += num
                    out_name = f"v_{uuid.uuid4().hex}.jpg"
                    cv2.imwrite(
                        os.path.join(RESULT_FOLDER, out_name),
                        r0.plot(),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                    )
                    results.append({"name": out_name, "potholes": num})
                    processed_frames += 1
                    if processed_frames >= MAX_VIDEO_FRAMES:
                        break
                except Exception:
                    continue
        cap.release()
    else:
        try:
            res = model(path, conf=0.25)
            r0 = res[0]
            num = len(r0.boxes)
            total_potholes += num
            out_name = f"r_{unique_id}.jpg"
            cv2.imwrite(
                os.path.join(RESULT_FOLDER, out_name),
                r0.plot(),
                [int(cv2.IMWRITE_JPEG_QUALITY), 85],
            )
            results.append({"name": out_name, "potholes": num})
        except Exception:
            pass
            
    # Save to history
    try:
        detection_data = {
            'potholes': total_potholes,
            'address': request.form.get('address', ''),
            'lat': request.form.get('lat', ''),
            'lon': request.form.get('lon', ''),
            'date': datetime.now().strftime('%Y-%m-%d, %H:%M:%S'),
            'images': [img['name'] for img in results]
        }
        save_to_history(detection_data)
    except Exception as e:
        print(f"Failed to save to history: {e}")
            
    return {"potholes": total_potholes, "images": results}


@app.route("/api/geocode", methods=["POST"])
def api_geocode():
    lat = (request.form.get("lat") or "").strip()
    lon = (request.form.get("lon") or "").strip()
    
    address = "Location not available"
    if lat and lon:
        try:
            loc = geolocator.reverse("%s,%s" % (lat, lon), timeout=5)
            if loc and getattr(loc, "address", None):
                address = loc.address
            else:
                address = "%s, %s" % (lat, lon)
        except Exception:
            address = "%s, %s" % (lat, lon)
    return {"address": address}

@app.route('/prepare_ghmc', methods=['POST'])
def prepare_ghmc():
    """Save complaint payload to ghmc_payload.json (includes absolute image paths)."""
    data = request.get_json() or {}
    imgs = data.get('all_images', [])
    image_paths = [os.path.join(RESULT_FOLDER, img) for img in imgs]
    payload = {
        'potholes': data.get('potholes'),
        'address': data.get('address'),
        'lat': data.get('lat'),
        'lon': data.get('lon'),
        'message': data.get('message'),
        'date': data.get('date'),
        'images': image_paths
    }
    pfile = os.path.join(BASE_DIR, 'ghmc_payload.json')
    import json
    with open(pfile, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {'status': 'ok', 'path': pfile}





def _get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "YOUR_IP"

if __name__ == "__main__":
    import sys
    port = int(os.environ.get("PORT", 8000))
    use_https = os.environ.get("USE_HTTPS", "").lower() in ("1", "true", "yes") or "--https" in sys.argv
    cert_file = os.path.join(BASE_DIR, "cert.pem")
    key_file = os.path.join(BASE_DIR, "key.pem")
    ssl_ctx = None
    if use_https and os.path.isfile(cert_file) and os.path.isfile(key_file):
        import ssl
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(cert_file, key_file)
        port = 8443
    my_ip = _get_local_ip()
    print("\n" + "="*50)
    print("AIR Pothole Detector")
    print("="*50)
    if ssl_ctx:
        print("Local:  https://127.0.0.1:%s" % port)
        print("Mobile: https://%s:%s" % (my_ip, port))
    else:
        print("Local:  http://127.0.0.1:%s" % port)
        print("Mobile: http://%s:%s" % (my_ip, port))
    print("="*50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=port, ssl_context=ssl_ctx, use_reloader=False) 