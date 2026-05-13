"""
server.py
─────────────────────────────────────────────────────────────────────────────
Flask + Socket.IO backend — RPi EDGE server.

- Serves the local HTML dashboard (dashboard/index.html)
- Pushes ECG waveform + predictions to the browser every 200ms via Socket.IO
- REST API: /api/start  /api/stop  /api/calibrate  /api/status
- Reads MONGO_URI, FLASK_SECRET_KEY, EDGE_DEVICE_ID from .env (via python-dotenv)

Run (RPi):
    source ~/ecg_env/bin/activate
    python server.py

Run (dev):
    python server.py

Then open: http://localhost:5000  (or http://<rpi-hostname>.local:5000)
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import queue
import threading
import time
import logging
import queue
from pathlib import Path

import requests

from dotenv import load_dotenv
load_dotenv()   # loads .env from the directory where server.py lives

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

# ── Path Setup ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR  = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from realtime_inference import ECGInferenceEngine

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("ECGServer")

# ── Flask + Socket.IO Setup ───────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(ROOT_DIR / "dashboard"),
    static_folder=str(ROOT_DIR / "dashboard"),
)

# Read from .env — never hardcode secrets in source code
_flask_secret = os.getenv("FLASK_SECRET_KEY")
if not _flask_secret:
    log.warning("FLASK_SECRET_KEY not set in .env — using insecure default (dev only!)")
    _flask_secret = "ecg_dev_fallback_secret_CHANGE_ME"
app.config["SECRET_KEY"] = _flask_secret

# RPi unique identifier (maps this device to a room/patient in MongoDB)
EDGE_DEVICE_ID = os.getenv("EDGE_DEVICE_ID", "rpi-room-unknown")

# Cloud API settings for RPi edge server
CLOUD_API_URL = os.getenv("CLOUD_API_URL")
EDGE_KEY = os.getenv("EDGE_KEY")
if not CLOUD_API_URL or not EDGE_KEY:
    log.warning("CLOUD_API_URL or EDGE_KEY not set in .env — will not upload to cloud")

# Use threading mode (works on Windows without gevent install issues)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    logger=False, engineio_logger=False)

# ── Global State ──────────────────────────────────────────────────────────
engine        = None
push_thread   = None
cloud_thread  = None
push_running  = False
engine_lock   = threading.Lock()

PUSH_INTERVAL = 0.20   # seconds — push to browser every 200ms
ECG_DISPLAY_POINTS = 500  # last N samples sent to browser (~5s at ~100Hz display)
cloud_queue   = queue.Queue(maxsize=1000)

# ══════════════════════════════════════════════════════════════════════════
# Cloud Upload Loop
# ══════════════════════════════════════════════════════════════════════════

def _post_to_cloud(endpoint: str, data: dict, retries: int = 3):
    """POST data to the Render cloud API with basic retry logic."""
    if not CLOUD_API_URL or not EDGE_KEY:
        return

    url = f"{CLOUD_API_URL.rstrip('/')}/api/ingest/{endpoint}"
    headers = {"X-Edge-Key": EDGE_KEY}

    for attempt in range(retries):
        try:
            resp = requests.post(url, json=data, headers=headers, timeout=5.0)
            resp.raise_for_status()
            log.debug(f"Successfully posted {endpoint} to cloud.")
            return True
        except requests.RequestException as e:
            log.warning(f"Cloud upload failed ({endpoint}) attempt {attempt+1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(5)  # Render cold start might take a moment
    
    log.error(f"Failed to post {endpoint} to cloud after {retries} attempts.")
    return False

def cloud_upload_loop():
    """Background thread to pop from cloud_queue and POST to Render."""
    global push_running
    while push_running or not cloud_queue.empty():
        try:
            task = cloud_queue.get(timeout=1.0)
            endpoint = task.get("endpoint")
            data = task.get("data")
            if endpoint and data:
                 _post_to_cloud(endpoint, data)
        except queue.Empty:
            continue
        except Exception as e:
            log.error(f"Cloud upload loop error: {e}")

# ══════════════════════════════════════════════════════════════════════════
# Background push loop — runs in its own thread
# ══════════════════════════════════════════════════════════════════════════

def push_data_loop():
    """Push ECG state to all connected browsers every 200ms."""
    global push_running
    while push_running:
        try:
            with engine_lock:
                eng = engine

            if eng is not None:
                ecg_buf  = eng.get_ecg_buffer()
                pred     = eng.get_latest_prediction()
                features = eng.get_latest_features()
                status   = eng.get_status()

                # Send last ECG_DISPLAY_POINTS samples (prevents huge payloads)
                ecg_slice = list(ecg_buf)[-ECG_DISPLAY_POINTS:]

                socketio.emit("update", {
                    "ecg"       : ecg_slice,
                    "prediction": pred,
                    "features"  : {k: round(float(v), 2) if isinstance(v, (int, float)) else v
                                   for k, v in features.items()},
                    "status"    : status,
                    "patient_id": eng.patient_id if hasattr(eng, "patient_id") else None
                })
                
                # Check for new cloud upload tasks from engine
                if hasattr(eng, "get_cloud_tasks"):
                    tasks = eng.get_cloud_tasks()
                    for task in tasks:
                        if not cloud_queue.full():
                            cloud_queue.put(task)

        except Exception as e:
            log.warning(f"Push error: {e}")

        time.sleep(PUSH_INTERVAL)


# ══════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


# ── API: Start Engine ─────────────────────────────────────────────────────
@app.route("/api/start", methods=["POST"])
def api_start():
    global engine, push_thread, cloud_thread, push_running

    data      = request.get_json() or {}
    demo_mode = data.get("demo_mode", True)
    port      = data.get("port", "COM3")

    # Stop existing engine first
    with engine_lock:
        eng = engine
    if eng is not None:
        eng.stop()

    try:
        if demo_mode:
            from ecg_simulator import EcgSimulator
            sim = EcgSimulator(record_name="119", inject_anomaly=True, loop=True)
            new_engine = ECGInferenceEngine(demo_mode=True, demo_stream=sim)
            log.info("Starting in DEMO mode.")
        else:
            new_engine = ECGInferenceEngine(port=port, demo_mode=False)
            log.info(f"Starting with hardware on {port}.")
            
        # ── Lookup Patient ID from MongoDB ─────────────────────────
        try:
            from database import collections
            device_doc = collections.devices.find_one({"device_id": EDGE_DEVICE_ID})
            if device_doc and device_doc.get("room_number"):
                room_number = device_doc["room_number"]
                patient_doc = collections.patients.find_one({"assigned_room": room_number})
                if patient_doc:
                    new_engine.patient_id = str(patient_doc["_id"])
                    log.info(f"Assigned patient {new_engine.patient_id} to engine.")
                else:
                    log.warning(f"No patient assigned to room {room_number}.")
            else:
                 log.warning(f"Device {EDGE_DEVICE_ID} not registered or has no room assigned.")
        except Exception as e:
            log.error(f"Failed to lookup patient from MongoDB: {e}")

        new_engine.start()

        with engine_lock:
            engine = new_engine

        # Start push thread if not already running
        if not push_running:
            push_running = True
            push_thread  = threading.Thread(target=push_data_loop,
                                             name="PushThread", daemon=True)
            push_thread.start()
            
            # Start cloud upload thread if configured
            if CLOUD_API_URL and EDGE_KEY:
                cloud_thread = threading.Thread(target=cloud_upload_loop,
                                                name="CloudUploadThread", daemon=True)
                cloud_thread.start()

        return jsonify({"ok": True, "mode": "demo" if demo_mode else "hardware", "port": port})

    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── API: Stop Engine ──────────────────────────────────────────────────────
@app.route("/api/stop", methods=["POST"])
def api_stop():
    global engine, push_running

    push_running = False

    with engine_lock:
        eng    = engine
        engine = None

    if eng is not None:
        eng.stop()
        log.info("Engine stopped.")

    return jsonify({"ok": True})


# ── API: Calibrate ────────────────────────────────────────────────────────
@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    with engine_lock:
        eng = engine
    if eng is None:
        return jsonify({"ok": False, "error": "Engine not running"}), 400
    eng.start_calibration()
    log.info("Calibration started.")
    return jsonify({"ok": True})


# ── API: Status ───────────────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    with engine_lock:
        eng = engine
    running = eng is not None
    status  = eng.get_status() if eng else "STOPPED"
    return jsonify({"running": running, "status": status, "device_id": EDGE_DEVICE_ID})



# ══════════════════════════════════════════════════════════════════════════
# Demo API Endpoints (Phase 5g-ii — for B.Tech presentation demo mode)
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/demo/records", methods=["GET"])
def demo_records():
    """
    GET /api/demo/records
    Returns metadata for all available demo records.
    Used by DemoControlPanel to populate the patient selector dropdown.
    """
    from ecg_simulator import DEMO_RECORD_INFO
    return jsonify(list(DEMO_RECORD_INFO.values())), 200


@app.route("/api/demo/start", methods=["POST"])
def demo_start():
    """
    POST /api/demo/start
    Body: {"record": "200", "mode": "mitbih"} or {"mode": "synthetic"}

    Stops any running engine and starts a fresh one in demo mode
    using the specified MIT-BIH record (or synthetic fallback).
    """
    global engine, push_thread, cloud_thread, push_running

    data        = request.get_json() or {}
    record_name = data.get("record", "119")
    mode        = data.get("mode", "mitbih")

    from ecg_simulator import EcgSimulator, DEMO_RECORD_INFO

    # Stop existing engine
    with engine_lock:
        old_eng = engine
    if old_eng is not None:
        old_eng.stop()
        with engine_lock:
            engine = None

    try:
        sim        = EcgSimulator(record_name=record_name, inject_anomaly=True, loop=True)
        new_engine = ECGInferenceEngine(demo_mode=True, demo_stream=sim)
        log.info(f"Demo start: record={record_name}, mode={mode}")

        # Re-resolve patient_id from MongoDB
        try:
            from database import collections
            device_doc = collections.devices.find_one({"device_id": EDGE_DEVICE_ID})
            if device_doc and device_doc.get("room_number"):
                patient_doc = collections.patients.find_one(
                    {"assigned_room": device_doc["room_number"]}
                )
                if patient_doc:
                    new_engine.patient_id = str(patient_doc["_id"])
                    log.info(f"Demo: assigned patient {new_engine.patient_id}")
        except Exception as e:
            log.warning(f"Demo: could not resolve patient_id: {e}")

        new_engine.start()

        with engine_lock:
            engine = new_engine

        if not push_running:
            push_running = True
            push_thread  = threading.Thread(target=push_data_loop,
                                             name="PushThread", daemon=True)
            push_thread.start()
            if CLOUD_API_URL and EDGE_KEY:
                cloud_thread = threading.Thread(target=cloud_upload_loop,
                                                name="CloudUploadThread", daemon=True)
                cloud_thread.start()

        info = DEMO_RECORD_INFO.get(record_name, {
            "record"     : record_name,
            "name"       : f"Record {record_name}",
            "description": "MIT-BIH record",
            "arrhythmia" : "Unknown",
            "expected_model_response": "Unknown",
        })

        return jsonify({
            "ok"         : True,
            "record"     : record_name,
            "description": info.get("description"),
            "name"       : info.get("name"),
            "arrhythmia" : info.get("arrhythmia"),
            "expected"   : info.get("expected_model_response"),
        }), 200

    except Exception as e:
        log.error(f"Demo start failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/demo/switch-to-arrhythmia", methods=["POST"])
def demo_switch_arrhythmia():
    """
    POST /api/demo/switch-to-arrhythmia
    Mid-demo: instantly swaps the running simulator to record 200
    (Ventricular bigeminy) for the live demonstration climax.

    Call flow during presentation:
      1. Start with record 100 (Normal) → professors see clean ECG
      2. Click this button → bigeminy arrhythmia starts immediately
      3. Wait ~15s → ABNORMAL alert fires → buzzer sounds
    """
    global engine, push_thread, cloud_thread, push_running
    from ecg_simulator import EcgSimulator, DEMO_RECORD_INFO

    with engine_lock:
        old_eng = engine
    if old_eng is not None:
        old_eng.stop()
        with engine_lock:
            engine = None

    try:
        record_name = "200"
        sim         = EcgSimulator(record_name=record_name, inject_anomaly=True, loop=True)
        new_engine  = ECGInferenceEngine(demo_mode=True, demo_stream=sim)
        log.info("Demo: switched to ARRHYTHMIA record 200 (Ventricular bigeminy)")

        try:
            from database import collections
            device_doc = collections.devices.find_one({"device_id": EDGE_DEVICE_ID})
            if device_doc and device_doc.get("room_number"):
                patient_doc = collections.patients.find_one(
                    {"assigned_room": device_doc["room_number"]}
                )
                if patient_doc:
                    new_engine.patient_id = str(patient_doc["_id"])
        except Exception as e:
            log.warning(f"Demo switch: could not resolve patient_id: {e}")

        new_engine.start()
        with engine_lock:
            engine = new_engine

        if not push_running:
            push_running = True
            push_thread  = threading.Thread(target=push_data_loop,
                                             name="PushThread", daemon=True)
            push_thread.start()
            if CLOUD_API_URL and EDGE_KEY:
                cloud_thread = threading.Thread(target=cloud_upload_loop,
                                                name="CloudUploadThread", daemon=True)
                cloud_thread.start()

        info = DEMO_RECORD_INFO["200"]
        return jsonify({
            "ok"         : True,
            "record"     : "200",
            "description": info["description"],
            "name"       : info["name"],
            "arrhythmia" : info["arrhythmia"],
            "expected"   : info["expected_model_response"],
            "message"    : "Switched to arrhythmia mode — ABNORMAL alert expected within ~15s",
        }), 200

    except Exception as e:
        log.error(f"Demo switch failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/demo/ground-truth", methods=["GET"])
def demo_ground_truth():
    """
    GET /api/demo/ground-truth
    Returns the current MIT-BIH expert annotation for the running
    demo stream at the current playback position.

    Lets professors see: "MIT-BIH expert said V (PVC) here,
    our model says ABNORMAL here — they match!"
    """
    with engine_lock:
        eng = engine

    if eng is None:
        return jsonify({"error": "No engine running"}), 400

    if not eng.demo_mode or eng.demo_stream is None:
        return jsonify({"error": "Engine not in demo mode"}), 400

    # Get current display buffer length as a proxy for playback position
    buf_len = len(eng.get_ecg_buffer())
    annotation = eng.demo_stream.get_current_annotation(buf_len)

    return jsonify({
        "record"      : eng.demo_stream.record_name,
        "beat_label"  : annotation["beat_label"],
        "description" : annotation["description"],
        "sample"      : annotation["sample"],
        "model_latest": eng.get_latest_prediction(),
    }), 200


# ══════════════════════════════════════════════════════════════════════════
# Socket.IO Events
# ══════════════════════════════════════════════════════════════════════════

@socketio.on("connect")
def on_connect():
    log.info(f"Browser connected.")
    emit("connected", {"msg": "ECG Server connected"})


@socketio.on("disconnect")
def on_disconnect():
    log.info("Browser disconnected.")


# ══════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  ECG Edge Server (RPi)")
    print(f"  Device ID : {EDGE_DEVICE_ID}")
    print("  Open      : http://localhost:5000")
    print("=" * 55)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
