"""
cloud_api.py
─────────────────────────────────────────────────────────────────────────────
ECG Project — Render Cloud REST API

This is a THIN, STATELESS Flask app with NO ML, NO serial port, NO inference.
It only reads/writes MongoDB Atlas and serves JWT-protected endpoints to the
React frontend and the RPi edge server.

Deployed on: Render (free tier)
Start command: gunicorn cloud_api:app
Procfile: web: gunicorn cloud_api:app --bind 0.0.0.0:$PORT

Endpoints:
  POST  /api/auth/login              — authenticate user, return JWT
  POST  /api/ingest/summary          — (RPi) save 5-sec ECG summary
  POST  /api/ingest/alert            — (RPi) save alert
  GET   /api/doctor/patients         — (doctor/nurse JWT) assigned patients
  GET   /api/patients/<id>/ecg-history — (doctor JWT) paginated summaries
  GET   /api/alerts                  — (doctor/nurse JWT) unacknowledged alerts
  POST  /api/alerts/<id>/acknowledge — (doctor/nurse JWT) mark alert seen
  GET   /api/patients/me             — (patient JWT) own records
  POST  /api/admin/users             — (admin JWT) create user
  POST  /api/admin/assign-device     — (admin JWT) map device_id → room
  POST  /api/admin/assign-patient    — (admin JWT) map patient → room
  POST  /api/admin/assign-doctor     — (admin JWT) link doctor to patient
  GET   /api/status                  — (public) health check (keep Render awake)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import logging
from datetime import datetime, timezone, timedelta
from functools import wraps

import bcrypt
import jwt
from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo.errors import DuplicateKeyError

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("CloudAPI")

# ── Flask App ─────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Vercel → Render

# ── Secrets from env ──────────────────────────────────────────────────────
JWT_SECRET  = os.getenv("JWT_SECRET", "")
EDGE_KEY    = os.getenv("EDGE_KEY", "")
FLASK_SECRET = os.getenv("FLASK_SECRET_KEY", "ecg_cloud_fallback")
app.config["SECRET_KEY"] = FLASK_SECRET

if not JWT_SECRET:
    log.warning("JWT_SECRET not set — tokens will use insecure fallback")
if not EDGE_KEY:
    log.warning("EDGE_KEY not set — RPi ingest endpoints are unprotected")

JWT_ALGO        = "HS256"
JWT_EXPIRES_H   = 12   # token lifetime in hours

# ── Database (lazy import — avoids cold-start failures if Mongo is slow) ──
def get_col(name: str):
    """Return a MongoDB collection by name (lazy singleton)."""
    from database import get_db
    return get_db()[name]


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _oid(id_str: str) -> ObjectId:
    """Convert string → ObjectId, raise ValueError on bad input."""
    try:
        return ObjectId(id_str)
    except (InvalidId, TypeError):
        raise ValueError(f"Invalid ObjectId: {id_str!r}")


def _serialize(doc: dict) -> dict:
    """Convert MongoDB doc fields to JSON-serialisable types."""
    if doc is None:
        return {}
    result = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            result[k] = str(v)
        elif isinstance(v, datetime):
            result[k] = v.isoformat()
        elif isinstance(v, list):
            result[k] = [
                str(i) if isinstance(i, ObjectId) else
                i.isoformat() if isinstance(i, datetime) else i
                for i in v
            ]
        else:
            result[k] = v
    return result


def _make_token(user_id: str, role: str) -> str:
    """Create a signed JWT for the given user."""
    payload = {
        "sub": user_id,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRES_H),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET or "insecure", algorithm=JWT_ALGO)


def _decode_token(token: str) -> dict:
    """Decode + verify a JWT. Raises jwt.InvalidTokenError on failure."""
    return jwt.decode(token, JWT_SECRET or "insecure", algorithms=[JWT_ALGO])


# ── Auth decorators ───────────────────────────────────────────────────────

def require_jwt(*allowed_roles):
    """
    Decorator: require a valid JWT with one of the allowed roles.
    Injects `g.user_id` and `g.role` into the request context.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            from flask import g
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                return jsonify({"error": "Missing or invalid Authorization header"}), 401
            token = auth.split(" ", 1)[1]
            try:
                payload = _decode_token(token)
            except jwt.ExpiredSignatureError:
                return jsonify({"error": "Token expired"}), 401
            except jwt.InvalidTokenError as e:
                return jsonify({"error": f"Invalid token: {e}"}), 401

            role = payload.get("role", "")
            if allowed_roles and role not in allowed_roles:
                return jsonify({"error": f"Forbidden — requires role: {allowed_roles}"}), 403

            g.user_id = payload["sub"]
            g.role    = role
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def require_edge_key(fn):
    """Decorator: require valid X-Edge-Key header (RPi → Render ingest)."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        key = request.headers.get("X-Edge-Key", "")
        if not EDGE_KEY or key != EDGE_KEY:
            return jsonify({"error": "Forbidden — invalid X-Edge-Key"}), 403
        return fn(*args, **kwargs)
    return wrapper


# ══════════════════════════════════════════════════════════════════════════
# Auth
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    """
    POST /api/auth/login
    Body: {"email": "...", "password": "..."}
    Returns: {"token": "<JWT>", "role": "...", "user_id": "..."}
    """
    data = request.get_json() or {}
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400

    user = get_col("users").find_one({"email": email})
    if not user:
        return jsonify({"error": "Invalid email or password"}), 401

    stored_hash = user.get("password_hash", "")
    if not bcrypt.checkpw(password.encode(), stored_hash.encode()):
        return jsonify({"error": "Invalid email or password"}), 401

    user_id = str(user["_id"])
    role    = user["role"]
    token   = _make_token(user_id, role)

    log.info(f"Login: {email} (role={role})")
    return jsonify({
        "token":    token,
        "role":     role,
        "user_id":  user_id,
        "username": user.get("username", ""),
    }), 200


# ══════════════════════════════════════════════════════════════════════════
# RPi Ingest Endpoints (called by edge server, not React)
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/ingest/summary", methods=["POST"])
@require_edge_key
def ingest_summary():
    """
    POST /api/ingest/summary
    Header: X-Edge-Key: <shared secret>
    Body: ECG summary dict from RPi (matches ecg_summaries schema)
    """
    data = request.get_json() or {}

    # Required fields
    patient_id_str = data.get("patient_id")
    device_id      = data.get("device_id")
    start_time_str = data.get("start_time")
    end_time_str   = data.get("end_time")
    prediction     = data.get("prediction")

    if not all([patient_id_str, device_id, start_time_str, end_time_str, prediction]):
        return jsonify({"error": "Missing required fields: patient_id, device_id, start_time, end_time, prediction"}), 400

    try:
        patient_oid = _oid(patient_id_str)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    doc = {
        "patient_id":        patient_oid,
        "device_id":         device_id,
        "start_time":        datetime.fromisoformat(start_time_str),
        "end_time":          datetime.fromisoformat(end_time_str),
        "heart_rate":        data.get("heart_rate"),
        "rr_mean":           data.get("rr_mean"),
        "rr_std":            data.get("rr_std"),
        "sdnn":              data.get("sdnn"),
        "rmssd":             data.get("rmssd"),
        "beat_variance":     data.get("beat_variance"),
        "r_peak_count":      data.get("r_peak_count"),
        "sqi":               data.get("sqi"),
        "prediction":        prediction,
        "probability":       data.get("probability"),
        "consecutive_count": data.get("consecutive_count", 0),
    }

    result = get_col("ecg_summaries").insert_one(doc)
    return jsonify({"ok": True, "inserted_id": str(result.inserted_id)}), 201


@app.route("/api/ingest/alert", methods=["POST"])
@require_edge_key
def ingest_alert():
    """
    POST /api/ingest/alert
    Header: X-Edge-Key: <shared secret>
    Body: alert dict from RPi

    Debounce: rejects if an unacknowledged alert exists for the same
    patient_id within the last 5 minutes.
    """
    data = request.get_json() or {}

    patient_id_str = data.get("patient_id")
    device_id      = data.get("device_id")

    if not patient_id_str or not device_id:
        return jsonify({"error": "patient_id and device_id are required"}), 400

    try:
        patient_oid = _oid(patient_id_str)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # ── 5-minute debounce check ──────────────────────────────────────────
    five_min_ago = datetime.now(timezone.utc) - timedelta(minutes=5)
    existing = get_col("alerts").find_one({
        "patient_id":   patient_oid,
        "acknowledged": False,
        "timestamp":    {"$gte": five_min_ago},
    })
    if existing:
        log.info(f"Alert debounced for patient {patient_id_str} — recent alert exists")
        return jsonify({"ok": True, "debounced": True, "existing_id": str(existing["_id"])}), 200

    doc = {
        "patient_id":        patient_oid,
        "device_id":         device_id,
        "severity":          data.get("severity", "HIGH"),
        "timestamp":         datetime.now(timezone.utc),
        "consecutive_count": data.get("consecutive_count", 3),
        "probability":       data.get("probability"),
        "acknowledged":      False,
        "acknowledged_by":   None,
    }

    result = get_col("alerts").insert_one(doc)
    log.warning(f"ALERT created for patient {patient_id_str} — id={result.inserted_id}")
    return jsonify({"ok": True, "inserted_id": str(result.inserted_id)}), 201


# ══════════════════════════════════════════════════════════════════════════
# Doctor / Nurse Endpoints
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/doctor/patients", methods=["GET"])
@require_jwt("doctor", "nurse")
def doctor_patients():
    """
    GET /api/doctor/patients
    Returns list of patients assigned to the logged-in doctor/nurse.
    """
    from flask import g
    doctor_oid = _oid(g.user_id)

    patients = list(get_col("patients").find({
        "$or": [
            {"assigned_doctors": doctor_oid},
            {"assigned_nurses":  doctor_oid},
        ]
    }))

    return jsonify({"patients": [_serialize(p) for p in patients]}), 200


@app.route("/api/patients/<patient_id>/ecg-history", methods=["GET"])
@require_jwt("doctor", "nurse", "admin")
def ecg_history(patient_id: str):
    """
    GET /api/patients/<id>/ecg-history?page=1&limit=50
    Returns paginated ECG summaries (newest first).
    """
    try:
        patient_oid = _oid(patient_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    page  = max(1, int(request.args.get("page", 1)))
    limit = min(200, max(1, int(request.args.get("limit", 50))))
    skip  = (page - 1) * limit

    cursor = (
        get_col("ecg_summaries")
        .find({"patient_id": patient_oid})
        .sort("start_time", -1)
        .skip(skip)
        .limit(limit)
    )

    docs  = [_serialize(d) for d in cursor]
    total = get_col("ecg_summaries").count_documents({"patient_id": patient_oid})

    return jsonify({
        "summaries": docs,
        "total":     total,
        "page":      page,
        "limit":     limit,
        "pages":     (total + limit - 1) // limit,
    }), 200


@app.route("/api/alerts", methods=["GET"])
@require_jwt("doctor", "nurse", "admin")
def get_alerts():
    """
    GET /api/alerts?patient_id=<id>&acknowledged=false
    Returns unacknowledged alerts (default) for a patient.
    """
    patient_id_str = request.args.get("patient_id")
    acked_param    = request.args.get("acknowledged", "false").lower()

    query: dict = {}
    if patient_id_str:
        try:
            query["patient_id"] = _oid(patient_id_str)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    if acked_param == "false":
        query["acknowledged"] = False

    alerts = list(
        get_col("alerts")
        .find(query)
        .sort("timestamp", -1)
        .limit(100)
    )
    return jsonify({"alerts": [_serialize(a) for a in alerts]}), 200


@app.route("/api/alerts/<alert_id>/acknowledge", methods=["POST"])
@require_jwt("doctor", "nurse", "admin")
def acknowledge_alert(alert_id: str):
    """
    POST /api/alerts/<id>/acknowledge
    Marks an alert as acknowledged by the requesting user.
    """
    from flask import g
    try:
        alert_oid = _oid(alert_id)
        user_oid  = _oid(g.user_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    result = get_col("alerts").update_one(
        {"_id": alert_oid},
        {"$set": {"acknowledged": True, "acknowledged_by": user_oid}},
    )

    if result.matched_count == 0:
        return jsonify({"error": "Alert not found"}), 404

    return jsonify({"ok": True}), 200


# ══════════════════════════════════════════════════════════════════════════
# Patient Endpoints
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/patients/me", methods=["GET"])
@require_jwt("patient")
def patient_me():
    """
    GET /api/patients/me
    Returns the patient's own record + recent ECG summaries.
    """
    from flask import g
    user_oid = _oid(g.user_id)

    patient = get_col("patients").find_one({"user_id": user_oid})
    if not patient:
        return jsonify({"error": "Patient record not found"}), 404

    # Last 24 hours of summaries
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    summaries = list(
        get_col("ecg_summaries")
        .find({"patient_id": patient["_id"], "start_time": {"$gte": since}})
        .sort("start_time", -1)
        .limit(500)
    )

    return jsonify({
        "patient":   _serialize(patient),
        "summaries": [_serialize(s) for s in summaries],
    }), 200


# ══════════════════════════════════════════════════════════════════════════
# Admin Endpoints
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/admin/users", methods=["POST"])
@require_jwt("admin")
def admin_create_user():
    """
    POST /api/admin/users
    Body: {"username": "...", "email": "...", "password": "...", "role": "..."}
    Creates a new user with a bcrypt-hashed password.
    """
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")
    role     = data.get("role", "patient")

    if not all([username, email, password]):
        return jsonify({"error": "username, email, and password are required"}), 400

    if role not in ("admin", "doctor", "nurse", "patient"):
        return jsonify({"error": "role must be: admin, doctor, nurse, patient"}), 400

    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    doc = {
        "username":      username,
        "email":         email,
        "password_hash": pw_hash,
        "role":          role,
        "created_at":    datetime.now(timezone.utc),
    }

    try:
        result = get_col("users").insert_one(doc)
    except DuplicateKeyError:
        return jsonify({"error": f"Email already exists: {email}"}), 409

    user_oid = result.inserted_id

    # Auto-create patient record so assign-patient works immediately
    if role == "patient":
        get_col("patients").insert_one({
            "user_id":          user_oid,
            "name":             username,
            "dob":              None,
            "assigned_room":    None,
            "assigned_doctors": [],
            "assigned_nurses":  [],
            "created_at":       datetime.now(timezone.utc),
        })

    log.info(f"Admin created user: {email} (role={role})")
    return jsonify({"ok": True, "user_id": str(user_oid)}), 201


@app.route("/api/admin/users", methods=["GET"])
@require_jwt("admin")
def admin_list_users():
    """GET /api/admin/users — list all users (admin only)."""
    users = list(get_col("users").find({}, {"password_hash": 0}))
    return jsonify({"users": [_serialize(u) for u in users]}), 200


@app.route("/api/admin/devices", methods=["GET"])
@require_jwt("admin")
def admin_list_devices():
    """GET /api/admin/devices — list all registered RPi devices."""
    devices = list(get_col("devices").find({}))
    return jsonify({"devices": [_serialize(d) for d in devices]}), 200


@app.route("/api/admin/patients", methods=["GET"])
@require_jwt("admin")
def admin_list_patients():
    """GET /api/admin/patients — list all patients."""
    patients = list(get_col("patients").find({}))
    return jsonify({"patients": [_serialize(p) for p in patients]}), 200


@app.route("/api/admin/assign-device", methods=["POST"])
@require_jwt("admin")
def admin_assign_device():
    """
    POST /api/admin/assign-device
    Body: {"device_id": "rpi-room-101", "room_number": "101"}
    Registers or updates an RPi device → room mapping.
    """
    data        = request.get_json() or {}
    device_id   = data.get("device_id", "").strip()
    room_number = data.get("room_number", "").strip()

    if not device_id or not room_number:
        return jsonify({"error": "device_id and room_number are required"}), 400

    get_col("devices").update_one(
        {"device_id": device_id},
        {"$set": {
            "device_id":     device_id,
            "room_number":   room_number,
            "status":        "active",
            "registered_at": datetime.now(timezone.utc),
        }},
        upsert=True,
    )
    return jsonify({"ok": True}), 200


@app.route("/api/admin/assign-patient", methods=["POST"])
@require_jwt("admin")
def admin_assign_patient():
    """
    POST /api/admin/assign-patient
    Body: {"patient_id": "<ObjectId>", "room_number": "101"}
    """
    data           = request.get_json() or {}
    patient_id_str = data.get("patient_id", "")
    room_number    = data.get("room_number", "").strip()

    if not patient_id_str or not room_number:
        return jsonify({"error": "patient_id and room_number are required"}), 400

    try:
        patient_oid = _oid(patient_id_str)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    result = get_col("patients").update_one(
        {"_id": patient_oid},
        {"$set": {"assigned_room": room_number}},
    )
    if result.matched_count == 0:
        return jsonify({"error": "Patient not found"}), 404

    return jsonify({"ok": True}), 200


@app.route("/api/admin/release-patient", methods=["POST"])
@require_jwt("admin")
def admin_release_patient():
    """
    POST /api/admin/release-patient
    Body: {"patient_id": "<ObjectId>"}
    Clears the patient's assigned_room so the room can be given to another patient.
    """
    data           = request.get_json() or {}
    patient_id_str = data.get("patient_id", "")
    try:
        patient_oid = _oid(patient_id_str)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    result = get_col("patients").update_one(
        {"_id": patient_oid},
        {"$set": {"assigned_room": None}},
    )
    if result.matched_count == 0:
        return jsonify({"error": "Patient not found"}), 404

    return jsonify({"ok": True}), 200


@app.route("/api/admin/assign-doctor", methods=["POST"])
@require_jwt("admin")
def admin_assign_doctor():
    """
    POST /api/admin/assign-doctor
    Body: {"patient_id": "<ObjectId>", "doctor_id": "<ObjectId>", "role": "doctor"|"nurse"}
    Links a doctor or nurse to a patient.
    """
    data           = request.get_json() or {}
    patient_id_str = data.get("patient_id", "")
    doctor_id_str  = data.get("doctor_id", "")
    staff_role     = data.get("role", "doctor")   # "doctor" or "nurse"

    if not patient_id_str or not doctor_id_str:
        return jsonify({"error": "patient_id and doctor_id are required"}), 400

    try:
        patient_oid = _oid(patient_id_str)
        doctor_oid  = _oid(doctor_id_str)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Look up the actual role from the users collection instead of trusting the
    # frontend payload — ensures nurses go into assigned_nurses, not assigned_doctors.
    staff_user = get_col("users").find_one({"_id": doctor_oid}, {"role": 1})
    if staff_user:
        staff_role = staff_user.get("role", "doctor")

    field = "assigned_nurses" if staff_role == "nurse" else "assigned_doctors"

    result = get_col("patients").update_one(
        {"_id": patient_oid},
        {"$addToSet": {field: doctor_oid}},
    )
    if result.matched_count == 0:
        return jsonify({"error": "Patient not found"}), 404

    return jsonify({"ok": True, "field": field}), 200


# ══════════════════════════════════════════════════════════════════════════
# Admin Utility
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/admin/fix-patients", methods=["POST"])
@require_jwt("admin")
def admin_fix_patients():
    """
    POST /api/admin/fix-patients
    One-time migration: create missing patients documents for any user with
    role=patient that has no corresponding patients record.
    Safe to call multiple times.
    """
    patient_users = list(get_col("users").find({"role": "patient"}))
    created = []
    for u in patient_users:
        existing = get_col("patients").find_one({"user_id": u["_id"]})
        if not existing:
            get_col("patients").insert_one({
                "user_id":          u["_id"],
                "name":             u.get("username", ""),
                "dob":              None,
                "assigned_room":    None,
                "assigned_doctors": [],
                "assigned_nurses":  [],
                "created_at":       datetime.now(timezone.utc),
            })
            created.append(u.get("email"))
    log.info(f"fix-patients: created {len(created)} missing records: {created}")
    return jsonify({"ok": True, "created": created, "total_fixed": len(created)}), 200


# ══════════════════════════════════════════════════════════════════════════
# Health Check
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/status", methods=["GET"])
def api_status():
    """
    GET /api/status
    Public health check — pinged by cron-job.org every 10 min to keep
    Render free tier awake.
    """
    return jsonify({
        "ok":      True,
        "service": "ECG Cloud API",
        "version": "2.0",
        "time":    datetime.now(timezone.utc).isoformat(),
    }), 200


# ══════════════════════════════════════════════════════════════════════════
# Error Handlers
# ══════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(e):
    log.error(f"Internal server error: {e}")
    return jsonify({"error": "Internal server error"}), 500


# ══════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    print("=" * 55)
    print("  ECG Cloud API (Render)")
    print(f"  Port: {port}")
    print("=" * 55)
    app.run(host="0.0.0.0", port=port, debug=False)
