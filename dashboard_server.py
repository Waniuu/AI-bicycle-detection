"""
FastAPI dashboard server for the bicycle & scooter counter v4.6.
  GET /              - Live dashboard (HTML)
  GET /video         - MJPEG stream
  GET /stats         - JSON stats (w tym logi terminala, OSD i config)
  GET /health        - Health check JSON
  GET /database      - Database viewer (HTML)
  GET /api/crossings /api/quarterly - JSON data
  GET /export/csv /export/detail    - CSV downloads
  POST /api/config                  - Zmiana ustawień (RAM / Disk)
  POST /api/config/reset            - Reset z pliku config.yaml
  GET /api/models                   - Lista modeli AI i aktywny model
  POST /api/models/select           - Zmiana aktywnego silnika AI
  GET /api/calibration/files        - Lista plików wideo z folderu
  POST /api/calibration/select      - Wybór aktywnego pliku do pętli
  POST /api/calibration/upload      - Bezpieczny upload i asynchroniczne skalowanie wideo
"""

import csv
import io
import os
import time
import logging
import asyncio
import yaml
from datetime import datetime

import cv2
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from pydantic import BaseModel, Field
from typing import Optional
class ConfigUpdateRequest(BaseModel):
    persist: bool = True
    line_y1: Optional[int] = Field(None, ge=0, le=2160)
    line_y2: Optional[int] = Field(None, ge=0, le=2160)
    line_x1: Optional[int] = Field(None, ge=0, le=3840)
    line_x2: Optional[int] = Field(None, ge=0, le=3840)
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_width: Optional[int] = Field(None, ge=1)
    max_width: Optional[int] = Field(None, ge=1)
    min_height: Optional[int] = Field(None, ge=1)
    max_height: Optional[int] = Field(None, ge=1)
    min_aspect_ratio: Optional[float] = Field(None, ge=0.0)
    min_movement_px: Optional[int] = Field(None, ge=0)
    show_line: Optional[bool] = None
    show_boxes: Optional[bool] = None
    show_tracks: Optional[bool] = None
    show_persons: Optional[bool] = None
    show_hud: Optional[bool] = None
LOG = logging.getLogger("bike_counter")

PROJECT_ROOT = "/home/student/CounterProject"
CONFIG_FILE_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

_BASE = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "calibration_samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

app = FastAPI(title="Mobility Counter", version="4.6")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates_dir = os.path.join(PROJECT_ROOT, "Dashboard", "templates")
if not os.path.exists(templates_dir):
    templates_dir = os.path.join(_BASE, "Dashboard", "templates")
templates = Jinja2Templates(directory=templates_dir)

_state = None
_db = None
_location_id = None
_device_id = None
_config = {}
_start_time = time.time()
_server_instance = None


def set_shared_state(s):
    global _state
    _state = s

def set_db(db, location_id, device_id=None):
    global _db, _location_id, _device_id
    _db = db
    _location_id = location_id
    _device_id = device_id

def set_config(cfg):
    global _config
    _config = cfg

def get_recent_logs(lines_count=15):
    log_path = _config.get("log_file")
    if not log_path:
        log_path = os.path.join(PROJECT_ROOT, "bicycle_counter.log")
    if not os.path.exists(log_path):
        return ["Waiting for log file..."]
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
            return [line.strip() for line in lines[-lines_count:]]
    except Exception as e:
        return [f"Error reading logs: {str(e)}"]

def run():
    global _server_instance
    host = _config.get("dashboard_host", "0.0.0.0")
    port = _config.get("dashboard_port", 8080)
    LOG.info("Dashboard server starting on %s:%d", host, port)
    
    config = uvicorn.Config(app=app, host=host, port=port, log_level="warning", loop="asyncio")
    _server_instance = uvicorn.Server(config)
    _server_instance.run()


def _get_stats():
    if _state is None:
        return {"error": "Pipeline not running"}
    
    with _state.lock:
        cfg = getattr(_state, "config", {})
        mdl_cfg = cfg.get("model", {})
        result = {
            "bicycle_count": getattr(_state, "bicycle_count", 0),
            "scooter_count": getattr(_state, "scooter_count", 0),
            "fps": round(getattr(_state, "fps", 0.0), 1),
            "frame_count": getattr(_state, "frame_count", 0),
            "active_tracks": len(getattr(_state, "active_tracks", [])),
            "camera_ok": getattr(_state, "camera_ok", False),
            "time_synced": datetime.now().year >= 2024,
            "calibration_mode": getattr(_state, "calibration_mode", False),
            "sample_recording": getattr(_state, "sample_recording", False),
            "current_sample_file": getattr(_state, "calib_file_name", "None"),
            "osd": cfg.get("osd", {}),
            "config": cfg,
            "active_model": mdl_cfg.get("active_model", "yolo11s_scooter"),
            "calib_paused": getattr(_state, "calib_paused", False),
            "calib_frame": getattr(_state, "calib_current_frame", 0),
            "calib_total_frames": getattr(_state, "calib_total_frames", 0),
        }
    
    if _db and _location_id:
        db_stats = _db.get_all_stats(_location_id, device_id=_device_id)
        events = _db.get_recent_crossings(_location_id, device_id=_device_id, limit=30)
        
        result["bike_today"] = db_stats.get("bike_today", db_stats.get("today_total", 0))
        result["scooter_today"] = db_stats.get("scooter_today", 0)
        result["bike_all_time"] = db_stats.get("bike_all_time", db_stats.get("all_time_total", 0))
        result["scooter_all_time"] = db_stats.get("scooter_all_time", 0)
        result["today_total"] = db_stats.get("today_total", 0)
        result["all_time_total"] = db_stats.get("all_time_total", 0)
        result["quarterly"] = db_stats.get("quarterly", [])
        result["events"] = events
    else:
        result["bike_today"] = result.get("bicycle_count", 0) 
        result["scooter_today"] = result.get("scooter_count", 0)
        result["bike_all_time"] = result.get("bicycle_count", 0)
        result["scooter_all_time"] = result.get("scooter_count", 0)
        result["today_total"] = result.get("bicycle_count", 0) + result.get("scooter_count", 0)
        result["all_time_total"] = result.get("bicycle_count", 0) + result.get("scooter_count", 0)
        result["quarterly"] = []
        result["events"] = []

    result["logs"] = get_recent_logs(15)
    return result


@app.get("/health")
async def health():
    uptime = round(time.time() - _start_time, 1)
    camera_ok = _state.camera_ok if _state else False
    db_ok = False
    if _db:
        try:
            conn = _db._conn()
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
            db_ok = True
            conn.close()
        except Exception:
            pass
    status = "healthy" if (camera_ok and db_ok) else "degraded"
    return JSONResponse({
        "status": status,
        "version": _config.get("version", "4.6"),
        "uptime_seconds": uptime,
        "camera_ok": camera_ok,
        "database_ok": db_ok,
        "bicycle_count": getattr(_state, "bicycle_count", 0) if _state else 0,
        "scooter_count": getattr(_state, "scooter_count", 0) if _state else 0,
        "device_name": _config.get("device_name", "unknown"),
        "location_name": _config.get("location_name", "unknown"),
    })


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    template_name = "dashboard.html"
    context = {
        "version": _config.get("version", "4.6"),
        "device_name": _config.get("device_name", ""),
        "request": request,
    }
    return templates.TemplateResponse(request, name=template_name, context=context)


@app.get("/video")
async def video_stream():
    def generate():
        while True:
            jpeg = None
            if _state:
                with _state.lock:
                    jpeg = getattr(_state, "latest_jpeg", None)
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stats")
async def stats():
    return JSONResponse(_get_stats())


@app.get("/database", response_class=HTMLResponse)
async def database_view(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="database.html",
        context={"version": _config.get("version", "4.6")}
    )

@app.get("/api/crossings")
async def api_crossings():
    if not _db or not _location_id:
        return JSONResponse([])
    conn = _db._conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT c.id, c.recorded_at, c.track_id, l.name, c.vehicle_type
               FROM crossings c JOIN locations l ON c.location_id=l.id
               WHERE c.location_id=%s AND c.device_id=%s
               ORDER BY c.recorded_at DESC""",
            (_location_id, _device_id),
        )
        rows = []
        for r in cur.fetchall():
            ts = r[1]
            rows.append({
                "id": r[0],
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S") if hasattr(ts, "strftime") else str(ts),
                "track_id": r[2],
                "location": r[3],
                "vehicle_type": r[4] if len(r) > 4 else "bicycle"
            })
        return JSONResponse(rows)
    finally:
        conn.close()


@app.get("/api/quarterly")
async def api_quarterly():
    if not _db or not _location_id:
        return JSONResponse([])
    conn = _db._conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT c15.date, c15.czas, (c15.il_row + c15.il_hul) as total_count
               FROM co_15_minut c15
               JOIN locations l ON c15.location_id = l.name
               WHERE l.id = %s AND l.device_id = %s
               ORDER BY c15.date DESC, c15.czas DESC""",
            (_location_id, _device_id),
        )
        rows = []
        for date_val, time_val, count in cur.fetchall():
            total_seconds = int(time_val.total_seconds())
            h = total_seconds // 3600
            m = (total_seconds % 3600) // 60
            eh, em = h, m + 15
            if em >= 60: 
                em, eh = 0, (h + 1) % 24
            date_str = date_val.strftime("%Y-%m-%d")
            rows.append({"date": date_str, "period": f"{h:02d}:{m:02d}-{eh:02d}:{em:02d}", "count": int(count)})
        return JSONResponse(rows)
    finally:
        conn.close()


@app.get("/export/csv")
async def export_csv():
    if not _db or not _location_id:
        return JSONResponse({"error": "no database"}, status_code=404)
    conn = _db._conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT c15.date, c15.czas, (c15.il_row + c15.il_hul) as total_count
               FROM co_15_minut c15
               JOIN locations l ON c15.lokalizacja = l.name
               WHERE l.id = %s AND l.device_id = %s
               ORDER BY c15.date ASC, c15.czas ASC""",
            (_location_id, _device_id),
        )
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Date", "Time Period", "Total"])
        for date_val, time_val, count in cur.fetchall():
            total_seconds = int(time_val.total_seconds())
            h = total_seconds // 3600
            m = (total_seconds % 3600) // 60
            eh, em = h, m + 15
            if em >= 60: 
                em, eh = 0, (h + 1) % 24
            date_str = date_val.strftime("%Y-%m-%d")
            writer.writerow([date_str, f"{h:02d}:{m:02d}-{eh:02d}:{em:02d}", int(count)])
        return StreamingResponse(
            io.BytesIO(buf.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=bicycle_counts.csv"},
        )
    finally:
        conn.close()


@app.get("/export/detail")
async def export_detail():
    if not _db or not _location_id:
        return JSONResponse({"error": "no database"}, status_code=404)
    conn = _db._conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT recorded_at, track_id, vehicle_type FROM crossings
               WHERE location_id=%s AND device_id=%s
               ORDER BY recorded_at""",
            (_location_id, _device_id),
        )
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Timestamp", "Track ID", "Vehicle Type"])
        for r in cur.fetchall():
            ts = r[0]
            tid = r[1]
            vtype = r[2] if len(r) > 2 else "bicycle"
            ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S") if hasattr(ts, "strftime") else str(ts)
            writer.writerow([ts_str, tid, vtype])
        return StreamingResponse(
            io.BytesIO(buf.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=crossings_detail.csv"},
        )
    finally:
        conn.close()


@app.get("/api/models")
async def get_models():
    if _state and hasattr(_state, "config"):
        cfg = _state.config
    else:
        cfg = _config.copy()

    model_cfg = cfg.get("model", {})
    active = model_cfg.get("active_model", "yolo26s")
    models = model_cfg.get("available_models", {})

    if not models:
        models = {
            "yolo11n_base": {
                "name": "YOLO11n (Tylko rowery)",
                "path": "/home/student/CounterProject/Silnik V2/yolo11n.engine",
                "supports_scooters": False
            },
             "yolo26s": {
                "name": "YOLO26",
                "path": "/home/student/CounterProject/SilnikV5/results/runs/yolo11s_licznik/weights/best.engine",
                "supports_scooters": True
            }
        }

    return JSONResponse({"active": active, "models": models})


@app.post("/api/models/select")
async def select_model(data: dict):
    model_key = data.get("model_key")
    if not model_key:
        return {"status": "error", "message": "No model_key specified"}
    
    LOG.info("[API DASHBOARDU] Odebrano żądanie zmiany na model: '%s'", model_key)
    
    cfg = {}
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = _config.copy()

    if "model" not in cfg:
        cfg["model"] = {}

    available = cfg["model"].get("available_models", {})
    if not available:
        available = {
            "yolo11n_base": {
                "name": "YOLO11n (Tylko rowery)",
                "path": "/home/student/CounterProject/Silnik V2/yolo11n.engine",
                "supports_scooters": False,
            },
             "yolo26s": {
                "name": "YOLO26",
                "path": "/home/student/CounterProject/SilnikV5/results/runs/yolo11s_licznik/weights/best.engine",
                "supports_scooters": True
            }
        }
        cfg["model"]["available_models"] = available

    if model_key not in available:
        LOG.error("[API DASHBOARDU] Błąd: Model '%s' nie istnieje w słowniku available_models!", model_key)
        return {"status": "error", "message": f"Model key '{model_key}' not found in configuration"}

    cfg["model"]["active_model"] = model_key
    cfg["model_path"] = available[model_key]["path"]

    if _state and hasattr(_state, "config"):
        with _state.lock:
            _state.config = cfg
            _state.model_reload_requested = True

    with open(CONFIG_FILE_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return {"status": "ok", "active_model": model_key}


@app.post("/api/config")
async def update_config(data: ConfigUpdateRequest):  # <--- Magia Pydantic
    cfg = {}
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = _config.copy()

    for sec in ["tracker", "model", "filters", "osd"]:
        if sec not in cfg: cfg[sec] = {}

    # Omijamy manualne sprawdzanie i konwersje (int(), float()), Pydantic już to zrobił!
    updates = data.model_dump(exclude_unset=True) # Zwraca tylko przesłane pola
    
    for key in ["show_line", "show_boxes", "show_tracks", "show_persons", "show_hud"]:
        if key in updates: cfg["osd"][key] = updates[key]
            
    for key in ["line_y1", "line_y2", "line_x1", "line_x2"]:
        if key in updates: cfg["tracker"][key] = updates[key]

    if "min_confidence" in updates: cfg["model"]["min_confidence"] = updates["min_confidence"]
    
    for key in ["min_width", "max_width", "min_height", "max_height", "min_aspect_ratio", "min_movement_px"]:
        if key in updates: cfg["filters"][key] = updates[key]

    if _state and hasattr(_state, "config"):
        _state.config = cfg

    if data.persist:
        with open(CONFIG_FILE_PATH, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        LOG.info("Config permanently saved to disk.")

    return {"status": "ok", "persisted": data.persist, "config": cfg}

@app.post("/api/config/reset")
async def reset_config():
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = _config.copy()

    if _state and hasattr(_state, "config"):
        _state.config = cfg

    LOG.info("Config reloaded from config.yaml file.")
    return {"status": "ok", "config": cfg}


@app.get("/api/calibration/files")
async def get_calibration_files():
    files = [
        f for f in os.listdir(SAMPLES_DIR) 
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) and not f.startswith('temp_')
    ]
    files.sort(reverse=True)
    return {"files": files}


@app.post("/api/calibration/select")
async def select_calibration_file(data: dict):
    filename = data.get("filename")
    if not filename:
        return {"status": "error", "message": "No filename specified"}
    
    file_path = os.path.join(SAMPLES_DIR, filename)
    if not os.path.exists(file_path):
        return {"status": "error", "message": "File does not exist"}
    
    if _state is not None:
        with _state.lock:
            _state.calib_file_path = file_path
            _state.calib_file_name = filename
            _state.calib_reload_requested = True
        return {"status": "ok", "active_file": filename}
    return {"status": "error", "message": "Pipeline not running"}


@app.post("/api/calibration/record")
async def record_calibration():
    if _state is not None:
        _state.record_sample_requested = True
        return {"status": "ok", "message": "Record sample requested"}
    return {"status": "error", "message": "Pipeline not running"}


@app.post("/api/calibration/toggle")
async def toggle_calibration(data: dict):
    if _state is not None:
        enabled = data.get("enabled", False)
        _state.calibration_mode = bool(enabled)
        return {"status": "ok", "calibration_mode": _state.calibration_mode}
    return {"status": "error", "message": "Pipeline not running"}


@app.post("/api/calibration/playback")
async def calibration_playback(data: dict):
    if _state is None:
        return {"status": "error", "message": "Pipeline not running"}
    action = data.get("action")
    if action == "play":
        _state.calib_paused = False
    elif action == "pause":
        _state.calib_paused = True
    elif action == "step":
        _state.calib_paused = True
        _state.calib_step = data.get("frames", 1)
    elif action == "seek":
        _state.calib_seek_frame = data.get("frame", 0)
    return {
        "status": "ok", 
        "calib_paused": _state.calib_paused, 
        "calib_frame": _state.calib_current_frame,
        "calib_total_frames": _state.calib_total_frames
    }


def _process_and_scale_video_sync(temp_path, final_path, target_w, target_h):
    cap_in = cv2.VideoCapture(temp_path)
    if not cap_in.isOpened():
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False, "Cannot decode uploaded video file"

    fps = cap_in.get(cv2.CAP_PROP_FPS) or 15.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(final_path, fourcc, fps, (target_w, target_h))

    while True:
        ret, frame = cap_in.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        out.write(resized_frame)

    cap_in.release()
    out.release()

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return True, None


@app.post("/api/calibration/upload")
async def calibration_upload(file: UploadFile = File(...)):
    raw_name = os.path.basename(file.filename)
    clean_name = raw_name.replace(" ", "_")
    if clean_name.startswith("temp_"):
        clean_name = clean_name.replace("temp_", "", 1)
    
    temp_path = os.path.join("/tmp", f"upload_{clean_name}")
    final_path = os.path.join(SAMPLES_DIR, clean_name)
    
    target_w = _config.get("camera_width", 640)
    target_h = _config.get("camera_height", 480)

    try:
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)

        success, err_msg = await asyncio.to_thread(
            _process_and_scale_video_sync, temp_path, final_path, target_w, target_h
        )

        if not success:
            return JSONResponse({"status": "error", "message": err_msg}, status_code=400)

        LOG.info("Calibration sample uploaded & scaled to %dx%d: %s", target_w, target_h, clean_name)
        
        if _state is not None:
            with _state.lock:
                _state.calib_file_path = final_path
                _state.calib_file_name = clean_name
                _state.calib_reload_requested = True

        return {
            "status": "ok", 
            "message": f"File {clean_name} uploaded & scaled to {target_w}x{target_h}", 
            "filename": clean_name
        }
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        LOG.error("Failed to upload and scale video: %s", str(e))
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)