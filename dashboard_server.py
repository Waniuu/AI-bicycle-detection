"""
FastAPI dashboard server for the bicycle & scooter counter v3.0.
  GET /          - Live dashboard (HTML)
  GET /video     - MJPEG stream
  GET /stats     - JSON stats (w tym logi terminala)
  GET /health    - Health check JSON
  GET /database  - Database viewer (HTML)
  GET /api/crossings /api/quarterly - JSON data
  GET /export/csv /export/detail    - CSV downloads
  POST /api/config                  - Zmiana ustawień na żywo
"""

import csv
import io
import os
import time
import logging
import yaml
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn

LOG = logging.getLogger("bike_counter")

# _BASE jest teraz głównym katalogiem projektu
_BASE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Mobility Counter", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Szablony znajdują się w podkatalogu
templates = Jinja2Templates(directory=os.path.join(_BASE, "Dashboard/templates"))

_state = None
_db = None
_location_id = None
_device_id = None
_config = {}
_start_time = time.time()


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
    """Odczytuje ostatnie linie z pliku logów aplikacji na podstawie config.yaml"""
    # Główna aplikacja konfiguruje plik logów, możemy pobrać ścieżkę stamtąd
    log_path = _config.get("log_file")
        
    # Domyślna ścieżka, jeśli brak wpisu w configu
    if not log_path:
        log_path = os.path.join(_BASE, "bicycle_counter.log")

    if not os.path.exists(log_path):
        return ["Waiting for log file..."]
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
            return [line.strip() for line in lines[-lines_count:]]
    except Exception as e:
        return [f"Error reading logs: {str(e)}"]


def run():
    host = _config.get("dashboard_host", "0.0.0.0")
    port = _config.get("dashboard_port", 8080)
    LOG.info("Dashboard server starting on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")


def _get_stats():
    if _state is None:
        return {"error": "Pipeline not running"}
    
    with _state.lock:
        result = {
            "bicycle_count": getattr(_state, "bicycle_count", 0),
            "scooter_count": getattr(_state, "scooter_count", 0),
            "fps": round(getattr(_state, "fps", 0.0), 1),
            "frame_count": getattr(_state, "frame_count", 0),
            "active_tracks": len(getattr(_state, "active_tracks", [])),
            "total_detections": getattr(_state, "total_detections", 0),
            "camera_ok": getattr(_state, "camera_ok", False),
            "time_synced": datetime.now().year >= 2024, # Dodajemy flagę statusu czasu
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
        result["bike_today"] = result.get("bicycle_count", 0) # Zabezpieczenie na wypadek braku klucza
        result["scooter_today"] = result["scooter_count"]
        result["bike_all_time"] = result["bicycle_count"]
        result["scooter_all_time"] = result["scooter_count"]
        result["today_total"] = result["bicycle_count"] + result["scooter_count"]
        result["all_time_total"] = result["bicycle_count"] + result["scooter_count"]
        result["quarterly"] = []
        result["events"] = []

    # Przekazanie logów do konsoli w dashboardzie
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
        "version": _config.get("version", "3.0"),
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
    # Używamy statycznie bogatszego szablonu deweloperskiego
    template_name = "dev/dashboard_dev.html"

    context = {
        "version": _config.get("version", "3.0"),
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
    return templates.TemplateResponse("database.html", {
        "request": request,
        "version": _config.get("version", "3.0"),
    })


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
            """SELECT bucket_date, bucket_hour, bucket_quarter, total_count
               FROM quarterly_counts
               WHERE location_id=%s AND device_id=%s
               ORDER BY bucket_date DESC, bucket_hour DESC, bucket_quarter DESC""",
            (_location_id, _device_id),
        )
        rows = []
        for date_val, hour, quarter, count in cur.fetchall():
            h, q, c = int(hour), int(quarter), int(count)
            ms = q * 15
            eh, em = h, ms + 15
            if em >= 60:
                em, eh = 0, (h + 1) % 24
            date_str = date_val.strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val)
            rows.append({"date": date_str, "period": f"{h:02d}:{ms:02d}-{eh:02d}:{em:02d}", "count": c})
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
            """SELECT bucket_date, bucket_hour, bucket_quarter, total_count
               FROM quarterly_counts WHERE location_id=%s AND device_id=%s
               ORDER BY bucket_date, bucket_hour, bucket_quarter""",
            (_location_id, _device_id),
        )
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Date", "Time Period", "Total"])
        for date_val, hour, quarter, count in cur.fetchall():
            h, q, c = int(hour), int(quarter), int(count)
            ms = q * 15
            eh, em = h, ms + 15
            if em >= 60:
                em, eh = 0, (h + 1) % 24
            date_str = date_val.strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val)
            writer.writerow([date_str, f"{h:02d}:{ms:02d}-{eh:02d}:{em:02d}", c])
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


@app.post("/api/config")
async def update_config(data: dict):
    # Ścieżka do config.yaml jest teraz względna do głównego katalogu projektu
    config_path = os.path.join(_BASE, "config.yaml")
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = _config

    if "tracker" not in cfg:
        cfg["tracker"] = {}
    if "model" not in cfg:
        cfg["model"] = {}

    if "crossing_line_y" in data:
        cfg["tracker"]["crossing_line_y"] = int(data["crossing_line_y"])
        if _state and hasattr(_state, "config"):
            _state.config["tracker"]["crossing_line_y"] = int(data["crossing_line_y"])

    if "min_confidence" in data:
        cfg["model"]["min_confidence"] = float(data["min_confidence"])
        if _state and hasattr(_state, "config"):
            _state.config["model"]["min_confidence"] = float(data["min_confidence"])

    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    return {"status": "ok"}