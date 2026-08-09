"""
FastAPI dashboard server for the bicycle counter v2.0.
  GET /          - Live dashboard (HTML)
  GET /video     - MJPEG stream
  GET /stats     - JSON stats
  GET /health    - Health check JSON
  GET /database  - Database viewer (HTML)
  GET /api/crossings /api/quarterly - JSON data
  GET /export/csv /export/detail    - CSV downloads
"""

import csv
import io
import os
import time
import logging
import yaml

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn

LOG = logging.getLogger("bike_counter")

_BASE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="Bicycle Counter", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory=os.path.join(_BASE, "templates"))

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
            "bicycle_count": _state.bicycle_count,
            "fps": round(_state.fps, 1),
            "frame_count": _state.frame_count,
            "active_tracks": len(_state.active_tracks),
            "total_detections": _state.total_detections,
            "camera_ok": _state.camera_ok,
        }
    if _db and _location_id:
        db_stats = _db.get_all_stats(_location_id, device_id=_device_id)
        events = _db.get_recent_crossings(_location_id, device_id=_device_id, limit=30)
        result["today_total"] = db_stats["today_total"]
        result["all_time_total"] = db_stats["all_time_total"]
        result["quarterly"] = db_stats["quarterly"]
        result["events"] = events
    else:
        result["today_total"] = result["bicycle_count"]
        result["all_time_total"] = result["bicycle_count"]
        result["quarterly"] = []
        result["events"] = []
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
        "version": _config.get("version", "2.0"),
        "uptime_seconds": uptime,
        "camera_ok": camera_ok,
        "database_ok": db_ok,
        "bicycle_count": _state.bicycle_count if _state else 0,
        "device_name": _config.get("device_name", "unknown"),
        "location_name": _config.get("location_name", "unknown"),
    })


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", {
        "version": _config.get("version", "2.0"),
        "device_name": _config.get("device_name", ""),
    })


@app.get("/video")
async def video_stream():
    def generate():
        while True:
            jpeg = None
            if _state:
                with _state.lock:
                    jpeg = _state.latest_jpeg
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stats")
async def stats():
    return JSONResponse(_get_stats())


@app.get("/database", response_class=HTMLResponse)
async def database_view(request: Request):
    return templates.TemplateResponse(request, "database.html", {
        "version": _config.get("version", "2.0"),
    })


@app.get("/api/crossings")
async def api_crossings():
    if not _db or not _location_id:
        return JSONResponse([])
    conn = _db._conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT c.id, c.recorded_at, c.track_id, l.name
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
            """SELECT recorded_at, track_id FROM crossings
               WHERE location_id=%s AND device_id=%s
               ORDER BY recorded_at""",
            (_location_id, _device_id),
        )
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Timestamp", "Track ID"])
        for ts, tid in cur.fetchall():
            ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S") if hasattr(ts, "strftime") else str(ts)
            writer.writerow([ts_str, tid])
        return StreamingResponse(
            io.BytesIO(buf.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=bicycle_crossings_detail.csv"},
        )
    finally:
        conn.close()

@app.post("/api/config")
async def update_config(data: dict):
	if "crossing_line" in data:
		state.config['tracker']['crossing_line_y'] = int(data["crossing_line_y"])

	if "min_confidence" in data:
		state.config['model']['min_confidence'] = float(data["min_confidence"])

	with open("config.yaml", "w") as f:
		yaml.dump(state.config,f)

	return{"status":"ok"}
