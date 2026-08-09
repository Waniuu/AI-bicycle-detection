# Bicycle Counter v2.0 — Full Project Specification

> **Version:** 2.0  
> **Target Hardware:** NVIDIA Jetson Orin Nano (8 GB)  
> **Firmware:** JetPack 6.x / L4T 39.2.0  
> **CUDA:** 13.2 | **TensorRT:** bundled with JetPack | **Python:** 3.12  
> **Camera:** Suyin HD USB Camera (`/dev/video0`, 640x480 @ 15 fps)  
> **Database:** MySQL 8.x (connection pooling via `mysql.connector`)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Configuration System](#3-configuration-system)
4. [File-by-File Reference](#4-file-by-file-reference)
5. [Data Flow — Step by Step](#5-data-flow--step-by-step)
6. [Database Schema](#6-database-schema)
7. [Dashboard API Reference](#7-dashboard-api-reference)
8. [Deployment & Running](#8-deployment--running)
9. [Systemd Service](#9-systemd-service)
10. [Backup & Maintenance](#10-backup--maintenance)
11. [Troubleshooting](#11-troubleshooting)
12. [Build from Scratch](#12-build-from-scratch)
13. [Possible Upgrades](#13-possible-upgrades)

---

## 1. Project Overview

A real-time bicycle counting system deployed on an NVIDIA Jetson Orin Nano. A USB camera captures video frames; a YOLO11n TensorRT model detects bicycles at ~10-18 ms per frame; a custom IoU+centroid tracker ensures each bicycle is counted exactly once; when a tracked bicycle crosses a horizontal line drawn across the middle of the frame, the system records the crossing event (timestamp, track ID) into a MySQL database and serves all analytics through a live web dashboard at port 8080.

### What it does

| Feature | Detail |
|---|---|
| **Object detection** | YOLO11n nano (80 COCO classes, only class 1 = bicycle is counted) |
| **Inference** | TensorRT FP16 engine on the Orin GPU, ~10-18 ms/frame |
| **Tracking** | IoU + centroid distance tracker with persistent IDs across frames |
| **Counting** | Objects crossing the horizontal line Y=240 are counted |
| **Deduplication** | Spatial + temporal dedup prevents counting the same bike twice |
| **Screenshots** | JPEG saved on every crossing event to `screenshots/` |
| **Database** | MySQL with per-device, per-location crossing events and 15-minute aggregated counts |
| **Web dashboard** | FastAPI at port 8080: live MJPEG video, stat cards, charts, events table |
| **Database viewer** | Separate page with full history, CSV export, auto-refresh |
| **Health endpoint** | JSON health check at `/health` for monitoring |
| **Config-driven** | All parameters in `config.yaml` with env var overrides (`BIKE_*` prefix) |
| **Multi-camera ready** | Each camera identified by `device_name` + `location_name` |

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Jetson Orin Nano                                                │
│                                                                  │
│  ┌─────────────┐     ┌──────────────────────────────────────┐   │
│  │ USB Camera   │────>│  bicycle_counter.py (main loop)       │   │
│  │ /dev/video0  │     │                                      │   │
│  │ 640x480@15fps│     │  1. cv2.VideoCapture(0).read()       │   │
│  └─────────────┘     │  2. model(frame) -> detections        │   │
│                      │  3. tracker.update(raw_dets)          │   │
│                      │  4. check persist + dedup + count     │   │
│                      │  5. save screenshot + db.record()     │   │
│                      │  6. draw_osd() -> JPEG encode         │   │
│                      │  7. state.latest_jpeg = jpeg_bytes    │   │
│                      └──────────┬───────────────────────────┘   │
│                                 │                                │
│                      ┌──────────▼───────────────────────────┐   │
│                      │  dashboard_server.py (FastAPI)         │   │
│                      │  Port 8080 (0.0.0.0)                  │   │
│                      │  GET /        -> dashboard.html        │   │
│                      │  GET /video   -> MJPEG stream          │   │
│                      │  GET /stats   -> JSON snapshot         │   │
│                      │  GET /health  -> Health check          │   │
│                      │  GET /database -> DB viewer            │   │
│                      │  GET /api/*   -> JSON data             │   │
│                      │  GET /export/* -> CSV downloads        │   │
│                      └──────────┬───────────────────────────┘   │
│                                 │                                │
│                      ┌──────────▼───────────────────────────┐   │
│                      │  database.py (MySQL)                   │   │
│                      │  Connection pool (size=4)              │   │
│                      │  Tables: devices, locations,           │   │
│                      │          crossings, quarterly_counts   │   │
│                      │  screenshots/ (max 1000 files)         │   │
│                      └──────────────────────────────────────┘   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  config.yaml  <--  config_loader.py  -->  BIKE_* env vars  │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Key design decisions

| Decision | Rationale |
|---|---|
| **TensorRT over PyTorch** | Orin Nano GPU (compute capability 8.7) doesn't match published PyTorch wheels; ultralytics loads TensorRT engine directly, bypassing PyTorch CUDA |
| **Custom tracker over DeepStream NvDCF** | DeepStream 8.0 no longer provides Python `pyds` bindings for tracking data; lightweight IoU+centroid tracker is sufficient for single-camera counting |
| **MySQL over SQLite** | Production database with connection pooling; supports multi-device deployments and concurrent dashboard reads |
| **Horizontal line counting** | Simplest approach for counting objects moving across a scene; line at Y=240 (center of 480px frame) |
| **Deduplication** | Spatial + temporal dedup prevents counting the same bicycle if it lingers near the counting line |
| **Config-driven** | All parameters in `config.yaml` with environment variable overrides; no code changes needed for tuning |
| **PIL for OSD** | Pillow used for drawing bounding boxes and text overlays (avoids OpenCV font limitations) |

---

## 3. Configuration System

### How it works

1. `config_loader.py` reads `config.yaml` from the project directory
2. Every parameter can be overridden by an environment variable prefixed with `BIKE_`
3. Type coercion is automatic (bool, int, float, str based on the default value)
4. The flattened config dict is passed to all modules at startup

### config.yaml structure

```yaml
version: "2.0"

device:
  name: "jetson-orin-01"          # Unique device identifier
  description: "Jetson Orin Nano - Main Unit"
  location_name: "Camera_01"      # Camera location name
  location_description: "Main entrance"

camera:
  device: "/dev/video0"           # V4L2 device path
  width: 640                      # Capture width
  height: 480                     # Capture height
  fps: 15                         # Target frame rate
  brightness: -30                 # V4L2 brightness (-64 to 64)

model:
  path: "yolo11n.engine"          # TensorRT engine path (relative to project dir)
  bicycle_class_id: 1             # COCO class ID for bicycle
  min_confidence: 0.45            # Minimum detection confidence

tracker:
  missing_ttl: 3.0                # Seconds before invisible track is dropped
  persist_time: 0.6               # Seconds a track must exist before counting
  iou_threshold: 0.15             # Minimum IoU for matching detection to track
  max_distance: 150               # Max centroid distance (px) for fallback matching

dedup:
  ttl: 5.0                        # Seconds to remember counted positions
  distance: 120                   # Min distance (px) between counted positions

database:
  host: "127.0.0.1"
  port: 3306
  user: "bikecounter"
  password: "BikeCount2026!"
  name: "bicycle_counter"
  pool_size: 4                    # MySQL connection pool size

retention:
  max_screenshots: 1000           # Max screenshots on disk
  keep_screenshots: 950           # Keep N newest after cleanup
  max_crossing_days: 90           # Delete crossings older than N days

dashboard:
  host: "0.0.0.0"                # Listen address
  port: 8080                      # Listen port

logging:
  level: "INFO"                   # DEBUG, INFO, WARNING, ERROR
  file: "bicycle_counter.log"     # Log file path (relative to project dir)
```

### Environment variable overrides

Every config key can be overridden. Examples:

```bash
export BIKE_CAMERA_FPS=30         # Override camera.fps
export BIKE_MIN_CONFIDENCE=0.6    # Override model.min_confidence
export BIKE_DB_HOST=192.168.1.100 # Override database.host
export BIKE_DASH_PORT=9090        # Override dashboard.port
export BIKE_LOG_LEVEL=DEBUG       # Override logging.level
```

Full mapping (YAML key -> env var -> default):

| YAML Path | Env Var | Default |
|---|---|---|
| `device.name` | `BIKE_DEVICE_NAME` | `jetson-orin-01` |
| `device.description` | `BIKE_DEVICE_DESC` | `""` |
| `device.location_name` | `BIKE_LOCATION_NAME` | `Camera_01` |
| `device.location_description` | `BIKE_LOCATION_DESC` | `""` |
| `camera.device` | `BIKE_CAMERA_DEVICE` | `/dev/video0` |
| `camera.width` | `BIKE_CAMERA_WIDTH` | `640` |
| `camera.height` | `BIKE_CAMERA_HEIGHT` | `480` |
| `camera.fps` | `BIKE_CAMERA_FPS` | `15` |
| `camera.brightness` | `BIKE_CAMERA_BRIGHTNESS` | `-30` |
| `model.path` | `BIKE_MODEL_PATH` | `yolo11n.engine` |
| `model.bicycle_class_id` | `BIKE_CLASS_ID` | `1` |
| `model.min_confidence` | `BIKE_MIN_CONFIDENCE` | `0.45` |
| `tracker.missing_ttl` | `BIKE_TRACK_MISSING_TTL` | `3.0` |
| `tracker.persist_time` | `BIKE_TRACK_PERSIST_TIME` | `0.6` |
| `tracker.iou_threshold` | `BIKE_TRACK_IOU` | `0.15` |
| `tracker.max_distance` | `BIKE_TRACK_MAX_DIST` | `150` |
| `dedup.ttl` | `BIKE_DEDUP_TTL` | `5.0` |
| `dedup.distance` | `BIKE_DEDUP_DIST` | `120` |
| `database.host` | `BIKE_DB_HOST` | `127.0.0.1` |
| `database.port` | `BIKE_DB_PORT` | `3306` |
| `database.user` | `BIKE_DB_USER` | `bikecounter` |
| `database.password` | `BIKE_DB_PASS` | `BikeCount2026!` |
| `database.name` | `BIKE_DB_NAME` | `bicycle_counter` |
| `database.pool_size` | `BIKE_DB_POOL` | `4` |
| `retention.max_screenshots` | `BIKE_MAX_SCREENSHOTS` | `1000` |
| `retention.keep_screenshots` | `BIKE_KEEP_SCREENSHOTS` | `950` |
| `retention.max_crossing_days` | `BIKE_MAX_CROSSING_DAYS` | `90` |
| `dashboard.host` | `BIKE_DASH_HOST` | `0.0.0.0` |
| `dashboard.port` | `BIKE_DASH_PORT` | `8080` |
| `logging.level` | `BIKE_LOG_LEVEL` | `INFO` |
| `logging.file` | `BIKE_LOG_FILE` | `bicycle_counter.log` |

---

## 4. File-by-File Reference

### Core Python

| File | Lines | Purpose |
|---|---|---|
| `bicycle_counter.py` | 326 | **Main pipeline.** Loads config, YOLO model, camera, MySQL, dashboard. Runs main loop: capture -> infer -> track -> count -> screenshot -> OSD -> JPEG encode. Handles signals, camera reconnect, graceful shutdown. |
| `dashboard_server.py` | 271 | **FastAPI web server.** 10 endpoints (HTML, MJPEG, JSON, CSV). Runs in daemon thread. Serves dashboard, video stream, stats, health check, database viewer, CSV exports. CORS enabled for all origins. |
| `database.py` | 210 | **MySQL module.** Connection pooling (size 4). CRUD for devices, locations, crossings, quarterly_counts. Atomic upserts for 15-min aggregation. Data retention cleanup. Screenshot cleanup (max 1000). |
| `config_loader.py` | 75 | **Config loader.** Reads `config.yaml`, flattens to dict, applies `BIKE_*` env var overrides with type coercion. |
| `export_csv.py` | 18 | **Standalone CSV export.** Connects to MySQL and dumps both quarterly counts and detailed crossings to CSV files. |

### Shell Scripts

| File | Lines | Purpose |
|---|---|---|
| `run.sh` | 32 | **Launcher.** Sets `LD_LIBRARY_PATH`/`GST_PLUGIN_PATH` for DeepStream/CUDA. Kills camera-holding processes. Prints dashboard URLs. Executes `bicycle_counter.py`. |
| `setup.sh` | 403 | **Full installer (7 steps).** System packages, Python deps, DeepStream-Yolo parser build, YOLO11n download/ONNX export, TensorRT engine build, verification. Run once on fresh Jetson. |
| `backup.sh` | 32 | **Backup script.** Dumps MySQL via `mysqldump`, tars screenshots, compresses both, cleans up backups older than 30 days. |

### Configuration & Service

| File | Lines | Purpose |
|---|---|---|
| `config.yaml` | 66 | **Central configuration.** All tunable parameters in one YAML file. |
| `bike-counter.service` | 21 | **Systemd unit file.** Runs as system service, depends on `mysql.service`, auto-restarts on failure. |

### Templates (HTML)

| File | Lines | Purpose |
|---|---|---|
| `templates/dashboard.html` | 177 | **Main dashboard.** Dark-themed, mobile-first. Live MJPEG feed, stat cards, crossing timeline (line chart), 15-min breakdown (bar chart), recent detections table. Polls `/stats` every 500ms. Uses Chart.js 4. |
| `templates/database.html` | 104 | **Database viewer.** Summary stats, 15-min aggregated counts table (with day totals), all detection events table, CSV download buttons. Auto-refreshes every 5 seconds. |

### Model Files

| File | Size | Purpose |
|---|---|---|
| `yolo11n.engine` | 7.9 MB | Pre-built TensorRT FP16 engine. **Must be rebuilt if moving to a different GPU.** |
| `yolo11n.onnx` | 10.7 MB | ONNX export of YOLO11n. Used as intermediate for TensorRT engine build. |
| `yolo11n.pt` | 5.4 MB | Original PyTorch weights. Used for ONNX export only. |

### Reference Files (not used in current pipeline)

| File | Purpose |
|---|---|
| `config_infer_primary.txt` | DeepStream nvinfer config (reference only) |
| `deepstream_app_config.txt` | DeepStream app config (reference only) |
| `labels.txt` | COCO 80-class label list |
| `libnvdsinfer_custom_impl_Yolo.so` | Compiled DeepStream-Yolo parser (reference only) |

### Data (auto-generated)

| Path | Purpose |
|---|---|
| `screenshots/*.jpg` | Crossing event screenshots, named `bike_HH-MM-SS_idN.jpg` |
| `bicycle_counter.log` | Application log file |
| `bicycle_counts.csv` | Exported 15-minute aggregated counts |
| `bicycle_crossings_detail.csv` | Exported individual crossing events |
| `backups/` | Database dumps and screenshot tarballs |

---

## 5. Data Flow — Step by Step

### 5.1 Startup sequence

1. **`run.sh`** is executed:
   - Sets `LD_LIBRARY_PATH` to include DeepStream (`/opt/nvidia/deepstream/deepstream/lib`) and CUDA 13.2 (`/usr/local/cuda-13.2/lib64`)
   - Sets `GST_PLUGIN_PATH` for GStreamer plugins
   - Sets `PIP_BREAK_SYSTEM_PACKAGES=1`
   - Runs `fuser -k /dev/video0` to kill any process holding the camera
   - Waits 0.5s for camera release
   - Prints dashboard URLs (localhost + LAN IP)
   - Executes `bicycle_counter.py` via `exec`

2. **`bicycle_counter.py`** initializes:
   - Loads config from `config.yaml` via `config_loader.load()`
   - Sets up logging (console + file)
   - Creates `screenshots/` directory if missing
   - Loads YOLO11n TensorRT engine: `YOLO("yolo11n.engine")`
   - Connects to MySQL, registers device and location (get-or-create)
   - Runs `cleanup_old_data()` to enforce retention policy
   - Opens USB camera via OpenCV V4L2 backend
   - Sets camera brightness via `v4l2-ctl`
   - Starts `dashboard_server` in a daemon thread
   - Enters main processing loop

### 5.2 Main loop (per frame)

```
capture frame (cv2.VideoCapture.read)
        |
        v
YOLO inference (model(frame, conf=0.45))
        |
        v
Filter: keep only class 1 (bicycle)
        |
        v
Tracker.update(raw_detections)
   - Expire old tracks (not seen > 3.0s)
   - Match existing tracks to new detections (IoU >= 0.15, then centroid <= 150px)
   - Create new tracks for unmatched detections
        |
        v
For each tracked detection:
   - If track is new (< 0.6s old), skip
   - If already counted, skip
   - Check dedup: is there a counted position within 120px in last 5.0s?
   - If not duplicate: mark counted, increment total, record screenshot + DB
        |
        v
Draw OSD (bounding boxes + count) using Pillow
        |
        v
Encode JPEG (quality=75) -> state.latest_jpeg
        |
        v
Save screenshot for new bikes (quality=90)
        |
        v
Record crossing in MySQL (insert into crossings + upsert quarterly_counts)
        |
        v
Cleanup screenshots if > 1000 files
```

### 5.3 Camera reconnect

After 30 consecutive `cap.read()` failures:
1. Release current camera handle
2. Wait 1 second
3. Try `open_camera()` again
4. If still fails, wait 5 seconds and retry

### 5.4 Graceful shutdown

- `SIGTERM` and `SIGINT` (Ctrl+C) trigger `_shutdown` event
- Main loop exits, prints final count
- Camera is released, DB connection closed

---

## 6. Database Schema

### MySQL tables

```sql
-- Device registry (one row per Jetson device)
CREATE TABLE devices (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Camera locations (one row per camera per device)
CREATE TABLE locations (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    device_id   INT NOT NULL,
    name        VARCHAR(100) NOT NULL,
    description TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (device_id) REFERENCES devices(id),
    UNIQUE KEY (device_id, name)
);

-- Individual crossing events
CREATE TABLE crossings (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    device_id   INT NOT NULL,
    location_id INT NOT NULL,
    recorded_at DATETIME NOT NULL,
    track_id    INT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (device_id) REFERENCES devices(id),
    FOREIGN KEY (location_id) REFERENCES locations(id),
    INDEX idx_recorded (recorded_at),
    INDEX idx_location (location_id, device_id)
);

-- 15-minute aggregated counts (atomic upserts)
CREATE TABLE quarterly_counts (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    device_id     INT NOT NULL,
    location_id   INT NOT NULL,
    bucket_date   DATE NOT NULL,
    bucket_hour   TINYINT NOT NULL,
    bucket_quarter TINYINT NOT NULL,  -- 0,1,2,3 (each = 15 min)
    total_count   INT DEFAULT 1,
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (device_id) REFERENCES devices(id),
    FOREIGN KEY (location_id) REFERENCES locations(id),
    UNIQUE KEY (device_id, location_id, bucket_date, bucket_hour, bucket_quarter)
);
```

### Quarterly bucket logic

The 15-minute bucket is calculated as: `minute // 15`

| Time Range | bucket_hour | bucket_quarter | Display |
|---|---|---|---|
| 00:00 - 00:14 | 0 | 0 | 00:00 |
| 00:15 - 00:29 | 0 | 1 | 00:15 |
| 00:30 - 00:44 | 0 | 2 | 00:30 |
| 00:45 - 00:59 | 0 | 3 | 00:45 |
| 13:30 - 13:44 | 13 | 2 | 13:30 |

### Atomic upsert

When recording a crossing, the quarterly count is updated atomically:

```sql
INSERT INTO quarterly_counts (device_id, location_id, bucket_date, bucket_hour, bucket_quarter, total_count)
VALUES (%s, %s, %s, %s, %s, 1)
ON DUPLICATE KEY UPDATE total_count = total_count + 1
```

This ensures no race conditions between the main pipeline and dashboard reads.

---

## 7. Dashboard API Reference

### Endpoints

| Method | Path | Content-Type | Description |
|---|---|---|---|
| `GET` | `/` | `text/html` | Main dashboard (renders `dashboard.html` via Jinja2) |
| `GET` | `/video` | `multipart/x-mixed-replace` | MJPEG stream of latest camera frame with OSD |
| `GET` | `/stats` | `application/json` | Full stats snapshot (see below) |
| `GET` | `/health` | `application/json` | Health check JSON |
| `GET` | `/database` | `text/html` | Database viewer page (renders `database.html`) |
| `GET` | `/api/crossings` | `application/json` | All crossing events for this device/location |
| `GET` | `/api/quarterly` | `application/json` | All 15-min aggregated counts |
| `GET` | `/export/csv` | `text/csv` | Download quarterly counts as CSV |
| `GET` | `/export/detail` | `text/csv` | Download all individual crossings as CSV |

### `/stats` response

```json
{
  "bicycle_count": 42,
  "fps": 14.5,
  "frame_count": 12345,
  "active_tracks": 2,
  "total_detections": 150,
  "camera_ok": true,
  "today_total": 42,
  "all_time_total": 91,
  "quarterly": [
    {"label": "13:30", "count": 32},
    {"label": "13:45", "count": 39}
  ],
  "events": [
    {"time_str": "13:42:15", "track_id": "7"},
    {"time_str": "13:41:02", "track_id": "6"}
  ]
}
```

### `/health` response

```json
{
  "status": "healthy",
  "version": "2.0",
  "uptime_seconds": 3600.0,
  "camera_ok": true,
  "database_ok": true,
  "bicycle_count": 42,
  "device_name": "jetson-orin-01",
  "location_name": "Camera_01"
}
```

Status is `"healthy"` when both camera and database are OK, `"degraded"` otherwise.

### `/api/crossings` response

```json
[
  {
    "id": 1,
    "timestamp": "2026-07-17T13:42:15",
    "track_id": 7,
    "location": "Camera_01"
  }
]
```

### `/api/quarterly` response

```json
[
  {
    "date": "2026-07-17",
    "period": "13:30-13:45",
    "count": 32
  }
]
```

### CORS

All origins are allowed. Only `GET` methods are permitted. This is read-only access.

---

## 8. Deployment & Running

### Quick start

```bash
cd ~/CounterProject
./run.sh
```

### What run.sh does

1. Sets `LD_LIBRARY_PATH` for DeepStream and CUDA 13.2 libraries
2. Sets `GST_PLUGIN_PATH` for GStreamer plugins
3. Runs `fuser -k /dev/video0` to kill any camera-holding process
4. Waits 0.5s for camera release
5. Prints dashboard URLs
6. Executes `bicycle_counter.py` via `exec`

### Accessing the dashboard

| URL | When to use |
|---|---|
| `http://localhost:8080` | On the Jetson itself |
| `http://<JETSON_IP>:8080` | From any device on the same LAN |
| `http://localhost:8080/health` | Health check for monitoring |
| `http://localhost:8080/database` | Database viewer |

### Stopping the system

Press `Ctrl+C` in the terminal where `run.sh` is running. The final bicycle count is printed to the log.

### Running without run.sh (manual)

```bash
export LD_LIBRARY_PATH="/opt/nvidia/deepstream/deepstream/lib:/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}"
export GST_PLUGIN_PATH="/opt/nvidia/deepstream/deepstream/lib/gst-plugins:${GST_PLUGIN_PATH:-}"
fuser -k /dev/video0 2>/dev/null; sleep 0.5
python3 ~/CounterProject/bicycle_counter.py
```

### MySQL setup (first time)

```bash
sudo mysql -u root -e "CREATE DATABASE IF NOT EXISTS bicycle_counter;"
sudo mysql -u root -e "CREATE USER IF NOT EXISTS 'bikecounter'@'127.0.0.1' IDENTIFIED BY 'BikeCount2026!';"
sudo mysql -u root -e "GRANT ALL PRIVILEGES ON bicycle_counter.* TO 'bikecounter'@'127.0.0.1';"
sudo mysql -u root -e "FLUSH PRIVILEGES;"
```

Tables are created automatically on first run (handled by `mysql.connector` pool initialization).

---

## 9. Systemd Service

### Service file: `bike-counter.service`

```ini
[Unit]
Description=Bicycle Counter Pipeline
After=network.target mysql.service
Wants=mysql.service

[Service]
Type=simple
User=student
WorkingDirectory=/home/student/CounterProject
ExecStartPre=/usr/bin/fuser -k /dev/video0 2>/dev/null || true
ExecStart=/home/student/CounterProject/run.sh
Restart=on-failure
RestartSec=5
TimeoutStopSec=10
KillMode=mixed
KillSignal=SIGTERM
Environment=LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:/usr/local/cuda-13.2/lib64
Environment=PIP_BREAK_SYSTEM_PACKAGES=1

[Install]
WantedBy=multi-user.target
```

### Installing and enabling

```bash
sudo cp bike-counter.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable bike-counter.service
sudo systemctl start bike-counter.service
```

### Managing the service

```bash
sudo systemctl status bike-counter.service    # Check status
sudo systemctl stop bike-counter.service      # Stop
sudo systemctl restart bike-counter.service   # Restart
sudo journalctl -u bike-counter -f            # Follow logs
sudo journalctl -u bike-counter --since today  # Today's logs
```

### Disabling the service

```bash
sudo systemctl stop bike-counter.service
sudo systemctl disable bike-counter.service
```

---

## 10. Backup & Maintenance

### Backup script: `backup.sh`

```bash
./backup.sh                    # Default: ./backups/
./backup.sh /path/to/backups   # Custom backup directory
```

What it does:
1. Dumps MySQL database via `mysqldump` (compressed with gzip)
2. Creates tarball of `screenshots/` directory
3. Deletes backup files older than 30 days

### Data retention

Enforced automatically at startup and on each crossing event:

| What | Limit | Behavior |
|---|---|---|
| Screenshots | Max 1000 files | Deletes oldest, keeps 950 newest |
| Crossing events | Max 90 days | Deletes rows older than 90 days |
| Quarterly counts | Max 90 days | Deletes rows older than 90 days |

### Log rotation

The application writes to `bicycle_counter.log`. For production, add a logrotate config:

```
# /etc/logrotate.d/bicycle-counter
/home/student/CounterProject/bicycle_counter.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
```

### Database credentials

Stored in `config.yaml` under `database:` section. The default credentials are:
- **User:** `bikecounter`
- **Password:** `BikeCount2026!`
- **Database:** `bicycle_counter`
- **Host:** `127.0.0.1:3306`

**IMPORTANT:** Change these credentials for production deployments.

---

## 11. Troubleshooting

### Camera Issues

| Symptom | Fix |
|---|---|
| `Cannot open camera` / `cap.isOpened() returns False` | Run `fuser -k /dev/video0` then restart. Another process holds the camera. |
| Camera overexposed (white in sun) | `v4l2-ctl -d /dev/video0 --set-ctrl brightness=-40`. Range: -64 to 64. |
| `VIDEOIO ERROR: V4L2: pixel format not supported` | Camera doesn't support MJPEG at 640x480. Try different USB camera. |
| Low frame rate | Check `camera.fps` in config.yaml (default 15). Some cameras max at 15fps at 640x480. |
| Camera reconnects repeatedly | Check USB cable, power supply, or try different USB port. |

### Inference Issues

| Symptom | Fix |
|---|---|
| `model(frame)` hangs or takes > 1s | TensorRT engine may be built for different GPU. Rebuild: `trtexec --onnx=yolo11n.onnx --saveEngine=yolo11n.engine --fp16` |
| CUDA out-of-memory | Orin Nano shares 8GB between CPU/GPU. Close other GPU apps. |
| `ultralytics` not found | `pip3 install ultralytics` with `export PIP_BREAK_SYSTEM_PACKAGES=1` |
| PyTorch CUDA mismatch warnings | Normal -- ultralytics bypasses PyTorch for TensorRT. Warnings are harmless. |

### Database Issues

| Symptom | Fix |
|---|---|
| `Access denied for user 'bikecounter'` | Check MySQL is running: `sudo systemctl status mysql`. Verify credentials in `config.yaml`. |
| `Can't connect to MySQL server` | Ensure MySQL is running: `sudo systemctl start mysql`. Check host/port in config. |
| Connection pool exhausted | Increase `database.pool_size` in config.yaml (default: 4). |
| Want to reset all data | `sudo mysql -u root -e "DROP DATABASE bicycle_counter; CREATE DATABASE bicycle_counter;"` |
| Want to reset screenshots | `rm ~/CounterProject/screenshots/*.jpg` |

### Dashboard Issues

| Symptom | Fix |
|---|---|
| Port 8080 already in use | `fuser -k 8080/tcp` then restart |
| Dashboard loads but video is black | Pipeline crashed. Check terminal for `[ERROR]` messages. |
| Dashboard shows "Pipeline not running" | Main loop not running. Check Python exceptions. |
| Charts don't update | Open browser DevTools -> Console. Check for fetch errors to `/stats`. |
| Database viewer shows "Loading..." | Check `/api/crossings` and `/api/quarterly` return data. |

### General System Issues

| Symptom | Fix |
|---|---|
| `ld.so.conf.d` errors about DeepStream libs | Ensure `/etc/ld.so.conf.d/deepstream.conf` contains `/opt/nvidia/deepstream/deepstream/lib` then `sudo ldconfig` |
| `pyds` import errors | pyds is NOT used in current pipeline. Ignore. |
| System slow after boot | Old `bike-counter.service` may be running. Disable: `sudo systemctl stop bike-counter && sudo systemctl disable bike-counter` |
| `mysql.connector` import error | `pip3 install mysql-connector-python` with `export PIP_BREAK_SYSTEM_PACKAGES=1` |

---

## 12. Build from Scratch

> **Prerequisites:** Fresh Jetson Orin Nano with JetPack 6.x (L4T 39.x), USB camera, internet access, sudo privileges.

### Quick method (automated)

```bash
cd ~/CounterProject
sudo chmod +x setup.sh
sudo ./setup.sh
```

### Manual method (step by step)

#### Step 1: System packages

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y build-essential cmake git wget pkg-config \
    python3-dev python3-pip python3-gi python3-gi-cairo \
    gir1.2-gstreamer-1.0 gir1.2-gst-rtsp-server-1.0 \
    libgstrtspserver-1.0-dev libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly gstreamer1.0-tools \
    libcairo2-dev libgirepository1.0-dev
```

#### Step 2: Python dependencies

```bash
export PIP_BREAK_SYSTEM_PACKAGES=1
pip3 install numpy "numpy<2"
pip3 install ultralytics onnx onnxslim
pip3 install fastapi "uvicorn[standard]" jinja2 python-multipart Pillow
pip3 install mysql-connector-python
```

> **DO NOT** `pip install --upgrade pip` -- the Debian-managed pip must stay intact for dpkg tracking.

#### Step 3: Verify DeepStream SDK

```bash
ls -la /opt/nvidia/deepstream/deepstream
# If missing, find and symlink:
# sudo mkdir -p /opt/nvidia/deepstream
# sudo ln -sf /path/to/deepstream-X.Y /opt/nvidia/deepstream/deepstream
```

#### Step 4: Build TensorRT engine (if needed)

The project ships a pre-built `yolo11n.engine`. Only rebuild if moving to a different GPU:

```bash
cd ~/CounterProject
trtexec --onnx=yolo11n.onnx --saveEngine=yolo11n.engine --fp16
```

#### Step 5: MySQL setup

```bash
sudo systemctl start mysql
sudo mysql -u root -e "CREATE DATABASE IF NOT EXISTS bicycle_counter;"
sudo mysql -u root -e "CREATE USER IF NOT EXISTS 'bikecounter'@'127.0.0.1' IDENTIFIED BY 'BikeCount2026!';"
sudo mysql -u root -e "GRANT ALL PRIVILEGES ON bicycle_counter.* TO 'bikecounter'@'127.0.0.1';"
sudo mysql -u root -e "FLUSH PRIVILEGES;"
```

#### Step 6: Verify installation

```bash
cd ~/CounterProject
python3 -c "from ultralytics import YOLO; print('ultralytics OK')"
python3 -c "import cv2; print('OpenCV OK')"
python3 -c "import fastapi; print('FastAPI OK')"
python3 -c "import mysql.connector; print('MySQL connector OK')"
python3 -c "from config_loader import load; print('Config OK:', load()['version'])"
```

---

## 13. Possible Upgrades

### Short-term (feature additions)

| Upgrade | Description | Effort |
|---|---|---|
| **Adjustable parameters via dashboard** | Settings panel to change `CROSSING_LINE_Y`, `MIN_CONFIDENCE`, `CAMERA_BRIGHTNESS` without restart | Medium |
| **Email/Telegram alerts** | Notification when threshold reached (e.g. > 100 bicycles/hour) | Low |
| **Historical date picker** | View past days' data with date navigation in dashboard | Medium |
| **Screenshot gallery** | Browse captured screenshots in dashboard with thumbnails | Medium |
| **WebSocket live events** | Replace 500ms polling with WebSocket for real-time dashboard updates | Medium |

### Mid-term (deployment hardening)

| Upgrade | Description | Effort |
|---|---|---|
| **Multi-camera support** | Multiple instances with different `location_name`, unified dashboard | High |
| **HTTPS / reverse proxy** | nginx with Let's Encrypt TLS for secure remote access | Medium |
| **Docker containerization** | Package as Docker container with `nvidia-docker2` GPU access | High |
| **OTA updates** | Over-the-air update mechanism for deployed Jetson devices | High |

### Long-term (advanced features)

| Upgrade | Description | Effort |
|---|---|---|
| **Custom YOLO training** | Fine-tune on bicycle-specific dataset for higher accuracy | High |
| **Speed estimation** | Frame-to-frame displacement with camera calibration | High |
| **Multi-class counting** | Extend to pedestrians, cars, scooters | Medium |
| **Edge-cloud sync** | Sync local MySQL to cloud dashboard (Firebase, AWS IoT) | High |
| **Analytics API** | RESTful API for external systems to query historical data | Medium |
| **GPU utilization monitoring** | Real-time GPU/CPU/memory stats in dashboard | Low |

---

## Appendix: Quick Reference Card

### Start/Stop

```bash
./run.sh                              # Start manually
sudo systemctl start bike-counter     # Start via systemd
sudo systemctl stop bike-counter      # Stop
```

### URLs

```
http://localhost:8080                  # Dashboard
http://localhost:8080/database         # Database viewer
http://localhost:8080/health           # Health check
http://localhost:8080/stats            # Stats JSON
http://localhost:8080/export/csv       # Download 15-min counts CSV
http://localhost:8080/export/detail    # Download all crossings CSV
```

### Config

```bash
nano ~/CounterProject/config.yaml     # Edit config
export BIKE_DASH_PORT=9090            # Override via env var
```

### Database

```bash
mysql -u bikecounter -p'BikeCount2026!' bicycle_counter   # Connect
./backup.sh                           # Backup DB + screenshots
```

### Logs

```bash
tail -f ~/CounterProject/bicycle_counter.log              # Follow log
sudo journalctl -u bike-counter -f                        # Follow systemd log
```
