#!/usr/bin/env python3
"""Generate Word (.docx) documentation for Bicycle Counter v2.0."""

from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml
import os

doc = Document()

# ── Styles ───────────────────────────────────────────────────────────
style = doc.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(11)
style.paragraph_format.space_after = Pt(4)
style.paragraph_format.line_spacing = 1.15

for level in range(1, 5):
    h = doc.styles[f'Heading {level}']
    h.font.name = 'Calibri'
    h.font.color.rgb = RGBColor(0x1A, 0x5C, 0x9E)

doc.styles['Heading 1'].font.size = Pt(20)
doc.styles['Heading 2'].font.size = Pt(16)
doc.styles['Heading 3'].font.size = Pt(13)

# Code style
code_style = doc.styles.add_style('CodeBlock', WD_STYLE_TYPE.PARAGRAPH)
code_style.font.name = 'Consolas'
code_style.font.size = Pt(9)
code_style.font.color.rgb = RGBColor(0x1E, 0x1E, 0x1E)
code_style.paragraph_format.space_before = Pt(4)
code_style.paragraph_format.space_after = Pt(4)
code_style.paragraph_format.left_indent = Cm(0.5)

# ── Helpers ──────────────────────────────────────────────────────────
def add_code(text):
    for line in text.strip().split('\n'):
        p = doc.add_paragraph(line, style='CodeBlock')
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F5F5F5"/>')
        p.paragraph_format.element.get_or_add_pPr().append(shading)

def add_table(headers, rows, col_widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    # Header row
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(10)
    # Data rows
    for row_data in rows:
        row = table.add_row()
        for i, val in enumerate(row_data):
            cell = row.cells[i]
            cell.text = str(val)
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10)
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Inches(w)
    doc.add_paragraph()

def add_bullet(text, bold_prefix=None):
    p = doc.add_paragraph(style='List Bullet')
    if bold_prefix:
        run = p.add_run(bold_prefix)
        run.bold = True
        p.add_run(text)
    else:
        p.add_run(text)

# ══════════════════════════════════════════════════════════════════════
# TITLE PAGE
# ══════════════════════════════════════════════════════════════════════
for _ in range(6):
    doc.add_paragraph()

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run('Bicycle Counter v2.0')
run.font.size = Pt(28)
run.bold = True
run.font.color.rgb = RGBColor(0x1A, 0x5C, 0x9E)

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run('Full Project Specification')
run.font.size = Pt(16)
run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

doc.add_paragraph()

info_lines = [
    'Target Hardware: NVIDIA Jetson Orin Nano (8 GB)',
    'Firmware: JetPack 6.x / L4T 39.2.0',
    'CUDA: 13.2 | TensorRT: bundled | Python: 3.12',
    'Camera: Suyin HD USB Camera (/dev/video0, 640x480 @ 15 fps)',
    'Database: MySQL 8.x',
    '',
    'Document version: 2.0',
    'Date: July 2026',
]
for line in info_lines:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(line)
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════
# TABLE OF CONTENTS (placeholder)
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('Table of Contents', level=1)
toc_items = [
    '1. Project Overview',
    '2. System Architecture',
    '3. Configuration System',
    '4. File-by-File Reference',
    '5. Data Flow - Step by Step',
    '6. Database Schema',
    '7. Dashboard API Reference',
    '8. Deployment & Running',
    '9. Systemd Service',
    '10. Backup & Maintenance',
    '11. Troubleshooting',
    '12. Build from Scratch',
    '13. Possible Upgrades',
    'Appendix: Quick Reference Card',
]
for item in toc_items:
    p = doc.add_paragraph(item)
    p.paragraph_format.space_after = Pt(2)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════
# 1. PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('1. Project Overview', level=1)

doc.add_paragraph(
    'A real-time bicycle counting system deployed on an NVIDIA Jetson Orin Nano. '
    'A USB camera captures video frames; a YOLO11n TensorRT model detects bicycles '
    'at ~10-18 ms per frame; a custom IoU+centroid tracker ensures each bicycle is '
    'counted exactly once; when a tracked bicycle crosses a horizontal line drawn '
    'across the middle of the frame, the system records the crossing event (timestamp, '
    'track ID) into a MySQL database and serves all analytics through a live web '
    'dashboard at port 8080.'
)

doc.add_heading('Features', level=2)
add_table(
    ['Feature', 'Detail'],
    [
        ['Object detection', 'YOLO11n nano (80 COCO classes, only class 1 = bicycle counted)'],
        ['Inference', 'TensorRT FP16 engine on Orin GPU, ~10-18 ms/frame'],
        ['Tracking', 'IoU + centroid distance tracker with persistent IDs across frames'],
        ['Counting', 'Objects crossing horizontal line Y=240 are counted'],
        ['Deduplication', 'Spatial + temporal dedup prevents counting the same bike twice'],
        ['Screenshots', 'JPEG saved on every crossing event to screenshots/'],
        ['Database', 'MySQL with per-device, per-location events and 15-min aggregated counts'],
        ['Web dashboard', 'FastAPI at port 8080: live MJPEG video, stat cards, charts, events'],
        ['Database viewer', 'Separate page with full history, CSV export, auto-refresh'],
        ['Health endpoint', 'JSON health check at /health for monitoring'],
        ['Config-driven', 'All parameters in config.yaml with BIKE_* env var overrides'],
        ['Multi-camera ready', 'Each camera identified by device_name + location_name'],
    ],
    col_widths=[2.0, 4.5]
)

# ══════════════════════════════════════════════════════════════════════
# 2. SYSTEM ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('2. System Architecture', level=1)

doc.add_heading('Architecture Diagram', level=2)
arch_text = """
Jetson Orin Nano
|
+------------------+     +----------------------------------------+
| USB Camera       |---->|  bicycle_counter.py (main loop)         |
| /dev/video0      |     |                                        |
| 640x480@15fps    |     |  1. cv2.VideoCapture(0).read()         |
+------------------+     |  2. model(frame) -> detections          |
                         |  3. tracker.update(raw_dets)            |
                         |  4. check persist + dedup + count       |
                         |  5. save screenshot + db.record()       |
                         |  6. draw_osd() -> JPEG encode           |
                         |  7. state.latest_jpeg = jpeg_bytes      |
                         +----------------+------------------------+
                                          |
                         +----------------v------------------------+
                         |  dashboard_server.py (FastAPI)           |
                         |  Port 8080 (0.0.0.0)                    |
                         |  GET /        -> dashboard.html          |
                         |  GET /video   -> MJPEG stream            |
                         |  GET /stats   -> JSON snapshot           |
                         |  GET /health  -> Health check            |
                         |  GET /database -> DB viewer              |
                         |  GET /api/*   -> JSON data               |
                         |  GET /export/* -> CSV downloads          |
                         +----------------+------------------------+
                                          |
                         +----------------v------------------------+
                         |  database.py (MySQL)                     |
                         |  Connection pool (size=4)                |
                         |  Tables: devices, locations,             |
                         |          crossings, quarterly_counts     |
                         |  screenshots/ (max 1000 files)           |
                         +----------------------------------------+
                                          |
                         +----------------------------------------+
                         |  config.yaml  <-  config_loader.py      |
                         |                   BIKE_* env vars       |
                         +----------------------------------------+
"""
add_code(arch_text)

doc.add_heading('Key Design Decisions', level=2)
add_table(
    ['Decision', 'Rationale'],
    [
        ['TensorRT over PyTorch', 'Orin Nano GPU (compute capability 8.7) does not match published PyTorch wheels; ultralytics loads TensorRT engine directly'],
        ['Custom tracker over DeepStream NvDCF', 'DeepStream 8.0 no longer provides Python pyds bindings for tracking data; lightweight IoU+centroid tracker is sufficient'],
        ['MySQL over SQLite', 'Production database with connection pooling; supports multi-device deployments and concurrent dashboard reads'],
        ['Horizontal line counting', 'Simplest approach for counting objects moving across a scene; line at Y=240 (center of 480px frame)'],
        ['Deduplication', 'Spatial + temporal dedup prevents counting the same bicycle if it lingers near the counting line'],
        ['Config-driven', 'All parameters in config.yaml with environment variable overrides; no code changes needed for tuning'],
        ['PIL for OSD', 'Pillow used for drawing bounding boxes and text overlays (avoids OpenCV font limitations)'],
    ],
    col_widths=[2.2, 4.3]
)

# ══════════════════════════════════════════════════════════════════════
# 3. CONFIGURATION SYSTEM
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('3. Configuration System', level=1)

doc.add_heading('How it works', level=2)
add_bullet('config_loader.py reads config.yaml from the project directory')
add_bullet('Every parameter can be overridden by an environment variable prefixed with BIKE_')
add_bullet('Type coercion is automatic (bool, int, float, str based on the default value)')
add_bullet('The flattened config dict is passed to all modules at startup')

doc.add_heading('config.yaml structure', level=2)
add_code("""
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
  path: "yolo11n.engine"          # TensorRT engine path
  bicycle_class_id: 1             # COCO class ID for bicycle
  min_confidence: 0.45            # Minimum detection confidence

tracker:
  missing_ttl: 3.0                # Seconds before invisible track is dropped
  persist_time: 0.6               # Seconds a track must exist before counting
  iou_threshold: 0.15             # Minimum IoU for matching
  max_distance: 150               # Max centroid distance (px) for fallback

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
  file: "bicycle_counter.log"     # Log file path
""")

doc.add_heading('Environment Variable Overrides', level=2)
doc.add_paragraph('Every config key can be overridden. Examples:')
add_code("""
export BIKE_CAMERA_FPS=30         # Override camera.fps
export BIKE_MIN_CONFIDENCE=0.6    # Override model.min_confidence
export BIKE_DB_HOST=192.168.1.100 # Override database.host
export BIKE_DASH_PORT=9090        # Override dashboard.port
export BIKE_LOG_LEVEL=DEBUG       # Override logging.level
""")

doc.add_paragraph('Full mapping (YAML key -> env var -> default):')
env_rows = [
    ['device.name', 'BIKE_DEVICE_NAME', 'jetson-orin-01'],
    ['device.description', 'BIKE_DEVICE_DESC', '""'],
    ['device.location_name', 'BIKE_LOCATION_NAME', 'Camera_01'],
    ['device.location_description', 'BIKE_LOCATION_DESC', '""'],
    ['camera.device', 'BIKE_CAMERA_DEVICE', '/dev/video0'],
    ['camera.width', 'BIKE_CAMERA_WIDTH', '640'],
    ['camera.height', 'BIKE_CAMERA_HEIGHT', '480'],
    ['camera.fps', 'BIKE_CAMERA_FPS', '15'],
    ['camera.brightness', 'BIKE_CAMERA_BRIGHTNESS', '-30'],
    ['model.path', 'BIKE_MODEL_PATH', 'yolo11n.engine'],
    ['model.bicycle_class_id', 'BIKE_CLASS_ID', '1'],
    ['model.min_confidence', 'BIKE_MIN_CONFIDENCE', '0.45'],
    ['tracker.missing_ttl', 'BIKE_TRACK_MISSING_TTL', '3.0'],
    ['tracker.persist_time', 'BIKE_TRACK_PERSIST_TIME', '0.6'],
    ['tracker.iou_threshold', 'BIKE_TRACK_IOU', '0.15'],
    ['tracker.max_distance', 'BIKE_TRACK_MAX_DIST', '150'],
    ['dedup.ttl', 'BIKE_DEDUP_TTL', '5.0'],
    ['dedup.distance', 'BIKE_DEDUP_DIST', '120'],
    ['database.host', 'BIKE_DB_HOST', '127.0.0.1'],
    ['database.port', 'BIKE_DB_PORT', '3306'],
    ['database.user', 'BIKE_DB_USER', 'bikecounter'],
    ['database.password', 'BIKE_DB_PASS', 'BikeCount2026!'],
    ['database.name', 'BIKE_DB_NAME', 'bicycle_counter'],
    ['database.pool_size', 'BIKE_DB_POOL', '4'],
    ['retention.max_screenshots', 'BIKE_MAX_SCREENSHOTS', '1000'],
    ['retention.keep_screenshots', 'BIKE_KEEP_SCREENSHOTS', '950'],
    ['retention.max_crossing_days', 'BIKE_MAX_CROSSING_DAYS', '90'],
    ['dashboard.host', 'BIKE_DASH_HOST', '0.0.0.0'],
    ['dashboard.port', 'BIKE_DASH_PORT', '8080'],
    ['logging.level', 'BIKE_LOG_LEVEL', 'INFO'],
    ['logging.file', 'BIKE_LOG_FILE', 'bicycle_counter.log'],
]
add_table(['YAML Path', 'Env Var', 'Default'], env_rows, col_widths=[2.2, 2.3, 2.0])

# ══════════════════════════════════════════════════════════════════════
# 4. FILE-BY-FILE REFERENCE
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('4. File-by-File Reference', level=1)

doc.add_heading('Core Python', level=2)
add_table(
    ['File', 'Lines', 'Purpose'],
    [
        ['bicycle_counter.py', '326', 'Main pipeline. Loads config, YOLO model, camera, MySQL, dashboard. Runs main loop: capture -> infer -> track -> count -> screenshot -> OSD -> JPEG encode. Handles signals, camera reconnect, graceful shutdown.'],
        ['dashboard_server.py', '271', 'FastAPI web server. 10 endpoints (HTML, MJPEG, JSON, CSV). Runs in daemon thread. Serves dashboard, video stream, stats, health check, database viewer, CSV exports.'],
        ['database.py', '210', 'MySQL module. Connection pooling (size 4). CRUD for devices, locations, crossings, quarterly_counts. Atomic upserts for 15-min aggregation. Data retention + screenshot cleanup.'],
        ['config_loader.py', '75', 'Config loader. Reads config.yaml, flattens to dict, applies BIKE_* env var overrides with type coercion.'],
        ['export_csv.py', '18', 'Standalone CSV export. Connects to MySQL and dumps both quarterly counts and detailed crossings to CSV files.'],
    ],
    col_widths=[1.8, 0.6, 4.1]
)

doc.add_heading('Shell Scripts', level=2)
add_table(
    ['File', 'Lines', 'Purpose'],
    [
        ['run.sh', '32', 'Launcher. Sets LD_LIBRARY_PATH/GST_PLUGIN_PATH for DeepStream/CUDA. Kills camera-holding processes. Prints dashboard URLs. Executes bicycle_counter.py.'],
        ['setup.sh', '403', 'Full installer (7 steps). System packages, Python deps, DeepStream-Yolo parser build, YOLO11n download/ONNX export, TensorRT engine build, verification.'],
        ['backup.sh', '32', 'Backup script. Dumps MySQL via mysqldump, tars screenshots, compresses both, cleans up backups older than 30 days.'],
    ],
    col_widths=[1.8, 0.6, 4.1]
)

doc.add_heading('Configuration & Service', level=2)
add_table(
    ['File', 'Lines', 'Purpose'],
    [
        ['config.yaml', '66', 'Central configuration. All tunable parameters in one YAML file.'],
        ['bike-counter.service', '21', 'Systemd unit file. Runs as system service, depends on mysql.service, auto-restarts on failure.'],
    ],
    col_widths=[1.8, 0.6, 4.1]
)

doc.add_heading('Templates (HTML)', level=2)
add_table(
    ['File', 'Lines', 'Purpose'],
    [
        ['templates/dashboard.html', '177', 'Main dashboard. Dark-themed, mobile-first. Live MJPEG feed, stat cards, crossing timeline, 15-min bar chart, recent detections. Polls /stats every 500ms. Chart.js 4.'],
        ['templates/database.html', '104', 'Database viewer. Summary stats, 15-min counts table, all events table, CSV downloads. Auto-refreshes every 5 seconds.'],
    ],
    col_widths=[1.8, 0.6, 4.1]
)

doc.add_heading('Model Files', level=2)
add_table(
    ['File', 'Size', 'Purpose'],
    [
        ['yolo11n.engine', '7.9 MB', 'Pre-built TensorRT FP16 engine. Must be rebuilt if moving to a different GPU.'],
        ['yolo11n.onnx', '10.7 MB', 'ONNX export of YOLO11n. Intermediate for TensorRT engine build.'],
        ['yolo11n.pt', '5.4 MB', 'Original PyTorch weights. Used for ONNX export only.'],
    ],
    col_widths=[1.8, 0.8, 3.9]
)

doc.add_heading('Reference Files (not used in current pipeline)', level=2)
add_table(
    ['File', 'Purpose'],
    [
        ['config_infer_primary.txt', 'DeepStream nvinfer config (reference only)'],
        ['deepstream_app_config.txt', 'DeepStream app config (reference only)'],
        ['labels.txt', 'COCO 80-class label list'],
        ['libnvdsinfer_custom_impl_Yolo.so', 'Compiled DeepStream-Yolo parser (reference only)'],
    ],
    col_widths=[2.5, 4.0]
)

doc.add_heading('Data (auto-generated)', level=2)
add_table(
    ['Path', 'Purpose'],
    [
        ['screenshots/*.jpg', 'Crossing event screenshots, named bike_HH-MM-SS_idN.jpg'],
        ['bicycle_counter.log', 'Application log file'],
        ['bicycle_counts.csv', 'Exported 15-minute aggregated counts'],
        ['bicycle_crossings_detail.csv', 'Exported individual crossing events'],
        ['backups/', 'Database dumps and screenshot tarballs'],
    ],
    col_widths=[2.5, 4.0]
)

# ══════════════════════════════════════════════════════════════════════
# 5. DATA FLOW
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('5. Data Flow - Step by Step', level=1)

doc.add_heading('5.1 Startup Sequence', level=2)
doc.add_paragraph('run.sh is executed:')
add_bullet('Sets LD_LIBRARY_PATH to include DeepStream and CUDA 13.2')
add_bullet('Sets GST_PLUGIN_PATH for GStreamer plugins')
add_bullet('Sets PIP_BREAK_SYSTEM_PACKAGES=1')
add_bullet('Runs fuser -k /dev/video0 to kill any camera-holding process')
add_bullet('Waits 0.5s for camera release')
add_bullet('Prints dashboard URLs (localhost + LAN IP)')
add_bullet('Executes bicycle_counter.py via exec')

doc.add_paragraph()
doc.add_paragraph('bicycle_counter.py initializes:')
add_bullet('Loads config from config.yaml via config_loader.load()')
add_bullet('Sets up logging (console + file)')
add_bullet('Creates screenshots/ directory if missing')
add_bullet('Loads YOLO11n TensorRT engine: YOLO("yolo11n.engine")')
add_bullet('Connects to MySQL, registers device and location (get-or-create)')
add_bullet('Runs cleanup_old_data() to enforce retention policy')
add_bullet('Opens USB camera via OpenCV V4L2 backend')
add_bullet('Sets camera brightness via v4l2-ctl')
add_bullet('Starts dashboard_server in a daemon thread')
add_bullet('Enters main processing loop')

doc.add_heading('5.2 Main Loop (per frame)', level=2)
add_code("""
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
""")

doc.add_heading('5.3 Camera Reconnect', level=2)
doc.add_paragraph('After 30 consecutive cap.read() failures:')
add_bullet('Release current camera handle')
add_bullet('Wait 1 second')
add_bullet('Try open_camera() again')
add_bullet('If still fails, wait 5 seconds and retry')

doc.add_heading('5.4 Graceful Shutdown', level=2)
add_bullet('SIGTERM and SIGINT (Ctrl+C) trigger _shutdown event')
add_bullet('Main loop exits, prints final count')
add_bullet('Camera is released, DB connection closed')

# ══════════════════════════════════════════════════════════════════════
# 6. DATABASE SCHEMA
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('6. Database Schema', level=1)

doc.add_heading('MySQL Tables', level=2)

doc.add_heading('devices', level=3)
add_code("""
CREATE TABLE devices (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

doc.add_heading('locations', level=3)
add_code("""
CREATE TABLE locations (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    device_id   INT NOT NULL,
    name        VARCHAR(100) NOT NULL,
    description TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (device_id) REFERENCES devices(id),
    UNIQUE KEY (device_id, name)
);
""")

doc.add_heading('crossings', level=3)
add_code("""
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
""")

doc.add_heading('quarterly_counts', level=3)
add_code("""
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
""")

doc.add_heading('Quarterly Bucket Logic', level=2)
doc.add_paragraph('The 15-minute bucket is calculated as: minute // 15')
add_table(
    ['Time Range', 'bucket_hour', 'bucket_quarter', 'Display'],
    [
        ['00:00 - 00:14', '0', '0', '00:00'],
        ['00:15 - 00:29', '0', '1', '00:15'],
        ['00:30 - 00:44', '0', '2', '00:30'],
        ['00:45 - 00:59', '0', '3', '00:45'],
        ['13:30 - 13:44', '13', '2', '13:30'],
    ],
    col_widths=[1.8, 1.2, 1.5, 1.2]
)

doc.add_heading('Atomic Upsert', level=2)
doc.add_paragraph('When recording a crossing, the quarterly count is updated atomically:')
add_code("""
INSERT INTO quarterly_counts
    (device_id, location_id, bucket_date, bucket_hour, bucket_quarter, total_count)
VALUES (%s, %s, %s, %s, %s, 1)
ON DUPLICATE KEY UPDATE total_count = total_count + 1
""")
doc.add_paragraph('This ensures no race conditions between the main pipeline and dashboard reads.')

# ══════════════════════════════════════════════════════════════════════
# 7. DASHBOARD API REFERENCE
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('7. Dashboard API Reference', level=1)

doc.add_heading('Endpoints', level=2)
add_table(
    ['Method', 'Path', 'Content-Type', 'Description'],
    [
        ['GET', '/', 'text/html', 'Main dashboard (renders dashboard.html via Jinja2)'],
        ['GET', '/video', 'multipart/x-mixed-replace', 'MJPEG stream of latest camera frame with OSD'],
        ['GET', '/stats', 'application/json', 'Full stats snapshot'],
        ['GET', '/health', 'application/json', 'Health check JSON'],
        ['GET', '/database', 'text/html', 'Database viewer page'],
        ['GET', '/api/crossings', 'application/json', 'All crossing events for this device/location'],
        ['GET', '/api/quarterly', 'application/json', 'All 15-min aggregated counts'],
        ['GET', '/export/csv', 'text/csv', 'Download quarterly counts as CSV'],
        ['GET', '/export/detail', 'text/csv', 'Download all individual crossings as CSV'],
    ],
    col_widths=[0.6, 1.5, 1.8, 2.6]
)

doc.add_heading('/stats Response', level=2)
add_code("""{
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
}""")

doc.add_heading('/health Response', level=2)
add_code("""{
  "status": "healthy",
  "version": "2.0",
  "uptime_seconds": 3600.0,
  "camera_ok": true,
  "database_ok": true,
  "bicycle_count": 42,
  "device_name": "jetson-orin-01",
  "location_name": "Camera_01"
}""")
doc.add_paragraph('Status is "healthy" when both camera and database are OK, "degraded" otherwise.')

doc.add_heading('/api/crossings Response', level=2)
add_code("""[
  {
    "id": 1,
    "timestamp": "2026-07-17T13:42:15",
    "track_id": 7,
    "location": "Camera_01"
  }
]""")

doc.add_heading('/api/quarterly Response', level=2)
add_code("""[
  {
    "date": "2026-07-17",
    "period": "13:30-13:45",
    "count": 32
  }
]""")

doc.add_heading('CORS', level=2)
doc.add_paragraph('All origins are allowed. Only GET methods are permitted. This is read-only access.')

# ══════════════════════════════════════════════════════════════════════
# 8. DEPLOYMENT & RUNNING
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('8. Deployment & Running', level=1)

doc.add_heading('Quick Start', level=2)
add_code("""
cd ~/CounterProject
./run.sh
""")

doc.add_heading('What run.sh Does', level=2)
add_bullet('Sets LD_LIBRARY_PATH for DeepStream and CUDA 13.2 libraries')
add_bullet('Sets GST_PLUGIN_PATH for GStreamer plugins')
add_bullet('Runs fuser -k /dev/video0 to kill any camera-holding process')
add_bullet('Waits 0.5s for camera release')
add_bullet('Prints dashboard URLs')
add_bullet('Executes bicycle_counter.py via exec')

doc.add_heading('Accessing the Dashboard', level=2)
add_table(
    ['URL', 'When to use'],
    [
        ['http://localhost:8080', 'On the Jetson itself'],
        ['http://<JETSON_IP>:8080', 'From any device on the same LAN'],
        ['http://localhost:8080/health', 'Health check for monitoring'],
        ['http://localhost:8080/database', 'Database viewer'],
    ],
    col_widths=[2.5, 4.0]
)

doc.add_heading('Stopping the System', level=2)
doc.add_paragraph('Press Ctrl+C in the terminal where run.sh is running. The final bicycle count is printed to the log.')

doc.add_heading('Running Without run.sh (Manual)', level=2)
add_code("""
export LD_LIBRARY_PATH="/opt/nvidia/deepstream/deepstream/lib:/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}"
export GST_PLUGIN_PATH="/opt/nvidia/deepstream/deepstream/lib/gst-plugins:${GST_PLUGIN_PATH:-}"
fuser -k /dev/video0 2>/dev/null; sleep 0.5
python3 ~/CounterProject/bicycle_counter.py
""")

doc.add_heading('MySQL Setup (First Time)', level=2)
add_code("""
sudo mysql -u root -e "CREATE DATABASE IF NOT EXISTS bicycle_counter;"
sudo mysql -u root -e "CREATE USER IF NOT EXISTS 'bikecounter'@'127.0.0.1' IDENTIFIED BY 'BikeCount2026!';"
sudo mysql -u root -e "GRANT ALL PRIVILEGES ON bicycle_counter.* TO 'bikecounter'@'127.0.0.1';"
sudo mysql -u root -e "FLUSH PRIVILEGES;"
""")
doc.add_paragraph('Tables are created automatically on first run.')

# ══════════════════════════════════════════════════════════════════════
# 9. SYSTEMD SERVICE
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('9. Systemd Service', level=1)

doc.add_heading('Service File: bike-counter.service', level=2)
add_code("""
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
""")

doc.add_heading('Installing and Enabling', level=2)
add_code("""
sudo cp bike-counter.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable bike-counter.service
sudo systemctl start bike-counter.service
""")

doc.add_heading('Managing the Service', level=2)
add_code("""
sudo systemctl status bike-counter.service    # Check status
sudo systemctl stop bike-counter.service      # Stop
sudo systemctl restart bike-counter.service   # Restart
sudo journalctl -u bike-counter -f            # Follow logs
sudo journalctl -u bike-counter --since today  # Today's logs
""")

doc.add_heading('Disabling the Service', level=2)
add_code("""
sudo systemctl stop bike-counter.service
sudo systemctl disable bike-counter.service
""")

# ══════════════════════════════════════════════════════════════════════
# 10. BACKUP & MAINTENANCE
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('10. Backup & Maintenance', level=1)

doc.add_heading('Backup Script: backup.sh', level=2)
add_code("""
./backup.sh                    # Default: ./backups/
./backup.sh /path/to/backups   # Custom backup directory
""")

doc.add_paragraph('What it does:')
add_bullet('Dumps MySQL database via mysqldump (compressed with gzip)')
add_bullet('Creates tarball of screenshots/ directory')
add_bullet('Deletes backup files older than 30 days')

doc.add_heading('Data Retention', level=2)
doc.add_paragraph('Enforced automatically at startup and on each crossing event:')
add_table(
    ['What', 'Limit', 'Behavior'],
    [
        ['Screenshots', 'Max 1000 files', 'Deletes oldest, keeps 950 newest'],
        ['Crossing events', 'Max 90 days', 'Deletes rows older than 90 days'],
        ['Quarterly counts', 'Max 90 days', 'Deletes rows older than 90 days'],
    ],
    col_widths=[1.8, 1.5, 3.2]
)

doc.add_heading('Log Rotation', level=2)
doc.add_paragraph('For production, add a logrotate config:')
add_code("""
# /etc/logrotate.d/bicycle-counter
/home/student/CounterProject/bicycle_counter.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
""")

doc.add_heading('Database Credentials', level=2)
doc.add_paragraph('Stored in config.yaml under database: section. Default credentials:')
add_bullet('User: bikecounter', bold_prefix='')
add_bullet('Password: BikeCount2026!', bold_prefix='')
add_bullet('Database: bicycle_counter', bold_prefix='')
add_bullet('Host: 127.0.0.1:3306', bold_prefix='')
p = doc.add_paragraph()
run = p.add_run('IMPORTANT: Change these credentials for production deployments.')
run.bold = True
run.font.color.rgb = RGBColor(0xCC, 0x00, 0x00)

# ══════════════════════════════════════════════════════════════════════
# 11. TROUBLESHOOTING
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('11. Troubleshooting', level=1)

doc.add_heading('Camera Issues', level=2)
add_table(
    ['Symptom', 'Fix'],
    [
        ['Cannot open camera / cap.isOpened() False', 'Run fuser -k /dev/video0 then restart'],
        ['Camera overexposed (white in sun)', 'v4l2-ctl -d /dev/video0 --set-ctrl brightness=-40 (range: -64 to 64)'],
        ['VIDEOIO ERROR: V4L2 pixel format not supported', 'Camera does not support MJPEG at 640x480. Try different USB camera.'],
        ['Low frame rate', 'Check camera.fps in config.yaml (default 15)'],
        ['Camera reconnects repeatedly', 'Check USB cable, power supply, or try different USB port'],
    ],
    col_widths=[2.5, 4.0]
)

doc.add_heading('Inference Issues', level=2)
add_table(
    ['Symptom', 'Fix'],
    [
        ['model(frame) hangs or takes > 1s', 'TensorRT engine may be built for different GPU. Rebuild: trtexec --onnx=yolo11n.onnx --saveEngine=yolo11n.engine --fp16'],
        ['CUDA out-of-memory', 'Orin Nano shares 8GB between CPU/GPU. Close other GPU apps.'],
        ['ultralytics not found', 'pip3 install ultralytics with export PIP_BREAK_SYSTEM_PACKAGES=1'],
        ['PyTorch CUDA mismatch warnings', 'Normal - ultralytics bypasses PyTorch for TensorRT. Harmless.'],
    ],
    col_widths=[2.5, 4.0]
)

doc.add_heading('Database Issues', level=2)
add_table(
    ['Symptom', 'Fix'],
    [
        ['Access denied for user bikecounter', 'Check MySQL is running: sudo systemctl status mysql. Verify credentials in config.yaml.'],
        ['Can not connect to MySQL server', 'Ensure MySQL is running: sudo systemctl start mysql. Check host/port in config.'],
        ['Connection pool exhausted', 'Increase database.pool_size in config.yaml (default: 4).'],
        ['Want to reset all data', 'sudo mysql -u root -e "DROP DATABASE bicycle_counter; CREATE DATABASE bicycle_counter;"'],
        ['Want to reset screenshots', 'rm ~/CounterProject/screenshots/*.jpg'],
    ],
    col_widths=[2.5, 4.0]
)

doc.add_heading('Dashboard Issues', level=2)
add_table(
    ['Symptom', 'Fix'],
    [
        ['Port 8080 already in use', 'fuser -k 8080/tcp then restart'],
        ['Dashboard loads but video is black', 'Pipeline crashed. Check terminal for [ERROR] messages.'],
        ['Dashboard shows Pipeline not running', 'Main loop not running. Check Python exceptions.'],
        ['Charts do not update', 'Open browser DevTools -> Console. Check for fetch errors to /stats.'],
        ['Database viewer shows Loading...', 'Check /api/crossings and /api/quarterly return data.'],
    ],
    col_widths=[2.5, 4.0]
)

doc.add_heading('General System Issues', level=2)
add_table(
    ['Symptom', 'Fix'],
    [
        ['ld.so.conf.d errors about DeepStream libs', 'Ensure /etc/ld.so.conf.d/deepstream.conf contains /opt/nvidia/deepstream/deepstream/lib then sudo ldconfig'],
        ['pyds import errors', 'pyds is NOT used in current pipeline. Ignore.'],
        ['System slow after boot', 'Old bike-counter.service may be running. Disable: sudo systemctl stop bike-counter && sudo systemctl disable bike-counter'],
        ['mysql.connector import error', 'pip3 install mysql-connector-python with export PIP_BREAK_SYSTEM_PACKAGES=1'],
    ],
    col_widths=[2.5, 4.0]
)

# ══════════════════════════════════════════════════════════════════════
# 12. BUILD FROM SCRATCH
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('12. Build from Scratch', level=1)

p = doc.add_paragraph()
run = p.add_run('Prerequisites: ')
run.bold = True
p.add_run('Fresh Jetson Orin Nano with JetPack 6.x (L4T 39.x), USB camera, internet access, sudo privileges.')

doc.add_heading('Quick Method (Automated)', level=2)
add_code("""
cd ~/CounterProject
sudo chmod +x setup.sh
sudo ./setup.sh
""")

doc.add_heading('Manual Method (Step by Step)', level=2)

doc.add_heading('Step 1: System Packages', level=3)
add_code("""
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y build-essential cmake git wget pkg-config \\
    python3-dev python3-pip python3-gi python3-gi-cairo \\
    gir1.2-gstreamer-1.0 gir1.2-gst-rtsp-server-1.0 \\
    libgstrtspserver-1.0-dev libgstreamer1.0-dev \\
    libgstreamer-plugins-base1.0-dev \\
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-good \\
    gstreamer1.0-plugins-ugly gstreamer1.0-tools \\
    libcairo2-dev libgirepository1.0-dev
""")

doc.add_heading('Step 2: Python Dependencies', level=3)
add_code("""
export PIP_BREAK_SYSTEM_PACKAGES=1
pip3 install numpy "numpy<2"
pip3 install ultralytics onnx onnxslim
pip3 install fastapi "uvicorn[standard]" jinja2 python-multipart Pillow
pip3 install mysql-connector-python
""")
p = doc.add_paragraph()
run = p.add_run('DO NOT ')
run.bold = True
run.font.color.rgb = RGBColor(0xCC, 0x00, 0x00)
p.add_run('pip install --upgrade pip - the Debian-managed pip must stay intact for dpkg tracking.')

doc.add_heading('Step 3: Verify DeepStream SDK', level=3)
add_code("""
ls -la /opt/nvidia/deepstream/deepstream
# If missing, find and symlink:
# sudo mkdir -p /opt/nvidia/deepstream
# sudo ln -sf /path/to/deepstream-X.Y /opt/nvidia/deepstream/deepstream
""")

doc.add_heading('Step 4: Build TensorRT Engine (if needed)', level=3)
doc.add_paragraph('The project ships a pre-built yolo11n.engine. Only rebuild if moving to a different GPU:')
add_code("""
cd ~/CounterProject
trtexec --onnx=yolo11n.onnx --saveEngine=yolo11n.engine --fp16
""")

doc.add_heading('Step 5: MySQL Setup', level=3)
add_code("""
sudo systemctl start mysql
sudo mysql -u root -e "CREATE DATABASE IF NOT EXISTS bicycle_counter;"
sudo mysql -u root -e "CREATE USER IF NOT EXISTS 'bikecounter'@'127.0.0.1' IDENTIFIED BY 'BikeCount2026!';"
sudo mysql -u root -e "GRANT ALL PRIVILEGES ON bicycle_counter.* TO 'bikecounter'@'127.0.0.1';"
sudo mysql -u root -e "FLUSH PRIVILEGES;"
""")

doc.add_heading('Step 6: Verify Installation', level=3)
add_code("""
cd ~/CounterProject
python3 -c "from ultralytics import YOLO; print('ultralytics OK')"
python3 -c "import cv2; print('OpenCV OK')"
python3 -c "import fastapi; print('FastAPI OK')"
python3 -c "import mysql.connector; print('MySQL connector OK')"
python3 -c "from config_loader import load; print('Config OK:', load()['version'])"
""")

# ══════════════════════════════════════════════════════════════════════
# 13. POSSIBLE UPGRADES
# ══════════════════════════════════════════════════════════════════════
doc.add_heading('13. Possible Upgrades', level=1)

doc.add_heading('Short-term (Feature Additions)', level=2)
add_table(
    ['Upgrade', 'Description', 'Effort'],
    [
        ['Adjustable parameters via dashboard', 'Settings panel to change CROSSING_LINE_Y, MIN_CONFIDENCE, CAMERA_BRIGHTNESS without restart', 'Medium'],
        ['Email/Telegram alerts', 'Notification when threshold reached (e.g. > 100 bicycles/hour)', 'Low'],
        ['Historical date picker', 'View past days data with date navigation in dashboard', 'Medium'],
        ['Screenshot gallery', 'Browse captured screenshots in dashboard with thumbnails', 'Medium'],
        ['WebSocket live events', 'Replace 500ms polling with WebSocket for real-time updates', 'Medium'],
    ],
    col_widths=[2.2, 3.5, 0.8]
)

doc.add_heading('Mid-term (Deployment Hardening)', level=2)
add_table(
    ['Upgrade', 'Description', 'Effort'],
    [
        ['Multi-camera support', 'Multiple instances with different location_name, unified dashboard', 'High'],
        ['HTTPS / reverse proxy', 'nginx with Let\'s Encrypt TLS for secure remote access', 'Medium'],
        ['Docker containerization', 'Package as Docker container with nvidia-docker2 GPU access', 'High'],
        ['OTA updates', 'Over-the-air update mechanism for deployed Jetson devices', 'High'],
    ],
    col_widths=[2.2, 3.5, 0.8]
)

doc.add_heading('Long-term (Advanced Features)', level=2)
add_table(
    ['Upgrade', 'Description', 'Effort'],
    [
        ['Custom YOLO training', 'Fine-tune on bicycle-specific dataset for higher accuracy', 'High'],
        ['Speed estimation', 'Frame-to-frame displacement with camera calibration', 'High'],
        ['Multi-class counting', 'Extend to pedestrians, cars, scooters', 'Medium'],
        ['Edge-cloud sync', 'Sync local MySQL to cloud dashboard (Firebase, AWS IoT)', 'High'],
        ['Analytics API', 'RESTful API for external systems to query historical data', 'Medium'],
        ['GPU utilization monitoring', 'Real-time GPU/CPU/memory stats in dashboard', 'Low'],
    ],
    col_widths=[2.2, 3.5, 0.8]
)

# ══════════════════════════════════════════════════════════════════════
# APPENDIX: QUICK REFERENCE CARD
# ══════════════════════════════════════════════════════════════════════
doc.add_page_break()
doc.add_heading('Appendix: Quick Reference Card', level=1)

doc.add_heading('Start / Stop', level=2)
add_code("""
./run.sh                              # Start manually
sudo systemctl start bike-counter     # Start via systemd
sudo systemctl stop bike-counter      # Stop
""")

doc.add_heading('URLs', level=2)
add_code("""
http://localhost:8080                  # Dashboard
http://localhost:8080/database         # Database viewer
http://localhost:8080/health           # Health check
http://localhost:8080/stats            # Stats JSON
http://localhost:8080/export/csv       # Download 15-min counts CSV
http://localhost:8080/export/detail    # Download all crossings CSV
""")

doc.add_heading('Config', level=2)
add_code("""
nano ~/CounterProject/config.yaml     # Edit config
export BIKE_DASH_PORT=9090            # Override via env var
""")

doc.add_heading('Database', level=2)
add_code("""
mysql -u bikecounter -p'BikeCount2026!' bicycle_counter   # Connect
./backup.sh                           # Backup DB + screenshots
""")

doc.add_heading('Logs', level=2)
add_code("""
tail -f ~/CounterProject/bicycle_counter.log              # Follow log
sudo journalctl -u bike-counter -f                        # Follow systemd log
""")

# ── Save ─────────────────────────────────────────────────────────────
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SPECIFICATION.docx')
doc.save(output_path)
print(f"Saved: {output_path}")
