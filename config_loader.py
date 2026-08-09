"""
Configuration loader — reads config.yaml and applies env overrides.
"""

import os
import yaml

_BASE = os.path.dirname(os.path.abspath(__file__))


def _env(key, default):
    val = os.environ.get(key)
    if val is None:
        return default
    if isinstance(default, bool):
        return val.lower() in ("1", "true", "yes")
    if isinstance(default, int):
        return int(val)
    if isinstance(default, float):
        return float(val)
    return val


def load():
    path = os.path.join(_BASE, "config.yaml")
    if os.path.exists(path):
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    d = cfg.get("device", {})
    m = cfg.get("model", {})
    c = cfg.get("camera", {})
    t = cfg.get("tracker", {})
    dd = cfg.get("dedup", {})
    db = cfg.get("database", {})
    ret = cfg.get("retention", {})
    dash = cfg.get("dashboard", {})
    log = cfg.get("logging", {})

    return {
        "version": cfg.get("version", "2.0"),
        "device_name": _env("BIKE_DEVICE_NAME", d.get("name", "jetson-orin-01")),
        "device_desc": _env("BIKE_DEVICE_DESC", d.get("description", "")),
        "location_name": _env("BIKE_LOCATION_NAME", d.get("location_name", "Camera_01")),
        "location_desc": _env("BIKE_LOCATION_DESC", d.get("location_description", "")),
        "camera_device": _env("BIKE_CAMERA_DEVICE", c.get("device", "/dev/video0")),
        "camera_width": _env("BIKE_CAMERA_WIDTH", c.get("width", 640)),
        "camera_height": _env("BIKE_CAMERA_HEIGHT", c.get("height", 480)),
        "camera_fps": _env("BIKE_CAMERA_FPS", c.get("fps", 15)),
        "camera_brightness": _env("BIKE_CAMERA_BRIGHTNESS", c.get("brightness", -30)),
        "model_path": _env("BIKE_MODEL_PATH", os.path.join(_BASE, m.get("path", "yolo11n.engine"))),
        "bicycle_class_id": _env("BIKE_CLASS_ID", m.get("bicycle_class_id", 1)),
        "min_confidence": _env("BIKE_MIN_CONFIDENCE", m.get("min_confidence", 0.45)),
        "track_missing_ttl": _env("BIKE_TRACK_MISSING_TTL", t.get("missing_ttl", 3.0)),
        "track_persist_time": _env("BIKE_TRACK_PERSIST_TIME", t.get("persist_time", 0.6)),
        "track_iou_threshold": _env("BIKE_TRACK_IOU", t.get("iou_threshold", 0.15)),
        "track_max_distance": _env("BIKE_TRACK_MAX_DIST", t.get("max_distance", 150)),
        "dedup_ttl": _env("BIKE_DEDUP_TTL", dd.get("ttl", 5.0)),
        "dedup_distance": _env("BIKE_DEDUP_DIST", dd.get("distance", 120)),
        "db_host": _env("BIKE_DB_HOST", db.get("host", "127.0.0.1")),
        "db_port": _env("BIKE_DB_PORT", db.get("port", 3306)),
        "db_user": _env("BIKE_DB_USER", db.get("user", "bikecounter")),
        "db_pass": _env("BIKE_DB_PASS", db.get("password", "BikeCount2026!")),
        "db_name": _env("BIKE_DB_NAME", db.get("name", "bicycle_counter")),
        "db_pool_size": _env("BIKE_DB_POOL", db.get("pool_size", 4)),
        "max_screenshots": _env("BIKE_MAX_SCREENSHOTS", ret.get("max_screenshots", 1000)),
        "keep_screenshots": _env("BIKE_KEEP_SCREENSHOTS", ret.get("keep_screenshots", 950)),
        "max_crossing_days": _env("BIKE_MAX_CROSSING_DAYS", ret.get("max_crossing_days", 90)),
        "dashboard_host": _env("BIKE_DASH_HOST", dash.get("host", "0.0.0.0")),
        "dashboard_port": _env("BIKE_DASH_PORT", dash.get("port", 8080)),
        "log_level": _env("BIKE_LOG_LEVEL", log.get("level", "INFO")),
        "log_file": _env("BIKE_LOG_FILE", os.path.join(_BASE, log.get("file", "bicycle_counter.log"))),
    }
