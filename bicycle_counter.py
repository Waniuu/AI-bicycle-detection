#!/usr/bin/env python3
"""
Bicycle Counter v3.0 — Production-grade real-time bicycle & scooter counting.
Counts unique bicycles and scooters detected on screen. Each vehicle is counted exactly once.
Serves live video + analytics at http://localhost:8080
"""

import sys
import os
import signal
import time
import io
import logging
import threading
import yaml
from collections import deque, Counter
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

# ── Load config ───────────────────────────────────────────────────
from config_loader import load as load_config
CFG = load_config()

# ── Logging ───────────────────────────────────────────────────────
LOG = logging.getLogger("bike_counter")
LOG.setLevel(getattr(logging, CFG["log_level"]))
_fmt = logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
_sh = logging.StreamHandler()
_sh.setFormatter(_fmt)
LOG.addHandler(_sh)
if CFG["log_file"]:
    _fh = logging.FileHandler(CFG["log_file"])
    _fh.setFormatter(_fmt)
    LOG.addHandler(_fh)

_BASE = os.path.dirname(os.path.abspath(__file__))
SCREENSHOT_DIR = os.path.join(_BASE, "screenshots")
SAMPLES_DIR = os.path.join(_BASE, "calibration_samples")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# ── Globals ───────────────────────────────────────────────────────
_shutdown = threading.Event()


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

def compute_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Track:
    __slots__ = ("id", "cx", "cy", "prev_cy", "start_cx", "start_cy", "bbox", "confidence", "last_seen", "created_at", "counted", "best_frame", "best_score", "class_name", "class_history", "history_centers")
    
    def __init__(self, tid, cx, cy, bbox, confidence, initial_class="vehicle"):
        self.id = tid
        self.cx = cx
        self.cy = cy
        self.prev_cy = cy
        self.start_cx = cx
        self.start_cy = cy
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = time.time()
        self.created_at = time.time()
        self.counted = False
        self.best_frame = None
        self.best_score = 0.0
        self.class_name = initial_class
        self.class_history = deque(maxlen=15)
        self.class_history.append(initial_class)
        self.history_centers = deque(maxlen=20)
        self.history_centers.append((cx, cy))

    def add_class_observation(self, cls_name):
        self.class_history.append(cls_name)

    def get_voted_class(self, min_ratio=0.50):
        if len(self.class_history) < 3:
            return self.class_name
        counts = Counter(self.class_history)
        most_common, count = counts.most_common(1)[0]
        ratio = count / len(self.class_history)
        if ratio >= min_ratio:
            self.class_name = most_common
        return self.class_name

    def get_total_displacement(self):
        return ((self.cx - self.start_cx)**2 + (self.cy - self.start_cy)**2)**0.5


class Tracker:
    def __init__(self):
        self.tracks = []
        self._next_id = 1

    def update(self, detections):
        now = time.time()
        self.tracks = [t for t in self.tracks if now - t.last_seen < CFG["track_missing_ttl"]]
        matched = []
        used = set()

        for track in self.tracks:
            best_score, best_j = -1.0, -1
            for j, det in enumerate(detections):
                if j in used:
                    continue
                iou = compute_iou(track.bbox, det["bbox_xyxy"])
                dist = ((track.cx - det["cx"])**2 + (track.cy - det["cy"])**2) ** 0.5
                if iou >= CFG["track_iou_threshold"]:
                    score = 1.0 + iou
                elif dist < CFG["track_max_distance"]:
                    score = 1.0 - (dist / CFG["track_max_distance"])
                else:
                    score = -1.0
                if score > best_score:
                    best_score, best_j = score, j

            if best_score > 0 and best_j >= 0:
                det = detections[best_j]
                used.add(best_j)
                
                track.prev_cy = track.cy
                track.cx, track.cy = det["cx"], det["cy"]
                track.bbox = det["bbox_xyxy"]
                track.confidence = det["confidence"]
                track.last_seen = now
                track.history_centers.append((track.cx, track.cy))
                track.add_class_observation(det.get("class_name", "vehicle"))
                
                matched.append({
                    "track_id": track.id, "cx": track.cx, "cy": track.cy,
                    "prev_cy": track.prev_cy, "bbox": det["bbox"], "bbox_xyxy": det["bbox_xyxy"],
                    "confidence": track.confidence, "counted": track.counted,
                    "class_id": det.get("class_id"), "class_name": det.get("class_name"),
                    "track_obj": track
                })

        for j, det in enumerate(detections):
            if j in used:
                continue
            t = Track(self._next_id, det["cx"], det["cy"], det["bbox_xyxy"], det["confidence"], initial_class=det.get("class_name", "vehicle"))
            self._next_id += 1
            self.tracks.append(t)
            matched.append({
                "track_id": t.id, "cx": t.cx, "cy": t.cy,
                "prev_cy": t.prev_cy, "bbox": det["bbox"], "bbox_xyxy": det["bbox_xyxy"],
                "confidence": t.confidence, "counted": False,
                "class_id": det.get("class_id"), "class_name": det.get("class_name"),
                "track_obj": t
            })

        return matched


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.config = CFG
        if "osd" not in self.config:
            self.config["osd"] = {
                "show_line": True,
                "show_boxes": True,
                "show_tracks": True,
                "show_persons": False,
                "show_hud": True
            }
        self.frame_count = 0
        self.bicycle_count = 0
        self.scooter_count = 0
        self.fps = 0.0
        self.active_tracks = set()
        self.total_detections = 0
        self.confidence_values = deque(maxlen=2000)
        self.latest_jpeg = None
        self._frame_times = deque(maxlen=60)
        self.counted_positions = []
        self.start_time = time.time()
        self.camera_ok = False
        
        # Flagi dynamicznej zmiany modelu
        self.model_reload_requested = False
        
        # Kalibracja & pliki
        self.record_sample_requested = False
        self.sample_recording = False
        self.calibration_mode = False
        self.calib_file_path = None
        self.calib_file_name = None
        self.calib_reload_requested = False
        self.calib_paused = False
        self.calib_step = 0
        self.calib_seek_frame = -1
        self.calib_current_frame = 0
        self.calib_total_frames = 0

state = SharedState()


def resolve_model_file(requested_path):
    """Priorytetyzuje plik .engine, a w razie braku zwraca .pt"""
    if requested_path.endswith(".pt"):
        engine_path = requested_path.rsplit(".pt", 1)[0] + ".engine"
        if os.path.exists(engine_path):
            return engine_path
    elif requested_path.endswith(".engine"):
        if os.path.exists(requested_path):
            return requested_path
        pt_path = requested_path.rsplit(".engine", 1)[0] + ".pt"
        if os.path.exists(pt_path):
            LOG.warning("TensorRT .engine not found, falling back to: %s", pt_path)
            return pt_path
    return requested_path


def get_model_definition():
    config_path = os.path.join(_BASE, "config.yaml")
    cfg_raw = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg_raw = yaml.safe_load(f) or {}
        except Exception:
            pass

    model_sec = cfg_raw.get("model", {})
    available = model_sec.get("available_models", {})
    active_key = model_sec.get("active_model", "yolo11s_scooter")

    if not available:
        available = {
            "yolo11s_scooter": {
                "name": "YOLO11s (Rowery + Hulajnogi)",
                "path": "/home/student/CounterProject/Silnik_hul_bikev4/results/runs/yolo11s_licznik/weights/best.engine",
                "supports_scooters": True,
                "bike_class_name": "cyclist",
                "scooter_class_name": "e-scooter"
            },
            "yolo11n_base": {
                "name": "YOLO11n (Tylko rowery)",
                "path": "/home/student/CounterProject/Silnik V2/yolo11n.engine",
                "supports_scooters": False,
                "bike_class_name": "bicycle",
                "scooter_class_name": None
            }
        }

    if active_key in available:
        info = available[active_key]
    else:
        active_key = next(iter(available))
        info = available[active_key]

    resolved_path = resolve_model_file(info["path"])
    return active_key, info, resolved_path


def draw_osd(img_pil, detections, total_count, person_detections=[]):
    draw = ImageDraw.Draw(img_pil)
    osd_cfg = state.config.get("osd", {})
    tracker_cfg = state.config.get("tracker", {})

    if osd_cfg.get("show_line", True):
        line_y = tracker_cfg.get("crossing_line_y", 240)
        line_x1 = tracker_cfg.get("line_x1", 0)
        line_x2 = tracker_cfg.get("line_x2", img_pil.size[0])
        
        draw.line([(line_x1, line_y), (line_x2, line_y)], fill="red", width=3)
        for x in (line_x1, line_x2):
            draw.line([(x - 5, line_y), (x + 5, line_y)], fill="red", width=2)
            draw.line([(x, line_y - 5), (x, line_y + 5)], fill="red", width=2)

    if osd_cfg.get("show_hud", True):
        draw.text((10, 10), f"Vehicles: {total_count}", fill="yellow")
        if getattr(state, "calibration_mode", False):
            draw.text((10, 25), f"[CALIBRATION: {state.calib_file_name or 'loop'}]", fill="yellow")

    if osd_cfg.get("show_persons", False) and person_detections:
        for p in person_detections:
            px1, py1, px2, py2 = p["bbox_xyxy"]
            draw.rectangle([px1, py1, px2, py2], outline="gray", width=1)
            draw.text((px1, py1 - 12), f"person [ignored] {p['confidence']:.0%}", fill="gray")

    for det in detections:
        x, y, w, h = det["bbox"]
        tid = det["track_id"]
        v_type = det.get("class_name", "vehicle")
        conf = det.get("confidence", 0.0)

        if osd_cfg.get("show_boxes", True):
            draw.rectangle([x, y, x + w, y + h], outline="lime", width=2)
            draw.text((x, y - 14), f"{v_type} #{tid} {conf:.0%}", fill="white")

        if osd_cfg.get("show_tracks", True):
            track_obj = det.get("track_obj")
            if track_obj and hasattr(track_obj, "history_centers"):
                pts = list(track_obj.history_centers)
                if len(pts) > 1:
                    draw.line(pts, fill="cyan", width=2)
                cx, cy = det["cx"], det["cy"]
                draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill="cyan", outline="white")

    return img_pil


def open_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG["camera_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG["camera_height"])
    cap.set(cv2.CAP_PROP_FPS, CFG["camera_fps"])
    os.system(f"v4l2-ctl -d {CFG['camera_device']} --set-ctrl brightness={CFG['camera_brightness']} 2>/dev/null")
    return cap


def _handle_signal(signum, frame):
    LOG.info("Shutdown signal received")
    _shutdown.set()


def main():
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    LOG.info("=" * 50)
    LOG.info("Bicycle Counter v%s starting", CFG.get("version", "3.0"))
    LOG.info("=" * 50)

    # ── Odczyt i ładowanie modelu ───────────────────────────────────
    active_key, model_info, model_path = get_model_definition()
    LOG.info("Active model key: %s", active_key)
    LOG.info("Loading YOLO model from: %s", model_path)
    model = YOLO(model_path)
    LOG.info("YOLO model loaded successfully (TensorRT=%s)", str(model_path.endswith('.engine')))

    from database import Database
    db = Database()
    device_id = db.get_or_create_device(CFG["device_name"], CFG["device_desc"])
    location_id = db.get_or_create_location(CFG["location_name"], CFG["location_desc"], device_id=device_id)
    db.cleanup_old_data(CFG["max_crossing_days"])
    LOG.info("MySQL ready (device=%d, location=%d)", device_id, location_id)

    cap = open_camera()
    if not cap.isOpened():
        LOG.error("Cannot open camera")
        sys.exit(1)
    state.camera_ok = True
    LOG.info("Camera opened: %dx%d", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    import dashboard_server
    LOG.info("Loading dashboard module: dashboard_server.py")
    dashboard_server.set_shared_state(state)
    dashboard_server.set_db(db, location_id, device_id)
    dashboard_server.set_config(CFG)
    web_thread = threading.Thread(target=dashboard_server.run, daemon=True)
    web_thread.start()
    LOG.info("Dashboard: http://localhost:%d", CFG["dashboard_port"])

    tracker = Tracker()
    consecutive_failures = 0
    LOG.info("Processing frames...")

    calib_cap = None
    rec_writer = None
    rec_frames_left = 0

    try:
        while not _shutdown.is_set():
            # Obsługa dynamicznej zmiany silnika w locie
            if state.model_reload_requested:
                with state.lock:
                    state.model_reload_requested = False
                    active_key, model_info, model_path = get_model_definition()

                LOG.info("Reloading model dynamically to: %s", model_path)
                try:
                    model = YOLO(model_path)
                    LOG.info("New model loaded successfully. Supports scooters: %s", model_info.get("supports_scooters"))
                except Exception as e:
                    LOG.error("Failed to reload model: %s", str(e))

            if state.calibration_mode:
                if state.calib_reload_requested or calib_cap is None or not calib_cap.isOpened():
                    state.calib_reload_requested = False
                    if calib_cap is not None:
                        calib_cap.release()
                    
                    if not state.calib_file_path or not os.path.exists(state.calib_file_path):
                        sample_files = [os.path.join(SAMPLES_DIR, f) for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                        if sample_files:
                            sample_files.sort(key=os.path.getmtime, reverse=True)
                            state.calib_file_path = sample_files[0]
                            state.calib_file_name = os.path.basename(sample_files[0])
                    
                    if state.calib_file_path and os.path.exists(state.calib_file_path):
                        calib_cap = cv2.VideoCapture(state.calib_file_path)
                        LOG.info("Loaded calibration video: %s", state.calib_file_path)
                    else:
                        LOG.warning("No calibration video found in %s, falling back to camera", SAMPLES_DIR)
                        state.calibration_mode = False
            else:
                if calib_cap is not None:
                    calib_cap.release()
                    calib_cap = None

            if state.calibration_mode and calib_cap and calib_cap.isOpened():
                state.calib_total_frames = int(calib_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                vid_fps = calib_cap.get(cv2.CAP_PROP_FPS) or 15.0
                time.sleep(1.0 / vid_fps)

                if state.calib_seek_frame >= 0:
                    calib_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(state.calib_total_frames - 1, state.calib_seek_frame)))
                    state.calib_seek_frame = -1
                    ret, frame = calib_cap.read()
                    if not ret:
                        calib_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = calib_cap.read()

                elif state.calib_step != 0:
                    target_pos = state.calib_current_frame + state.calib_step
                    target_pos = max(0, min(state.calib_total_frames - 1, target_pos))
                    calib_cap.set(cv2.CAP_PROP_POS_FRAMES, target_pos)
                    state.calib_step = 0
                    ret, frame = calib_cap.read()
                    if not ret:
                        calib_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = calib_cap.read()

                elif state.calib_paused:
                    if 'frame' not in locals() or frame is None:
                        ret, frame = calib_cap.read()
                        if not ret:
                            calib_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = calib_cap.read()
                    else:
                        ret = True
                else:
                    ret, frame = calib_cap.read()
                    if not ret:
                        calib_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = calib_cap.read()

                if ret:
                    pos = int(calib_cap.get(cv2.CAP_PROP_POS_FRAMES))
                    state.calib_current_frame = max(0, pos - 1)
            else:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        LOG.warning("Camera read failed %d times, reconnecting...", consecutive_failures)
                        cap.release()
                        time.sleep(1)
                        cap = open_camera()
                        state.camera_ok = cap.isOpened()
                        consecutive_failures = 0
                        if not state.camera_ok:
                            LOG.error("Camera reconnect failed, retrying in 5s...")
                            time.sleep(5)
                        else:
                            LOG.info("Camera reconnected")
                    continue

                consecutive_failures = 0
                state.camera_ok = True

            if state.record_sample_requested:
                state.record_sample_requested = False
                state.sample_recording = True
                rec_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 15
                rec_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame.shape[1]
                rec_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0]
                
                timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                sample_file_name = f"sample_{timestamp_str}.mp4"
                sample_file_path = os.path.join(SAMPLES_DIR, sample_file_name)
                
                rec_writer = cv2.VideoWriter(sample_file_path, cv2.VideoWriter_fourcc(*'mp4v'), rec_fps, (rec_w, rec_h))
                rec_frames_left = 10 * rec_fps
                LOG.info("Recording 10s sample to %s (%d frames)...", sample_file_path, rec_frames_left)

            if state.sample_recording:
                if rec_writer and rec_writer.isOpened():
                    rec_writer.write(frame)
                    rec_frames_left -= 1
                    if rec_frames_left <= 0:
                        rec_writer.release()
                        rec_writer = None
                        state.sample_recording = False
                        LOG.info("Sample saved: %s", sample_file_name)
                        with state.lock:
                            state.calib_file_path = sample_file_path
                            state.calib_file_name = sample_file_name
                            state.calib_reload_requested = True

            current_conf = state.config.get("model", {}).get("min_confidence", 0.40)
            results = model(frame, verbose=False, conf=current_conf)

            raw_dets = []
            person_detections_in_frame = []

            target_bike_cls = model_info.get("bike_class_name", "cyclist")
            target_scooter_cls = model_info.get("scooter_class_name", "e-scooter")

            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bbox_xyxy = box.xyxy[0].cpu().numpy()
                    bx1, by1, bx2, by2 = bbox_xyxy
                    raw_class_name = model.names.get(class_id, "unknown")

                    if raw_class_name == "person":
                        person_detections_in_frame.append({"bbox_xyxy": bbox_xyxy, "confidence": conf})
                        continue

                    normalized_class_name = None
                    if raw_class_name == target_bike_cls:
                        normalized_class_name = "cyclist"
                    elif target_scooter_cls and raw_class_name == target_scooter_cls:
                        normalized_class_name = "e-scooter"

                    if not normalized_class_name:
                        continue

                    raw_dets.append({
                        "cx": (bx1 + bx2) / 2.0,
                        "cy": (by1 + by2) / 2.0,
                        "bbox": (bx1, by1, bx2 - bx1, by2 - by1),
                        "bbox_xyxy": bbox_xyxy,
                        "confidence": conf,
                        "class_id": class_id,
                        "class_name": normalized_class_name
                    })

                    state.total_detections += 1
                    state.confidence_values.append(conf)

            detections = tracker.update(raw_dets)
            active_ids = set()
            new_vehicles = []
            now_check = time.time()

            line_y = state.config.get("tracker", {}).get("crossing_line_y", 240)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            current_tracker_cfg = state.config.get("tracker", {})
            use_line = current_tracker_cfg.get("use_crossing_line", True)

            for det in detections:
                tid = det["track_id"]
                active_ids.add(tid)

                track_obj = det.get("track_obj")
                if not track_obj:
                    continue

                voted_class = track_obj.get_voted_class(min_ratio=0.50)
                bx1, by1, bx2, by2 = det["bbox_xyxy"]
                w, h = bx2 - bx1, by2 - by1
                current_score = det["confidence"] * (w * h)

                if current_score > track_obj.best_score:
                    track_obj.best_score = current_score
                    frame_with_box = img_pil.copy()
                    draw_box = ImageDraw.Draw(frame_with_box)
                    draw_box.rectangle([bx1, by1, bx2, by2], outline="lime", width=3)
                    draw_box.rectangle([bx1, by1 - 18, bx1 + 80, by1], fill="black")
                    label_text = f"{voted_class} #{tid} {det['confidence']:.0%}"
                    draw_box.text((bx1 + 2, by1 - 16), label_text, fill="lime")
                    track_obj.best_frame = frame_with_box

                class_name = voted_class

                filters_cfg = state.config.get("filters", {})
                min_w = filters_cfg.get("min_width", 25)
                min_h = filters_cfg.get("min_height", 25)
                if w < min_w or h < min_h:
                    continue

                cam_w = state.config.get("camera_width", 640)
                cam_h = state.config.get("camera_height", 480)
                max_w = filters_cfg.get("max_width", cam_w + 1)
                max_h = filters_cfg.get("max_height", cam_h + 1)
                if w > max_w or h > max_h:
                    continue

                min_ratio = filters_cfg.get("min_aspect_ratio", 0.0)
                if min_ratio > 0.0 and (w / float(h)) < min_ratio:
                    continue

                track_persist = state.config.get("tracker", {}).get("track_persist_time", 0.3)
                if now_check - track_obj.created_at < track_persist:
                    continue

                if not track_obj.counted:
                    should_count = False

                    min_move = filters_cfg.get("min_movement_px", 0)
                    if min_move > 0 and track_obj.get_total_displacement() < min_move:
                        continue

                    if use_line:
                        line_x1 = current_tracker_cfg.get("line_x1", 0)
                        line_x2 = current_tracker_cfg.get("line_x2", 9999)
                        was_above_or_near = track_obj.prev_cy < (line_y + 50)
                        is_now_at_or_below = det["cy"] >= line_y
                        within_x_bounds = line_x1 <= det["cx"] <= line_x2
                        if was_above_or_near and is_now_at_or_below and within_x_bounds:
                            should_count = True
                    else:
                        should_count = True

                    if should_count:
                        track_obj.counted = True
                        if not state.calibration_mode:
                            if class_name == "cyclist":
                                state.bicycle_count += 1
                                LOG.info("NEW CYCLIST #%d detected | Total: %d", tid, state.bicycle_count)
                            elif class_name == "e-scooter":
                                state.scooter_count += 1
                                LOG.info("NEW E-SCOOTER #%d detected | Total: %d", tid, state.scooter_count)
                            else:
                                LOG.warning("Counted unknown vehicle type: %s", class_name)

                            new_vehicles.append({
                                "track_id": tid,
                                "vehicle_type": class_name,
                                "image": track_obj.best_frame or img_pil
                            })
                        else:
                            LOG.info("CALIBRATION MODE: Suppressed vehicle #%d (%s)", tid, class_name)

            state.counted_positions = [(x, y, t) for x, y, t in state.counted_positions if now_check - t < CFG["dedup_ttl"]]

            with state.lock:
                state.active_tracks = active_ids
                state.frame_count += 1
                now = time.time()
                state._frame_times.append(now)
                if len(state._frame_times) > 1:
                    dt = state._frame_times[-1] - state._frame_times[0]
                    if dt > 0:
                        state.fps = (len(state._frame_times) - 1) / dt

            img_pil = draw_osd(img_pil, detections, state.bicycle_count, person_detections_in_frame)
            jpeg_buf = io.BytesIO()
            img_pil.save(jpeg_buf, format="JPEG", quality=75)
            with state.lock:
                state.latest_jpeg = jpeg_buf.getvalue()

            for det in new_vehicles:
                v_type = det.get("vehicle_type", "bike")
                ts_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                fname = f"{ts_str}_{v_type}_id{det['track_id']}.jpg"
                save_img = det["image"]
                save_img.save(os.path.join(SCREENSHOT_DIR, fname), format="JPEG", quality=90)
                iso_ts = datetime.now().isoformat(timespec="seconds")
                db.record_crossing(
                    location_id=location_id,
                    timestamp_str=iso_ts,
                    track_id=int(det["track_id"]),
                    device_id=device_id,
                    vehicle_type=v_type
                )
                db.cleanup_screenshots()
    except Exception:
        LOG.exception("Pipeline error")
    finally:
        LOG.info("Final bicycle count: %d", state.bicycle_count)
        cap.release()
        db.close()
        LOG.info("Shutdown complete")


if __name__ == "__main__":
    main()