#!/usr/bin/env python3
"""
Bicycle Counter v2.0 — Production-grade real-time bicycle counting.
Counts unique bicycles detected on screen. Each bike is counted exactly once.
Serves live video + analytics at http://localhost:8080
"""

import sys
import os
import signal
import time
import io
import logging
import threading
from collections import deque
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

SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")

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
    
    __slots__ = ("id", "cx", "cy", "prev_cy", "bbox", "confidence", "last_seen", "created_at", "counted", "best_frame", "best_score", "class_name")
    
    def __init__(self, tid, cx, cy, bbox, confidence):
        self.id = tid
        self.cx = cx
        self.cy = cy
        self.prev_cy = cy  
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = time.time()
        self.created_at = time.time()
        self.counted = False
        self.best_frame = None  
        self.best_score = 0.0   
        self.class_name="vehicle"

        
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
                
                matched.append({
                    "track_id": track.id, "cx": track.cx, "cy": track.cy,
                    "prev_cy": track.prev_cy, "bbox": det["bbox"], "bbox_xyxy": det["bbox_xyxy"],
                    "confidence": track.confidence, "counted": track.counted,
                    "class_id": det.get("class_id"), "class_name": det.get("class_name"),
                    "track_obj": track
                })

        # 2. Rejestracja nowych pojazdów
        for j, det in enumerate(detections):
            if j in used:
                continue
            t = Track(self._next_id, det["cx"], det["cy"], det["bbox_xyxy"], det["confidence"])
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
        self.frame_count = 0
        self.bicycle_count = 0
        self.fps = 0.0
        self.active_tracks = set()
        self.total_detections = 0
        self.confidence_values = deque(maxlen=2000)
        self.latest_jpeg = None
        self._frame_times = deque(maxlen=60)
        self.counted_positions = []
        self.start_time = time.time()
        self.camera_ok = False

state = SharedState()


def draw_osd(img_pil, detections, total_count):
    draw = ImageDraw.Draw(img_pil)

    tracker_cfg = CFG.get("tracker", {})
    if tracker_cfg.get("use_crossing_line", False):
        line_y = tracker_cfg.get("crossing_line_y", CFG.get("crossing_line_y", 240))
        width, _ = img_pil.size
        draw.line([(0, line_y), (width, line_y)], fill="red", width=3)


    draw.text((10, 10), f"Vehicles: {total_count}", fill="yellow")
    for det in detections:
        x, y, w, h = det["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline="lime", width=2)
        draw.text((x, y - 14), f"{det['confidence']:.0%}", fill="white")
    return img_pil


# ---------------------------------------------------------------------------
# Camera reconnect helper
# ---------------------------------------------------------------------------
def open_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG["camera_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG["camera_height"])
    cap.set(cv2.CAP_PROP_FPS, CFG["camera_fps"])
    os.system(f"v4l2-ctl -d {CFG['camera_device']} --set-ctrl brightness={CFG['camera_brightness']} 2>/dev/null")
    return cap


# ---------------------------------------------------------------------------
# Signal handler
# ---------------------------------------------------------------------------
def _handle_signal(signum, frame):
    LOG.info("Shutdown signal received")
    _shutdown.set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    LOG.info("=" * 50)
    LOG.info("Bicycle Counter v%s starting", CFG["version"])
    LOG.info("=" * 50)
    LOG.info("Camera: %dx%d @ %d fps (%s)", CFG["camera_width"], CFG["camera_height"], CFG["camera_fps"], CFG["camera_device"])
    LOG.info("Model: %s", CFG["model_path"])
    LOG.info("Device: %s | Location: %s", CFG["device_name"], CFG["location_name"])

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    model = YOLO(CFG["model_path"])
    LOG.info("YOLO model loaded")

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
    dashboard_server.set_shared_state(state)
    dashboard_server.set_db(db, location_id, device_id)
    dashboard_server.set_config(CFG)
    web_thread = threading.Thread(target=dashboard_server.run, daemon=True)
    web_thread.start()
    LOG.info("Dashboard: http://localhost:%d", CFG["dashboard_port"])

    tracker = Tracker()
    consecutive_failures = 0
    LOG.info("Processing frames...")

    try:
        while not _shutdown.is_set():
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
            current_conf = CFG.get("model", {}).get("min_confidence", CFG.get("min_confidence", 0.45))
            results = model(frame, verbose=False, conf=current_conf)#wykryte obiekty przez model
            raw_dets = []

            model_cfg = CFG.get("model", {})
            target_classes = model_cfg.get("target_classes", {0: "e-scooter", 1: "bicycle"})

            
            for r in results: # ta pętla sprawdza wszystkie obiekty wykryte przez model
                for box in r.boxes:
                    class_id = int(box.cls[0]) #pobranie numeru obiektu(0:e-scooter,1:bike) i zamiana na int
                    

                    # Pobieramy nazwę klasy, sprawdzając int oraz str (zabezpieczenie przed YAML)
                    class_name = target_classes.get(class_id) or target_classes.get(str(class_id))

                    # Jeśli wykryty obiekt nie jest ani rowerem, ani hulajnogą – pomijamy
                    if class_name is None:
                        continue

                    conf = float(box.conf[0]) # aktualna pewność AI
                    bx1, by1, bx2, by2 = map(float, box.xyxy[0])

                    raw_dets.append({ #słownik
                        "cx": (bx1 + bx2) / 2.0, 
                        "cy": (by1 + by2) / 2.0,
                        "bbox": (bx1, by1, bx2 - bx1, by2 - by1),
                        "bbox_xyxy": (bx1, by1, bx2, by2), 
                        "confidence": conf,
                        "class_id": class_id,         # DODANE: id klasy (0 lub 1)
                        "class_name": class_name      # DODANE: czytelna nazwa ('e-scooter' / 'bicycle')
                    })

                    state.total_detections += 1
                    state.confidence_values.append(conf)

            detections = tracker.update(raw_dets) # powiedzenie trackerowi jaki obiekt jest śledzony
            active_ids = set()
            new_vehicles = []
            now_check = time.time()

            line_y = CFG.get("tracker", {}).get("crossing_line_y", CFG.get("crossing_line_y", 240))
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            use_line = CFG.get("tracker", {}).get("use_crossing_line", False)

            
            #rysowane są ramki, wybierane zdjęcia, sprawdzanie przekroczenia linii i rejestracja nowych pojazdów
            for det in detections:
                # detections to lista wykrytych obiektów 
                tid = det["track_id"] # pobranie identyfikatora pojazdu z trackera
                active_ids.add(tid)# dodanie identyfikatora do zbioru aktywnych obiektów

                track_obj = det.get("track_obj") # pobranie teczki pojazdu
                if not track_obj:
                    continue

                class_name = det.get("class_name","vehicle")
                track_obj.class_name = class_name
                bx1, by1, bx2, by2 = det["bbox_xyxy"]
                w, h = bx2 - bx1, by2 - by1

               
                current_score = det["confidence"] * (w * h)
                if current_score > track_obj.best_score:
                    track_obj.best_score = current_score

                  
                    frame_with_box = img_pil.copy()
                    draw_box = ImageDraw.Draw(frame_with_box)
                    draw_box.rectangle([bx1, by1, bx2, by2], outline="lime", width=3)
                    draw_box.rectangle([bx1, by1 - 18, bx1 + 80, by1], fill="black")
                    label_text = f"{class_name} #{tid} {det['confidence']:.0%}"
                    draw_box.text((bx1 + 2, by1 - 16), label_text, fill="lime")
                    
                    track_obj.best_frame = frame_with_box

           
                if not track_obj.counted:
                    should_count = False

                    if use_line:
                        was_above_or_near = track_obj.prev_cy < (line_y + 50)
                        is_now_at_or_below = det["cy"] >= line_y
                        if was_above_or_near and is_now_at_or_below:
                            should_count = True
                    else:
                       
                        should_count = True

                    if should_count:
                        track_obj.counted = True
                        state.bicycle_count += 1

                        new_vehicles.append({
                            "track_id": tid,
                            "vehicle_type": class_name,
                            "image": track_obj.best_frame or img_pil
                        })
                        LOG.info("NEW VEHICLE (%s) #%d detected | Total: %d", class_name.upper(), tid, state.bicycle_count)

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

          
            img_pil = draw_osd(img_pil, detections, state.bicycle_count)
            jpeg_buf = io.BytesIO()
            img_pil.save(jpeg_buf, format="JPEG", quality=75)
            with state.lock:
                state.latest_jpeg = jpeg_buf.getvalue()

            for det in new_vehicles:
                v_type = det.get("vehicle_type", "bike")
                ts_str = time.strftime("%H-%M-%S")
                fname = f"{v_type}_{ts_str}_id{det['track_id']}.jpg"
                
                
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
