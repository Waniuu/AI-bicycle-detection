#!/usr/bin/env python3
"""
Bicycle Counter v4.7 — HD Camera Fix + DB Unlocked + NMS Aggressive.
Wymusza kodek MJPG dla rozdzielczości HD i rozwiązuje problem pustych klatek.
"""

import sys
import os
import signal
import time
import logging
import threading
from collections import deque, defaultdict
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# ── Load config ───────────────────────────────────────────────────
from config_loader import load as load_config
try:
    CFG = load_config()
except:
    CFG = {"log_level": "INFO", "camera_width": 1280, "camera_height": 720, "camera_device": "/dev/video0"}

# ── Logging ───────────────────────────────────────────────────────
LOG = logging.getLogger("bike_counter")
LOG.setLevel(logging.INFO)
_fmt = logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
_sh = logging.StreamHandler()
_sh.setFormatter(_fmt)
LOG.addHandler(_sh)

_BASE = os.path.dirname(os.path.abspath(__file__))
SCREENSHOT_DIR = os.path.join(_BASE, "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

_shutdown = threading.Event()

# ---------------------------------------------------------------------------
# Matematyka do zliczania (przecinanie linii)
# ---------------------------------------------------------------------------
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# ---------------------------------------------------------------------------
# Pamięć współdzielona
# ---------------------------------------------------------------------------
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.config = CFG
        self.fps = 0.0
        self.active_tracks = set()
        self.latest_jpeg = None
        self.camera_ok = False
        self.bicycle_count = 0 
        self.scooter_count = 0
        self.model_reload_requested = False
        
        self.calibration_mode = False
        self.sample_recording = False
        self.calib_file_path = None
        self.calib_file_name = "None"
        self.calib_reload_requested = False
        self.calib_paused = False
        self.calib_step = 0
        self.calib_seek_frame = -1
        self.calib_current_frame = 0
        self.calib_total_frames = 0
        
        self._frame_times = deque(maxlen=60)

state = SharedState()

class ThreadedCamera:
    def __init__(self, src=0):
        
       # Wymuszamy wykorzystanie silnika FFMPEG
        self.cap = cv2.VideoCapture("rtsp://admini:licznik26@10.42.0.130:554/stream1")
        
    
        # Minimalizacja bufora dla niskiego opóźnienia
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        final_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        final_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if final_w <= 0 or final_h <= 0:
            LOG.warning(f"Kamera stanowczo odrzuca połączenie! Zablokowała się na: {final_w}x{final_h}")
        else:
            LOG.info(f"Kamera URUCHOMIONA pomyślnie w rozdzielczości: {final_w}x{final_h}")
            
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()

def _handle_signal(signum, frame):
    LOG.info("Shutdown signal received")
    _shutdown.set()

def main():
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    LOG.info("=" * 50)
    LOG.info("Bicycle Counter v4.7 (HD Camera Fix + DB Unlocked + NMS)")
    LOG.info("=" * 50)

    db = None
    location_id = None
    device_id = None
    try:
        from database import Database
        db = Database()
        device_id = db.get_or_create_device(CFG.get("device_name", "OrinNano"), CFG.get("device_desc", ""))
        location_id = db.get_or_create_location(CFG.get("location_name", "TestLocation"), CFG.get("location_desc", ""), device_id=device_id)
        db.cleanup_old_data(CFG.get("max_crossing_days", 30))
        LOG.info(f"MySQL ready (device={device_id}, location={location_id})")
    except Exception as e:
        LOG.error(f"Failed to initialize Database: {e}")

    model_path = "/home/student/CounterProject/SilnikV5/results/runs/yolo11s_licznik/weights/best.engine"
    if not os.path.exists(model_path):
        model_path = model_path.replace(".engine", ".pt")

    LOG.info(f"Loading YOLO model from: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        LOG.error(f"Failed to load model: {e}")
        sys.exit(1)

    cap = ThreadedCamera(0)
    state.camera_ok = cap.isOpened()

    import dashboard_server
    dashboard_server.set_shared_state(state)
    dashboard_server.set_config(CFG)
    if db:
        dashboard_server.set_db(db, location_id, device_id)
        
    web_thread = threading.Thread(target=dashboard_server.run, daemon=True)
    web_thread.start()

    track_history = defaultdict(lambda: deque(maxlen=25))
    counted_tracks = set()
    
    consecutive_failures = 0
    calib_cap = None

    try:
        while not _shutdown.is_set():
            if state.calibration_mode:
                if state.calib_reload_requested or calib_cap is None or not calib_cap.isOpened():
                    state.calib_reload_requested = False
                    if calib_cap is not None:
                        calib_cap.release()
        
                    # WYMUSZENIE ŚCIEŻKI DO PLIKÓW WIDEO
                    if state.calib_file_path:
                        video_path = state.calib_file_path.strip()
                        if not video_path.startswith('/'):
                            video_path = os.path.join("/home/student/CounterProject/calibration_samples", video_path)
                            
                        if os.path.exists(video_path):
                            # FIX: Wymuszamy silnik FFMPEG do odczytu MP4 w systemie Linux
                            calib_cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
                            
                            if calib_cap.isOpened():
                                LOG.info(f"Pomyślnie otwarto wideo testowe: {video_path}")
                            else:
                                LOG.error(f"BŁĄD KODEKA: OpenCV nie potrafi zdekodować pliku {video_path}")
                                state.calibration_mode = False
                        else:
                            LOG.warning(f"Nie znaleziono pliku wideo: {video_path}")
                            state.calibration_mode = False
                    else:
                        state.calibration_mode = False

            if state.calibration_mode and calib_cap and calib_cap.isOpened():
                state.calib_total_frames = int(calib_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                vid_fps = calib_cap.get(cv2.CAP_PROP_FPS) or 30.0

                if state.calib_seek_frame >= 0:
                    calib_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(state.calib_total_frames - 1, state.calib_seek_frame)))
                    state.calib_seek_frame = -1
                    ret, frame = calib_cap.read()
                elif state.calib_step != 0:
                    target_pos = state.calib_current_frame + state.calib_step
                    calib_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(state.calib_total_frames - 1, target_pos)))
                    state.calib_step = 0
                    ret, frame = calib_cap.read()
                elif state.calib_paused:
                    # FIX: Wymuszamy fizyczne odczytanie klatki z WIDEO w miejscu, 
                    # co nadpisuje wszelkie "duchy" z kamery na żywo.
                    current_pos = calib_cap.get(cv2.CAP_PROP_POS_FRAMES)
                    if current_pos > 0:
                        calib_cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
                    ret, frame = calib_cap.read()
                else:
                    ret, frame = calib_cap.read()
                    if not ret:
                        calib_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = calib_cap.read()

                if ret:
                    pos = int(calib_cap.get(cv2.CAP_PROP_POS_FRAMES))
                    state.calib_current_frame = max(0, pos - 1)
                    if not state.calib_paused:
                        time.sleep(1.0 / vid_fps)
            else:
                if calib_cap is not None:
                    calib_cap.release()
                    calib_cap = None
                
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        cap.release()
                        time.sleep(1)
                        cap = ThreadedCamera(0)
                        consecutive_failures = 0
                    continue
                consecutive_failures = 0

            if not ret or frame is None:
                continue

            current_conf = state.config.get("model", {}).get("min_confidence", 0.40)
            use_line = state.config.get("tracker", {}).get("use_crossing_line", True)
            
            line_x1 = int(state.config.get("tracker", {}).get("line_x1", 0))
            line_y1 = int(state.config.get("tracker", {}).get("line_y1", int(frame.shape[0]/2)))
            line_x2 = int(state.config.get("tracker", {}).get("line_x2", frame.shape[1]))
            line_y2 = int(state.config.get("tracker", {}).get("line_y2", int(frame.shape[0]/2)))

            if state.config.get("osd", {}).get("show_line", True):
                cv2.line(frame, (line_x1, line_y1), (line_x2, line_y2), (0, 0, 255), 3)
                cv2.circle(frame, (line_x1, line_y1), 6, (0, 0, 255), -1)
                cv2.circle(frame, (line_x2, line_y2), 6, (0, 0, 255), -1)

            results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=current_conf, iou=0.40)
            
            # NAPRAWA: Definicja zmiennej przed użyciem (zapobiega wybuchom na pustych klatkach)
            active_ids = set()

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                clss = results[0].boxes.cls.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()

                best_detections = {}
                for box, track_id, class_id, conf in zip(boxes, track_ids, clss, confs):
                    if track_id not in best_detections or conf > best_detections[track_id]['conf']:
                        best_detections[track_id] = {'box': box, 'class_id': class_id, 'conf': conf}

                for track_id, det in best_detections.items():
                    box = det['box']
                    class_id = det['class_id']
                    conf = det['conf']
                    
                    raw_class_name = model.names.get(class_id, "unknown").lower()
                    norm_class = None
                    if "scooter" in raw_class_name:
                        norm_class = "e-scooter"
                    elif "bicycle" in raw_class_name or "bike" in raw_class_name:
                        norm_class = "bicycle"

                    if not norm_class:
                        continue
                        
                    active_ids.add(track_id)
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    
                    track = track_history[track_id]
                    track.append((cx, cy))
                    
                    if track_id not in counted_tracks:
                        should_count = False
                        
                        if use_line and len(track) >= 5:
                            A = track[-2] 
                            B = track[-1]
                            C = (line_x1, line_y1)
                            D = (line_x2, line_y2)
                            
                            if segments_intersect(A, B, C, D):
                                should_count = True
                                
                        if should_count:
                            counted_tracks.add(track_id)
                            
                            if norm_class == "bicycle":
                                state.bicycle_count += 1
                                LOG.info(f"COUNTED: CYCLIST #{track_id} | Total: {state.bicycle_count}")
                            elif norm_class == "e-scooter":
                                state.scooter_count += 1
                                LOG.info(f"COUNTED: E-SCOOTER #{track_id} | Total: {state.scooter_count}")
                            
                            if db and location_id:
                                ts_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                                iso_ts = datetime.now().isoformat(timespec="seconds")
                                
                                db.record_crossing(
                                    location_id=location_id,
                                    timestamp_str=iso_ts,
                                    track_id=int(track_id),
                                    device_id=device_id,
                                    vehicle_type=norm_class,
                                    frame=frame,              # PRZEKAZUJEMY KLATKĘ DO BAZY
                                    bbox=(x1, y1, x2, y2)     # I KOORDYNATY Z RAMKĄ
                                )
                                db.cleanup_screenshots()

                    color = (0, 255, 0) if track_id not in counted_tracks else (255, 128, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(frame, (x1, y1 - 18), (x1 + 130, y1), (0, 0, 0), -1)
                    cv2.putText(frame, f"{norm_class} #{track_id} {conf:.0%}", (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                    if len(track) > 1:
                        pts = np.array(track, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], False, (255, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 4, (255, 255, 0), -1)

            for track_id in list(track_history.keys()):
                if track_id not in active_ids:
                    if len(track_history[track_id]) > 0:
                        track_history[track_id].popleft()
                    else:
                        del track_history[track_id]
                        counted_tracks.discard(track_id)

            ret_enc, jpeg_buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ret_enc:
                with state.lock:
                    state.latest_jpeg = jpeg_buf.tobytes()
                    state.active_tracks = active_ids
                    
                    now = time.time()
                    state._frame_times.append(now)
                    if len(state._frame_times) > 1:
                        dt = state._frame_times[-1] - state._frame_times[0]
                        if dt > 0:
                            state.fps = (len(state._frame_times) - 1) / dt

    except Exception as e:
        LOG.exception("Pipeline error")
    finally:
        cap.release()
        if calib_cap is not None:
            calib_cap.release()
        if db:
            db.close()
        LOG.info("Shutdown complete")

if __name__ == "__main__":
    main()