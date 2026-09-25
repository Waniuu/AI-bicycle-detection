"""
MySQL database module for bicycle counter v5.0.
Uses Background Worker Queue to prevent YOLO pipeline blocking.
"""

import os
import time
import logging
import threading
import queue
from datetime import datetime, timedelta
import cv2

import mysql.connector
from mysql.connector import pooling

LOG = logging.getLogger("bike_counter")

CSV_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOT_DIR = os.path.join(CSV_DIR, "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

class Database:
    def __init__(self, cfg=None):
        if cfg is None:
            from config_loader import load
            cfg = load()
        
        self._cfg = cfg
        self.pool = pooling.MySQLConnectionPool(
            pool_name="bike_pool",
            pool_size=cfg["db_pool_size"],
            pool_reset_session=True,
            host=cfg["db_host"],
            port=cfg["db_port"],
            user=cfg["db_user"],
            password=cfg["db_pass"],
            database=cfg["db_name"],
            charset="utf8mb4",
            autocommit=True,
        )
        self._create_tables_if_not_exist()

        # --- SYSTEM KOLEJEK (WORKER THREAD) ---
        self.task_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        LOG.info("Asynchronous Database Worker thread started.")

    def _worker_loop(self):
        """Wątek tła: konsumuje zadania z kolejki, uwalniając wątek YOLO od zapisów I/O"""
        while True:
            task = self.task_queue.get()
            if task is None: 
                break # Sygnał zamknięcia
            
            try:
                if task["type"] == "crossing":
                    self._process_crossing(task)
                elif task["type"] == "cleanup":
                    self._process_cleanup()
            except Exception as e:
                LOG.error(f"Background worker error: {e}")
            finally:
                self.task_queue.task_done()

    def _process_crossing(self, t):
        """Fizyczny zapis do bazy i na dysk - wykonywany asynchronicznie"""
        conn = self._conn()
        try:
            cur = conn.cursor()
            dt = datetime.fromisoformat(t["timestamp_str"])
            minute_bucket = (dt.minute // 15) * 15
            bucket_time = f"{dt.hour:02d}:{minute_bucket:02d}:00"
            
            is_bike = 1 if t["vehicle_type"] in ('cyclist', 'bicycle') else 0
            is_scooter = 1 if t["vehicle_type"] == 'e-scooter' else 0

            # Zapis do tabeli crossings
            cur.execute(
                "INSERT INTO crossings (device_id, location_id, recorded_at, track_id, vehicle_type) VALUES (%s, %s, %s, %s, %s)",
                (t["device_id"], t["location_id"], dt, t["track_id"], t["vehicle_type"]),
            )
            # Agregacja co 15 minut
            cur.execute(
                """INSERT INTO co_15_minut
                   (location_id, date, czas, il_row, il_hul)
                   VALUES ((SELECT name FROM locations WHERE id = %s), %s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE 
                   il_row = il_row + VALUES(il_row), 
                   il_hul = il_hul + VALUES(il_hul),
                   synced_at = NULL""",
                (t["location_id"], dt.date(), bucket_time, is_bike, is_scooter),
            )
            conn.commit()

            # ZAPIS ZDJĘCIA (I/O Operation przesunięta z YOLO do tego wątku)
            if t["frame"] is not None:
                fname = f"{dt.strftime('%Y-%m-%d_%H-%M-%S')}_{t['vehicle_type']}_id{t['track_id']}.jpg"
                save_img = t["frame"].copy()
                x1, y1, x2, y2 = t["bbox"]
                cv2.rectangle(save_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(save_img, f"{t['vehicle_type']} #{t['track_id']}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(SCREENSHOT_DIR, fname), save_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                
        except Exception as e:
            LOG.error(f"DB Insert failed: {e}")
        finally:
            conn.close()

    def record_crossing(self, location_id, timestamp_str, track_id, device_id, vehicle_type, frame=None, bbox=None):
        """Główny wątek YOLO wywołuje to - funkcja jest natychmiastowa"""
        if device_id is None:
            device_id = self.get_or_create_device("default")
        
        self.task_queue.put({
            "type": "crossing",
            "location_id": location_id,
            "timestamp_str": timestamp_str,
            "track_id": track_id,
            "device_id": device_id,
            "vehicle_type": vehicle_type,
            "frame": frame,
            "bbox": bbox
        })

    def cleanup_screenshots(self):
        self.task_queue.put({"type": "cleanup"})

    def _process_cleanup(self):
        max_ss = self._cfg.get("max_screenshots", 1000)
        keep = self._cfg.get("keep_screenshots", 950)
        files = sorted([f for f in os.listdir(SCREENSHOT_DIR) if f.endswith(".jpg")], key=lambda f: os.path.getmtime(os.path.join(SCREENSHOT_DIR, f)))
        if len(files) > max_ss:
            for f in files[: len(files) - keep]:
                try: os.remove(os.path.join(SCREENSHOT_DIR, f))
                except OSError: pass

    def _conn(self):
        return self.pool.get_connection()

    def _create_tables_if_not_exist(self):
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS devices (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(100) NOT NULL UNIQUE, description TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);")
            cur.execute("CREATE TABLE IF NOT EXISTS locations (id INT AUTO_INCREMENT PRIMARY KEY, device_id INT NOT NULL, name VARCHAR(100) NOT NULL, description TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (device_id) REFERENCES devices(id), UNIQUE KEY (device_id, name));")
            cur.execute("CREATE TABLE IF NOT EXISTS crossings (id INT AUTO_INCREMENT PRIMARY KEY, device_id INT NOT NULL, location_id INT NOT NULL, recorded_at DATETIME NOT NULL, track_id INT NOT NULL, vehicle_type VARCHAR(50) NOT NULL DEFAULT 'bicycle', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (device_id) REFERENCES devices(id), FOREIGN KEY (location_id) REFERENCES locations(id), INDEX idx_recorded (recorded_at), INDEX idx_location (location_id, device_id));")
            cur.execute("CREATE TABLE IF NOT EXISTS co_15_minut (id INT AUTO_INCREMENT PRIMARY KEY, location_id VARCHAR(100) NOT NULL, date DATE NOT NULL, czas TIME NOT NULL, il_row INT DEFAULT 0, il_hul INT DEFAULT 0, synced_at DATETIME DEFAULT NULL, UNIQUE KEY idx_agregacja (location_id, date, czas));")
            conn.commit()
        finally:
            conn.close()

    def get_or_create_device(self, name, description=""):
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id FROM devices WHERE name=%s", (name,))
            row = cur.fetchone()
            if row: return row[0]
            cur.execute("INSERT INTO devices (name, description) VALUES (%s, %s)", (name, description))
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_or_create_location(self, name, description="", device_id=None):
        if device_id is None: device_id = self.get_or_create_device("default")
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id FROM locations WHERE name=%s AND device_id=%s", (name, device_id))
            row = cur.fetchone()
            if row: return row[0]
            cur.execute("INSERT INTO locations (device_id, name, description) VALUES (%s, %s, %s)", (device_id, name, description))
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_all_stats(self, location_id, device_id=None):
        if device_id is None: device_id = self.get_or_create_device("default")
        conn = self._conn()
        try:
            cur = conn.cursor()
            today = datetime.now().date()
            cur.execute("""SELECT SUM(CASE WHEN vehicle_type IN ('bicycle', 'cyclist') THEN 1 ELSE 0 END), SUM(CASE WHEN vehicle_type = 'e-scooter' THEN 1 ELSE 0 END), SUM(CASE WHEN vehicle_type IN ('bicycle', 'cyclist') AND DATE(recorded_at) = %s THEN 1 ELSE 0 END), SUM(CASE WHEN vehicle_type = 'e-scooter' AND DATE(recorded_at) = %s THEN 1 ELSE 0 END) FROM crossings WHERE location_id=%s AND device_id=%s""", (today, today, location_id, device_id))
            bike_all, scoot_all, bike_today, scoot_today = cur.fetchone()
            
            cur.execute("SELECT czas, (il_row + il_hul) FROM co_15_minut WHERE location_id = (SELECT name FROM locations WHERE id = %s) AND date = %s ORDER BY czas", (location_id, today))
            quarterly = [{"label": str(t)[:5], "count": int(c)} for t, c in cur.fetchall()]

            return {
                "bike_today": int(bike_today or 0), "scooter_today": int(scoot_today or 0),
                "bike_all_time": int(bike_all or 0), "scooter_all_time": int(scoot_all or 0),
                "today_total": int(bike_today or 0) + int(scoot_today or 0),
                "all_time_total": int(bike_all or 0) + int(scoot_all or 0), "quarterly": quarterly
            }
        finally:
            conn.close()

    def get_recent_crossings(self, location_id, device_id=None, limit=30):
        if device_id is None: device_id = self.get_or_create_device("default")
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT recorded_at, track_id, vehicle_type FROM crossings WHERE location_id=%s AND device_id=%s ORDER BY id DESC LIMIT %s", (location_id, device_id, limit))
            return [{"time_str": (ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))).strftime("%H:%M:%S"), "track_id": str(int(tid)), "vehicle_type": vtype} for ts, tid, vtype in cur.fetchall()]
        finally:
            conn.close()

    def cleanup_old_data(self, max_days):
        if max_days <= 0: return
        cutoff = datetime.now() - timedelta(days=max_days)
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM crossings WHERE recorded_at < %s", (cutoff,))
            cur.execute("DELETE FROM co_15_minut WHERE date < %s", (cutoff.date(),))
            conn.commit()
        finally:
            conn.close()

    def close(self):
        self.task_queue.put(None) # Bezpieczne zamykanie wątku tła