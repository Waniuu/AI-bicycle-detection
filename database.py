"""
MySQL database module for bicycle counter.
Uses connection pooling. Config-driven via config_loader.
"""

import csv
import io
import os
import time
import logging
from datetime import datetime, timedelta

import mysql.connector
from mysql.connector import pooling

LOG = logging.getLogger("bike_counter")

CSV_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOT_DIR = os.path.join(CSV_DIR, "screenshots")


class Database:
    def __init__(self, cfg=None):
        if cfg is None:
            from config_loader import load
            cfg = load()
        
        LOG.info("Attempting to synchronize time with NTP server...")
        if os.system("sudo timedatectl set-ntp true && sudo systemctl restart systemd-timesyncd") == 0:
            LOG.info("Time synchronized successfully.")

        if datetime.now().year < 2024:
            LOG.warning("System time is incorrect (year < 2024). 'Today' stats will be inaccurate until time is synced.")
        
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

    def _conn(self):
        return self.pool.get_connection()

    def _create_tables_if_not_exist(self):
        conn = None
        try:
            conn = self.pool.get_connection()
            cur = conn.cursor()

            cur.execute("""
                CREATE TABLE IF NOT EXISTS devices (
                    id          INT AUTO_INCREMENT PRIMARY KEY,
                    name        VARCHAR(100) NOT NULL UNIQUE,
                    description TEXT,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS locations (
                    id          INT AUTO_INCREMENT PRIMARY KEY,
                    device_id   INT NOT NULL,
                    name        VARCHAR(100) NOT NULL,
                    description TEXT,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (device_id) REFERENCES devices(id),
                    UNIQUE KEY (device_id, name)
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS crossings (
                    id           INT AUTO_INCREMENT PRIMARY KEY,
                    device_id    INT NOT NULL,
                    location_id  INT NOT NULL,
                    recorded_at  DATETIME NOT NULL,
                    track_id     INT NOT NULL,
                    vehicle_type VARCHAR(50) NOT NULL DEFAULT 'bicycle',
                    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (device_id) REFERENCES devices(id),
                    FOREIGN KEY (location_id) REFERENCES locations(id),
                    INDEX idx_recorded (recorded_at),
                    INDEX idx_location (location_id, device_id)
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS co_15_minut (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    location_id VARCHAR(100) NOT NULL,
                    date DATE NOT NULL,
                    czas TIME NOT NULL,
                    il_row INT DEFAULT 0,
                    il_hul INT DEFAULT 0,
                    synced_at DATETIME DEFAULT NULL,
                    UNIQUE KEY idx_agregacja (location_id, date, czas)
                );
            """)
            conn.commit()
            LOG.info("Database tables checked/created successfully.")
        except mysql.connector.Error as e:
            LOG.error(f"Failed to create database tables: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn and conn.is_connected():
                conn.close()

    def get_or_create_device(self, name, description=""):
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id FROM devices WHERE name=%s", (name,))
            row = cur.fetchone()
            if row:
                return row[0]
            cur.execute(
                "INSERT INTO devices (name, description) VALUES (%s, %s)",
                (name, description),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_or_create_location(self, name, description="", device_id=None):
        if device_id is None:
            device_id = self.get_or_create_device("default")
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id FROM locations WHERE name=%s AND device_id=%s",
                (name, device_id),
            )
            row = cur.fetchone()
            if row:
                return row[0]
            cur.execute(
                "INSERT INTO locations (device_id, name, description) VALUES (%s, %s, %s)",
                (device_id, name, description),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def record_crossing(self, location_id, timestamp_str, track_id, device_id=None, vehicle_type=None):
        ALLOWED_TYPES = ('bicycle', 'cyclist', 'e-scooter')

        if device_id is None:
            device_id = self.get_or_create_device("default")
        
        dt = datetime.fromisoformat(timestamp_str)
        
        minute_bucket = (dt.minute // 15) * 15
        bucket_time = f"{dt.hour:02d}:{minute_bucket:02d}:00"
        
        is_bike = 1 if vehicle_type in ('cyclist', 'bicycle') else 0
        is_scooter = 1 if vehicle_type == 'e-scooter' else 0

        conn = self._conn()
        try:
            cur = conn.cursor()
            
            cur.execute( # dodanie rekordu do tabeli crossings
                "INSERT INTO crossings (device_id, location_id, recorded_at, track_id, vehicle_type) VALUES (%s, %s, %s, %s, %s)",
                (device_id, location_id, dt, track_id, vehicle_type),
            )
            cur.execute(
                """INSERT INTO co_15_minut
                   (location_id, date, czas, il_row, il_hul)
                   VALUES ((SELECT name FROM locations WHERE id = %s), %s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE 
                   il_row = il_row + VALUES(il_row), 
                   il_hul = il_hul + VALUES(il_hul),
                   synced_at = NULL""",
                (location_id, dt.date(), bucket_time, is_bike, is_scooter),
            )
            conn.commit()
        except Exception:
            LOG.exception("Failed to record crossing")
        except mysql.connector.Error as err:
            LOG.error("Database error during crossing record: %s", err)
            conn.rollback() # Wycofaj transakcję w razie błędu
        except Exception as e:
            LOG.exception("An unexpected error occurred while recording crossing: %s", e)
        finally:
            conn.close()

    def _format_time_label(self, time_obj):
        if isinstance(time_obj, timedelta):
            total_seconds = int(time_obj.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours:02d}:{minutes:02d}"
        return str(time_obj)[:5]

    def get_today_quarterly(self, location_id, device_id=None):
        if device_id is None:
            device_id = self.get_or_create_device("default")
        conn = self._conn()
        try:
            cur = conn.cursor()
            today = datetime.now().date()
            cur.execute(
                """SELECT czas, (il_row + il_hul) as total
                   FROM co_15_minut
                   WHERE location_id = (SELECT name FROM locations WHERE id = %s)
                     AND date = %s
                   ORDER BY czas""",
                (location_id, today),
            )
            
            return [
                {"label": self._format_time_label(t), "count": int(c)}
                for t, c in cur.fetchall()
            ]
        finally:
            conn.close()

    def get_all_stats(self, location_id, device_id=None):
        if device_id is None:
            device_id = self.get_or_create_device("default")
        conn = self._conn()
        try:
            cur = conn.cursor()
            today = datetime.now().date()
            
            cur.execute(
                """SELECT
                    SUM(CASE WHEN vehicle_type IN ('bicycle', 'cyclist') THEN 1 ELSE 0 END) as bike_all_time,
                    SUM(CASE WHEN vehicle_type = 'e-scooter' THEN 1 ELSE 0 END) as scooter_all_time,
                    SUM(CASE WHEN vehicle_type IN ('bicycle', 'cyclist') AND DATE(recorded_at) = %s THEN 1 ELSE 0 END) as bike_today,
                    SUM(CASE WHEN vehicle_type = 'e-scooter' AND DATE(recorded_at) = %s THEN 1 ELSE 0 END) as scooter_today
                   FROM crossings
                   WHERE location_id=%s AND device_id=%s""",
                (today, today, location_id, device_id),
            )
            stats = cur.fetchone()
            bike_all_time, scooter_all_time, bike_today, scooter_today = stats

            cur.execute(
                """SELECT czas, (il_row + il_hul) as total
                   FROM co_15_minut
                   WHERE location_id = (SELECT name FROM locations WHERE id = %s)
                     AND date = %s
                   ORDER BY czas""",
                (location_id, today),
            )
            
            quarterly_data = [
                {"label": self._format_time_label(t), "count": int(c)}
                for t, c in cur.fetchall()
            ]

            return {
                "bike_today": int(bike_today or 0),
                "scooter_today": int(scooter_today or 0),
                "bike_all_time": int(bike_all_time or 0),
                "scooter_all_time": int(scooter_all_time or 0),
                "today_total": int(bike_today or 0) + int(scooter_today or 0),
                "all_time_total": int(bike_all_time or 0) + int(scooter_all_time or 0),
                "quarterly": quarterly_data,
            }
        finally:
            conn.close()

    def get_recent_crossings(self, location_id, device_id=None, limit=30):
        if device_id is None:
            device_id = self.get_or_create_device("default")
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """SELECT recorded_at, track_id, vehicle_type FROM crossings
                   WHERE location_id=%s AND device_id=%s
                   ORDER BY id DESC LIMIT %s""",
                (location_id, device_id, limit),
            )
            events = []
            for ts, tid, vtype in cur.fetchall():
                dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
                events.append({"time_str": dt.strftime("%H:%M:%S"), "track_id": str(int(tid)), "vehicle_type": vtype})
            return events
        finally:
            conn.close()

    def cleanup_old_data(self, max_days):
        if max_days <= 0:
            return
        cutoff = datetime.now() - timedelta(days=max_days)
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM crossings WHERE recorded_at < %s", (cutoff,))
            deleted = cur.rowcount
            
            # Zaktualizowane na 'date'
            cur.execute("DELETE FROM co_15_minut WHERE date < %s", (cutoff.date(),))
            deleted_q = cur.rowcount
            
            if deleted or deleted_q:
                LOG.info("Retention cleanup: %d crossings, %d aggregated rows deleted", deleted, deleted_q)
            conn.commit()
        finally:
            conn.close()

    def cleanup_screenshots(self):
        max_ss = self._cfg.get("max_screenshots", 1000)
        keep = self._cfg.get("keep_screenshots", 950)
        if not os.path.isdir(SCREENSHOT_DIR):
            return
        files = sorted(
            [f for f in os.listdir(SCREENSHOT_DIR) if f.endswith(".jpg")],
            key=lambda f: os.path.getmtime(os.path.join(SCREENSHOT_DIR, f)),
        )
        if len(files) <= max_ss:
            return
        for f in files[: len(files) - keep]:
            try:
                os.remove(os.path.join(SCREENSHOT_DIR, f))
            except OSError:
                pass
        LOG.info("Cleaned up %d old screenshots", len(files) - keep)

    def close(self):
        pass