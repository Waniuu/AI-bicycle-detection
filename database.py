"""
MySQL database module for bicycle counter.
Uses connection pooling. Config-driven via config_loader.
"""

import csv
import io
import logging
import os
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

    def _conn(self):
        return self.pool.get_connection()

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

    def record_crossing(self, location_id, timestamp_str, track_id, device_id=None):
        if device_id is None:
            device_id = self.get_or_create_device("default")
        dt = datetime.fromisoformat(timestamp_str)
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO crossings (device_id, location_id, recorded_at, track_id) VALUES (%s, %s, %s, %s)",
                (device_id, location_id, dt, track_id),
            )
            cur.execute(
                """INSERT INTO quarterly_counts
                   (device_id, location_id, bucket_date, bucket_hour, bucket_quarter, total_count)
                   VALUES (%s, %s, %s, %s, %s, 1)
                   ON DUPLICATE KEY UPDATE total_count = total_count + 1""",
                (device_id, location_id, dt.date(), dt.hour, dt.minute // 15),
            )
            conn.commit()
        except Exception:
            LOG.exception("Failed to record crossing")
        finally:
            conn.close()

    def get_today_quarterly(self, location_id, device_id=None):
        if device_id is None:
            device_id = self.get_or_create_device("default")
        conn = self._conn()
        try:
            cur = conn.cursor()
            today = datetime.now().date()
            cur.execute(
                """SELECT bucket_hour, bucket_quarter, total_count
                   FROM quarterly_counts
                   WHERE location_id=%s AND device_id=%s AND bucket_date=%s
                   ORDER BY bucket_hour, bucket_quarter""",
                (location_id, device_id, today),
            )
            return [
                {"label": f"{int(h):02d}:{int(q)*15:02d}", "count": int(c)}
                for h, q, c in cur.fetchall()
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
                """SELECT COALESCE(SUM(total_count), 0)
                   FROM quarterly_counts WHERE location_id=%s AND device_id=%s AND bucket_date=%s""",
                (location_id, device_id, today),
            )
            today_total = int(cur.fetchone()[0])
            cur.execute(
                """SELECT COALESCE(SUM(total_count), 0)
                   FROM quarterly_counts WHERE location_id=%s AND device_id=%s""",
                (location_id, device_id),
            )
            all_time_total = int(cur.fetchone()[0])
            return {
                "today_total": today_total,
                "all_time_total": all_time_total,
                "quarterly": self.get_today_quarterly(location_id, device_id),
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
                """SELECT recorded_at, track_id FROM crossings
                   WHERE location_id=%s AND device_id=%s
                   ORDER BY id DESC LIMIT %s""",
                (location_id, device_id, limit),
            )
            events = []
            for ts, tid in cur.fetchall():
                dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
                events.append({"time_str": dt.strftime("%H:%M:%S"), "track_id": str(int(tid))})
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
            cur.execute("DELETE FROM quarterly_counts WHERE bucket_date < %s", (cutoff.date(),))
            deleted_q = cur.rowcount
            if deleted or deleted_q:
                LOG.info("Retention cleanup: %d crossings, %d quarterly rows deleted", deleted, deleted_q)
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
