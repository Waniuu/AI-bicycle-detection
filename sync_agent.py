#!/usr/bin/env python3
"""
Sync Agent for Bicycle Counter

Ten skrypt jest odpowiedzialny za okresowe wysyłanie zagregowanych danych
z lokalnej bazy danych do centralnej bazy danych w siedzibie głównej.

Działanie:
1. Łączy się z lokalną bazą danych.
2. Pobiera wszystkie rekordy z `quarterly_counts`, które nie zostały jeszcze zsynchronizowane (`synced_at IS NULL`).
3. Łączy się z centralną bazą danych (HQ).
4. Wysyła paczkę danych do centrali, używając `INSERT ... ON DUPLICATE KEY UPDATE`.
5. Jeśli wysyłka się powiedzie, oznacza lokalne rekordy jako zsynchronizowane, ustawiając `synced_at`.

Uruchamianie:
Skrypt powinien być uruchamiany cyklicznie, np. co 15 minut za pomocą crona.
*/15 * * * * /usr/bin/python3 /home/student/CounterProject/sync_agent.py >> /home/student/CounterProject/sync.log 2>&1
"""

import logging
import os
from datetime import datetime

import mysql.connector

# --- Konfiguracja ---
# W docelowym rozwiązaniu te dane powinny pochodzić z pliku config.yaml
LOCAL_DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "bikecounter",
    "password": "BikeCount2026!",
    "database": "bicycle_counter",
}

HQ_DB_CONFIG = {
    "host": "192.168.1.100",  # Przykładowy adres IP serwera w centrali
    "port": 3306,
    "user": "hq_user",
    "password": "HqPassword!",
    "database": "hq_bicycle_counter",
}

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sync.log")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    datefmt="%Y-%m-%d %H:%M:%S"
)

def run_sync():
    logging.info("Starting sync process...")
    local_conn = None
    hq_conn = None

    try:
        # 1. Połącz z lokalną bazą i pobierz dane do synchronizacji
        local_conn = mysql.connector.connect(**LOCAL_DB_CONFIG)
        local_cur = local_conn.cursor()

        # Pobieramy wszystkie rekordy, które nie mają ustawionej daty synchronizacji
        local_cur.execute("SELECT id, device_id, location_id, bucket_date, bucket_hour, bucket_quarter, total_count FROM quarterly_counts WHERE synced_at IS NULL")
        records_to_sync = local_cur.fetchall()

        if not records_to_sync:
            logging.info("No new records to sync. Exiting.")
            return

        logging.info(f"Found {len(records_to_sync)} records to synchronize.")

        # 2. Połącz z bazą w centrali i wyślij dane
        hq_conn = mysql.connector.connect(**HQ_DB_CONFIG)
        hq_cur = hq_conn.cursor()

        # Używamy `executemany` do wydajnego wstawienia wielu rekordów
        # Zakładamy, że tabela w centrali ma identyczną strukturę
        insert_query = """
            INSERT INTO quarterly_counts (device_id, location_id, bucket_date, bucket_hour, bucket_quarter, total_count)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE total_count = total_count + VALUES(total_count)
        """
        # Przekształcamy rekordy, aby pasowały do zapytania (pomijamy lokalne ID)
        hq_data = [r[1:] for r in records_to_sync]
        hq_cur.executemany(insert_query, hq_data)
        hq_conn.commit()
        logging.info(f"Successfully sent {hq_cur.rowcount} operations to HQ database.")

        # 3. Oznacz lokalne rekordy jako zsynchronizowane
        record_ids = [r[0] for r in records_to_sync]
        update_query = f"UPDATE quarterly_counts SET synced_at = %s WHERE id IN ({','.join(['%s']*len(record_ids))})"
        local_cur.execute(update_query, [datetime.now()] + record_ids)
        local_conn.commit()
        logging.info(f"Marked {local_cur.rowcount} local records as synced.")

    except mysql.connector.Error as err:
        logging.error(f"Database error during sync: {err}")
        if hq_conn:
            hq_conn.rollback()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        if local_conn and local_conn.is_connected():
            local_conn.close()
        if hq_conn and hq_conn.is_connected():
            hq_conn.close()
        logging.info("Sync process finished.")

if __name__ == "__main__":
    run_sync()