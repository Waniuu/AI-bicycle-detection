#!/usr/bin/env python3
import random
from datetime import datetime, timedelta
import mysql.connector

# Konfiguracja połączenia z lokalną bazą na Jetsonie
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "bikecounter",
    "password": "BikeCount2026!",
    "database": "bicycle_counter"
}

LOCATION_NAME = "Camera_01"


def get_hourly_weight(hour):
    """Zwraca mnożnik natężenia ruchu w zależności od pory dnia w centrum miasta."""
    if 7 <= hour < 9:      # Szczyt poranny (dojazdy do pracy/szkół)
        return (12, 35), (6, 22)
    elif 9 <= hour < 15:   # Środek dnia (ruch umiarkowany)
        return (3, 14), (2, 10)
    elif 15 <= hour < 18:  # Główny szczyt popołudniowy
        return (15, 45), (10, 30)
    elif 18 <= hour < 22:  # Wieczorny ruch rekreacyjny
        return (5, 18), (4, 16)
    else:                  # Noc (22:00 - 06:00)
        return (0, 2), (0, 1)


def generate_city_traffic(days=3):
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()

    now = datetime.now()
    # Zaokrąglenie aktualnego czasu w dół do pełnego kwadransa
    current_bucket = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
    start_time = current_bucket - timedelta(days=days)

    query = """
        INSERT INTO co_15_minut (location_id, date, czas, il_row, il_hul, synced_at)
        VALUES (%s, %s, %s, %s, %s, NULL)
        ON DUPLICATE KEY UPDATE 
            il_row = VALUES(il_row),
            il_hul = VALUES(il_hul),
            synced_at = NULL
    """

    cursor_time = start_time
    total_records = 0
    total_bikes = 0
    total_scooters = 0

    print(f"🚀 Wypełnianie bazy danymi testowymi dla '{LOCATION_NAME}'...")

    while cursor_time <= current_bucket:
        hour = cursor_time.hour
        bike_range, scooter_range = get_hourly_weight(hour)

        # Losowanie liczby pojazdów w danym kwadransie
        bikes = random.randint(*bike_range)
        scooters = random.randint(*scooter_range)

        # Piątkowy/weekendowy bonus dla hulajnóg wieczorem
        if cursor_time.weekday() in (4, 5) and 18 <= hour <= 23:
            scooters = int(scooters * 1.5)

        date_val = cursor_time.date()
        time_val = cursor_time.strftime("%H:%M:%S")

        cur.execute(query, (LOCATION_NAME, date_val, time_val, bikes, scooters))

        total_records += 1
        total_bikes += bikes
        total_scooters += scooters
        cursor_time += timedelta(minutes=15)

    conn.commit()
    cur.close()
    conn.close()

    print(f"✅ Zakończono sukcesem!")
    print(f"   • Wstawiono kwadransów: {total_records}")
    print(f"   • Łącznie rowerów:     {total_bikes}")
    print(f"   • Łącznie hulajnóg:    {total_scooters}")


if __name__ == "__main__":
    generate_city_traffic(days=3)