import time
import logging
import requests
import schedule
import mysql.connector

# --- LOGOWANIE ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [AGENT API] %(message)s",
    datefmt="%H:%M:%S"
)

# --- KONFIGURACJA LOKALNEJ BAZY ---
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "bikecounter",
    "password": "BikeCount2026!",
    "database": "bicycle_counter"
}

# --- KONFIGURACJA DOCELOWEGO API  ---
API_URL = "http://192.168.200.201/do_bazy_danych_api.php"
API_HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": "klucz-1"
}

def run_sync():
    logging.info("Rozpoczynam synchronizację...")
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("SELECT id, location_id, date, czas, il_row, il_hul FROM co_15_minut WHERE synced_at IS NULL")
        wiersze = cur.fetchall()

        if not wiersze:
            logging.info("Brak nowych danych do wysłania.")
            return

        
        paczka_danych = {
            "miejsce": [],
            "data_pomiaru": [],
            "czas_pomiaru": [],
            "ile_rowerow": [],
            "ile_hulajnog": []
        }
        lista_id = []
        
        for wiersz in wiersze:
            paczka_danych["miejsce"].append(wiersz[1])
            paczka_danych["data_pomiaru"].append(str(wiersz[2]))
            paczka_danych["czas_pomiaru"].append(str(wiersz[3]))
            paczka_danych["ile_rowerow"].append(int(wiersz[4]))
            paczka_danych["ile_hulajnog"].append(int(wiersz[5]))
            
            lista_id.append(wiersz[0])

        odpowiedz = requests.post(API_URL, json=paczka_danych, headers=API_HEADERS, timeout=10)
        
        if odpowiedz.status_code in [200, 201]:
            placeholders = ','.join(['%s'] * len(lista_id))
            zapytanie_update = f"UPDATE co_15_minut SET synced_at = NOW() WHERE id IN ({placeholders})"
            cur.execute(zapytanie_update, lista_id)
            conn.commit()
            logging.info(f"Sukces: Wysłano i oznaczono {len(lista_id)} pomiarów. Odpowiedź serwera: {odpowiedz.text.strip()}")
        else:
            logging.error(f"Błąd API: Zwróciło kod {odpowiedz.status_code}. Treść: {odpowiedz.text}")
            
    except mysql.connector.Error as db_err:
        logging.error(f"Błąd lokalnej bazy danych: {db_err}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Błąd połączenia z siecią/API: {req_err}")
    except Exception as e:
        logging.error(f"Nieoczekiwany błąd programu: {e}")
        
    finally:
        if conn and conn.is_connected():
            conn.close()

def uruchom_petle_agenta():
    logging.info("Uruchamiam Agenta Synchronizacji...")
    run_sync() 
    schedule.every(15).minutes.do(run_sync)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    try:
        uruchom_petle_agenta()
    except KeyboardInterrupt:
        logging.info("Agent zatrzymany przez użytkownika.")