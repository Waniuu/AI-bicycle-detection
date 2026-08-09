#!/usr/bin/env python3
"""Export bicycle counting data to CSV from MySQL. Run anytime."""

from database import Database

db = Database()

try:
    device_id = db.get_or_create_device("jetson-orin-01", "Jetson Orin Nano - Main Unit")
    location_id = db.get_or_create_location("Camera_01", "Main entrance", device_id=device_id)
    db.export_to_csv(location_id, device_id)
    db.export_all_crossings_csv(location_id, device_id)
    print("\nFiles created:")
    print("  bicycle_counts.csv           — 15-minute aggregated counts")
    print("  bicycle_crossings_detail.csv — every individual crossing event")
    print("\nYou can also view data in the browser: http://localhost:8080/database")
finally:
    db.close()
