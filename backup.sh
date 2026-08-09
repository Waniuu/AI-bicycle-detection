#!/usr/bin/env bash
# backup.sh — Snapshot the MySQL database and screenshots.
# Usage: ./backup.sh [backup_dir]
# Default backup dir: /home/student/CounterProject/backups

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="${1:-${SCRIPT_DIR}/backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$BACKUP_DIR"

echo "[$(date)] Starting backup..."

# Database dump
mysqldump -u bikecounter -p'BikeCount2026!' bicycle_counter \
  --single-transaction --quick --lock-tables=false \
  > "${BACKUP_DIR}/bikecounter_${TIMESTAMP}.sql" 2>/dev/null

# Compress
gzip "${BACKUP_DIR}/bikecounter_${TIMESTAMP}.sql"
echo "  Database: ${BACKUP_DIR}/bikecounter_${TIMESTAMP}.sql.gz"

# Screenshots tarball
if [ -d "${SCRIPT_DIR}/screenshots" ] && [ "$(ls -A "${SCRIPT_DIR}/screenshots" 2>/dev/null)" ]; then
  tar czf "${BACKUP_DIR}/screenshots_${TIMESTAMP}.tar.gz" -C "${SCRIPT_DIR}" screenshots/
  echo "  Screenshots: ${BACKUP_DIR}/screenshots_${TIMESTAMP}.tar.gz"
fi

# Cleanup backups older than 30 days
find "$BACKUP_DIR" -name '*.gz' -mtime +30 -delete 2>/dev/null || true

echo "[$(date)] Backup complete."
