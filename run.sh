#!/usr/bin/env bash
# run.sh — Launch the bicycle counter pipeline.
# Sets CUDA/DeepStream paths, kills camera hogs, shows URL, then starts.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS_DIR="/opt/nvidia/deepstream/deepstream"

export LD_LIBRARY_PATH="${DS_DIR}/lib:/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}"
export GST_PLUGIN_PATH="${DS_DIR}/lib/gst-plugins:${GST_PLUGIN_PATH:-}"
export PATH="${DS_DIR}/bin:${PATH:-}"
export PIP_BREAK_SYSTEM_PACKAGES=1

fuser -k /dev/video0 2>/dev/null || true
sleep 0.5

echo "=========================================="
echo "  Bicycle Counter v2.0"
echo "=========================================="
echo ""
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "  Dashboard:  http://localhost:8080"
if [ -n "$IP" ]; then
    echo "  Network:    http://${IP}:8080"
fi
echo "  Health:     http://localhost:8080/health"
echo "  Config:     ${SCRIPT_DIR}/config.yaml"
echo "  Logs:       ${SCRIPT_DIR}/bicycle_counter.log"
echo "=========================================="
echo ""

exec python3 "${SCRIPT_DIR}/bicycle_counter.py" "$@"
