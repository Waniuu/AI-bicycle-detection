#!/usr/bin/env bash
# ============================================================================
# Jetson Orin Nano – Bicycle Counter Setup Script
# ============================================================================
# Run this script BEFORE bicycle_counter.py.
# Tested on: JetPack 5.x / 6.x (L4T 35.x / 36.x) on Jetson Orin Nano.
#
# Usage:
#   chmod +x setup.sh
#   sudo ./setup.sh
#
# What this script does:
#   1. Installs system packages and GStreamer plugins
#   2. Installs Python dependencies (pyds, PyGObject, numpy)
#   3. Clones and builds the DeepStream-Yolo custom parser plugin
#   4. Downloads YOLO11n nano weights and exports to ONNX
#   5. Builds the TensorRT engine from the ONNX model
#   6. Verifies the full installation
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/build"
DEEPSTREAM_DIR="/opt/nvidia/deepstream/deepstream"
CUSTOM_PARSER_DIR="${WORK_DIR}/DeepStream-Yolo"
YOLO11N_ONNX="${SCRIPT_DIR}/yolo11n.onnx"
YOLO11N_ENGINE="${SCRIPT_DIR}/yolo11n.engine"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# ============================================================================
# Pre-flight checks
# ============================================================================
preflight() {
    info "Running pre-flight checks..."

    if [[ $EUID -ne 0 ]]; then
        fail "This script must be run as root. Use: sudo ./setup.sh"
    fi

    if ! command -v jetson_release &>/dev/null && [[ ! -e /etc/nv_tegra_release ]]; then
        warn "jetson_release not found – this may not be a Jetson device."
    fi

    # DeepStream may be at the standard path, a symlink we created, or inside
    # a Downloads folder (e.g. manually extracted tarball).
    if [[ ! -d "${DEEPSTREAM_DIR}" ]]; then
        # Try to find a deepstream-*/opt/nvidia/deepstructure pattern
        local FOUND
        FOUND=$(find /home -maxdepth 6 -type d -name "deepstream-*" 2>/dev/null | head -1)
        if [[ -n "${FOUND}" ]]; then
            info "DeepStream found at non-standard path: ${FOUND}"
            info "Creating symlink at ${DEEPSTREAM_DIR}..."
            sudo mkdir -p /opt/nvidia/deepstream
            sudo ln -sf "${FOUND}" "${DEEPSTREAM_DIR}"
        else
            fail "DeepStream SDK not found at ${DEEPSTREAM_DIR}. Install JetPack with DeepStream first."
        fi
    fi

    info "DeepStream found at ${DEEPSTREAM_DIR}"
    ok "Pre-flight checks passed."
}

# ============================================================================
# 1. System packages and GStreamer plugins
# ============================================================================
install_system_packages() {
    info "===== Step 1/7: Installing system packages ====="

    # Fix any broken packages from a prior failed install (e.g. deepstream-9.0
    # pulling cuda-cudart deps that may not be resolvable on the Jetson apt
    # repo). dpkg --configure -a + apt --fix-broken attempts to repair state.
    info "Attempting to fix broken packages..."
    dpkg --configure -a --force-all 2>/dev/null || true
    apt-get -f install -y 2>/dev/null || true

    apt-get update

    # Each package is installed separately so one failure does not abort the
    # entire step.  Packages already installed will simply be skipped.
    local PACKAGES=(
        build-essential
        cmake
        git
        wget
        pkg-config
        libgstrtspserver-1.0-dev
        libgstreamer1.0-dev
        libgstreamer-plugins-base1.0-dev
        gstreamer1.0-plugins-bad
        gstreamer1.0-plugins-good
        gstreamer1.0-plugins-ugly
        gstreamer1.0-tools
        python3-dev
        python3-pip
        python3-gi
        python3-gi-cairo
        gir1.2-gstreamer-1.0
        gir1.2-gst-rtsp-server-1.0
        libcairo2-dev
        libgirepository1.0-dev
    )

    local FAILED=()

    for pkg in "${PACKAGES[@]}"; do
        if dpkg -s "${pkg}" &>/dev/null | grep -q "Status: install ok installed"; then
            info "  ${pkg} already installed, skipping."
            continue
        fi
        if apt-get install -y --no-install-recommends "${pkg}" 2>&1; then
            ok "  ${pkg} installed."
        else
            warn "  ${pkg} failed to install – skipping."
            FAILED+=("${pkg}")
        fi
    done

    if [[ ${#FAILED[@]} -gt 0 ]]; then
        warn "Some packages could not be installed: ${FAILED[*]}"
        warn "This is usually caused by the broken deepstream-9.0 package."
        warn "If GStreamer/pyds still work, you can safely ignore these failures."
    fi

    ok "System packages step complete."
}

# ============================================================================
# 2. Python dependencies
# ============================================================================
install_python_deps() {
    info "===== Step 2/7: Installing Python dependencies ====="

    # On Jetson, pyds and PyGObject require access to system-level GStreamer
    # and GI libraries, so we MUST install system-wide (not in a venv).
    # PEP 668 blocks this by default on Python 3.12+; the environment variable
    # PIP_BREAK_SYSTEM_PACKAGES=1 is the reliable workaround across all pip
    # versions (Debian-patched pip may not accept the CLI flag).
    export PIP_BREAK_SYSTEM_PACKAGES=1

    # Do NOT upgrade pip here – on Ubuntu/Debian Jetson images the system pip
    # is Debian-managed and cannot be upgraded without breaking dpkg tracking.
    # The existing pip 24.0+ works fine for our installs.

    # pyds – DeepStream Python bindings
    # The pip package works on JetPack 5.x/6.x. If it fails, build from source:
    #   cd ${DEEPSTREAM_DIR}/sources && make -C pyds
    pip3 install pyds || {
        warn "pip install pyds failed – building from DeepStream sources..."
        cd "${DEEPSTREAM_DIR}/sources"
        make -C pyds
        pip3 install "${DEEPSTREAM_DIR}/sources/pyds/pyds-*.whl" || \
            fail "pyds installation failed. Build manually: cd ${DEEPSTREAM_DIR}/sources && make -C pyds"
    }

    # PyGObject for GLib/GStreamer introspection bindings
    pip3 install PyGObject || warn "PyGObject pip install failed (system package may suffice)."

    # NumPy for any array math in post-processing
    pip3 install numpy

    # Force numpy<2 for compatibility with system matplotlib on Jetson
    pip3 install "numpy<2" 2>/dev/null || true

    ok "Python dependencies installed."
}

# ============================================================================
# 3. Build DeepStream-Yolo custom parser (nvdsinfer_custom_impl_Yolo.so)
# ============================================================================
build_custom_parser() {
    info "===== Step 3/7: Building DeepStream-Yolo custom parser ====="

    mkdir -p "${WORK_DIR}"

    if [[ -d "${CUSTOM_PARSER_DIR}" ]]; then
        info "DeepStream-Yolo already cloned, pulling latest..."
        cd "${CUSTOM_PARSER_DIR}"
        git pull
    else
        info "Cloning DeepStream-Yolo repository..."
        git clone https://github.com/marcoslucianops/DeepStream-Yolo.git "${CUSTOM_PARSER_DIR}"
    fi

    # When run under sudo, the clone is owned by root. Fix ownership so
    # make (and future runs) can write .o files without issues.
    chown -R "$(logname 2>/dev/null || echo ${SUDO_USER:-root}):$(id -gn)" "${CUSTOM_PARSER_DIR}" 2>/dev/null || true

    cd "${CUSTOM_PARSER_DIR}"

    # Detect CUDA version from the system installation
    local CUDA_VER
    CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
    if [[ -z "${CUDA_VER}" ]]; then
        # Fallback: find the highest installed cuda-X.Y directory
        CUDA_VER=$(ls -d /usr/local/cuda-* 2>/dev/null | grep -oP 'cuda-\K[0-9]+\.[0-9]+' | sort -V | tail -1)
    fi
    if [[ -z "${CUDA_VER}" ]]; then
        fail "Cannot detect CUDA version. Is the CUDA toolkit installed?"
    fi
    info "Detected CUDA version: ${CUDA_VER}"

    NVDS_VERSION="6.4"

    info "Building nvdsinfer_custom_impl_Yolo.so (NVDS ${NVDS_VERSION}, CUDA ${CUDA_VER})..."

    make -C nvdsinfer_custom_impl_Yolo clean 2>/dev/null || true

    make -C nvdsinfer_custom_impl_Yolo \
        CUDA_VER="${CUDA_VER}" \
        NVDS_VERSION="${NVDS_VERSION}" \
        -j"$(nproc)" || fail "Custom parser build failed."

    # Copy the built shared library to the project directory
    cp nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so "${SCRIPT_DIR}/"

    ok "Custom parser built and copied to ${SCRIPT_DIR}/libnvdsinfer_custom_impl_Yolo.so"
}

# ============================================================================
# 4. Download and export YOLO11n to ONNX
# ============================================================================
download_yolo11n() {
    info "===== Step 4/7: Downloading YOLO11n and exporting to ONNX ====="

    # Install ultralytics (YOLO Python package) for model export
    pip3 install ultralytics onnx onnxslim

    if [[ -f "${YOLO11N_ONNX}" ]]; then
        ok "YOLO11n ONNX already exists at ${YOLO11N_ONNX}, skipping download."
        return
    fi

    info "Downloading YOLO11n nano weights and exporting to ONNX..."
    python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.export(format='onnx', imgsz=640, opset=12, simplify=True)
print('ONNX export complete: yolo11n.onnx')
"

    if [[ ! -f yolo11n.onnx ]]; then
        fail "ONNX export failed – yolo11n.onnx not found."
    fi

    # Move to project directory if exported elsewhere
    if [[ ! -f "${YOLO11N_ONNX}" ]]; then
        mv yolo11n.onnx "${YOLO11N_ONNX}"
    fi

    ok "YOLO11n ONNX model ready at ${YOLO11N_ONNX}"
}

# ============================================================================
# 5. Build TensorRT engine from ONNX
# ============================================================================
build_tensorrt_engine() {
    info "===== Step 5/7: Building TensorRT engine from ONNX ====="

    if [[ -f "${YOLO11N_ENGINE}" ]]; then
        ok "TensorRT engine already exists at ${YOLO11N_ENGINE}, skipping."
        return
    fi

    if ! command -v trtexec &>/dev/null; then
        fail "trtexec not found. Ensure TensorRT is installed via JetPack."
    fi

    info "Building TensorRT FP16 engine (this may take several minutes on Orin Nano)..."
    trtexec \
        --onnx="${YOLO11N_ONNX}" \
        --saveEngine="${YOLO11N_ENGINE}" \
        --fp16

    if [[ ! -f "${YOLO11N_ENGINE}" ]]; then
        fail "TensorRT engine build failed."
    fi

    ok "TensorRT engine built at ${YOLO11N_ENGINE}"
}

# ============================================================================
# 6. Copy required tracker config
# ============================================================================
copy_tracker_config() {
    info "===== Step 6/7: Verifying tracker config ====="

    TRACKER_CFG="${DEEPSTREAM_DIR}/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"

    if [[ -f "${TRACKER_CFG}" ]]; then
        ok "NvDCF tracker config found at ${TRACKER_CFG}"
    else
        warn "NvDCF tracker config not found at expected path."
        warn "The pipeline references this file. Ensure it exists or update deepstream_app_config.txt."
    fi
}

# ============================================================================
# 7. Verify installation
# ============================================================================
verify_installation() {
    info "===== Step 7/7: Verifying installation ====="

    local errors=0

    # Check Python imports
    info "Checking Python imports..."
    python3 -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst; print('  PyGObject + GStreamer OK')" 2>&1 || { warn "PyGObject/GStreamer import failed"; ((errors++)); }
    python3 -c "import pyds; print('  pyds OK')" 2>&1 || { warn "pyds import failed"; ((errors++)); }
    python3 -c "import numpy; print('  numpy OK')" 2>&1 || { warn "numpy import failed"; ((errors++)); }

    # Check GStreamer elements
    info "Checking GStreamer elements..."
    export LD_LIBRARY_PATH="${DEEPSTREAM_DIR}/lib:${LD_LIBRARY_PATH:-}"
    export GST_PLUGIN_PATH="${DEEPSTREAM_DIR}/lib/gst-plugins:${GST_PLUGIN_PATH:-}"
    for elem in nvarguscamerasrc nvvideoconvert nvstreammux nvinfer nvtracker nvdsosd nveglglessink; do
        if gst-inspect-1.0 "${elem}" &>/dev/null 2>&1; then
            ok "  ${elem} found"
        else
            warn "  ${elem} NOT found – may need DeepStream env sourced or GStreamer restart"
            ((errors++))
        fi
    done

    # Check project files
    info "Checking project files..."
    for f in bicycle_counter.py config_infer_primary.txt deepstream_app_config.txt labels.txt libnvdsinfer_custom_impl_Yolo.so yolo11n.engine; do
        if [[ -f "${SCRIPT_DIR}/${f}" ]]; then
            ok "  ${f} found"
        else
            warn "  ${f} NOT found in ${SCRIPT_DIR}"
            ((errors++))
        fi
    done

    echo ""
    if [[ ${errors} -eq 0 ]]; then
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  ALL CHECKS PASSED – READY TO RUN!     ${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        info "Run the bicycle counter with:"
        info "  cd ${SCRIPT_DIR}"
        info "  python3 bicycle_counter.py"
    else
        echo -e "${YELLOW}========================================${NC}"
        echo -e "${YELLOW}  ${errors} WARNING(S) – review above     ${NC}"
        echo -e "${YELLOW}========================================${NC}"
        echo ""
        warn "Fix the warnings above before running bicycle_counter.py"
    fi
}

# ============================================================================
# Cleanup build artifacts (optional)
# ============================================================================
cleanup() {
    info "Cleaning up apt cache..."
    apt-get clean
    rm -rf /var/lib/apt/lists/*
    ok "Cleanup done."
}

# ============================================================================
# Main
# ============================================================================
main() {
    echo ""
    echo -e "${CYAN}============================================${NC}"
    echo -e "${CYAN}  Bicycle Counter – Jetson Orin Nano Setup  ${NC}"
    echo -e "${CYAN}============================================${NC}"
    echo ""

    # Step 0: Repair dpkg state if a prior install was interrupted
    info "Repairing dpkg state if needed..."
    dpkg --configure -a --force-all 2>/dev/null || true

    preflight
    install_system_packages
    install_python_deps
    build_custom_parser
    download_yolo11n
    build_tensorrt_engine
    copy_tracker_config
    verify_installation
    cleanup

    echo ""
    info "Setup complete. Review the output above for any warnings."
}

main "$@"
