#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f "main.py" ]; then
	echo "ERROR: main.py is missing. Cannot start vision program."
	exit 1
fi

if [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
	# shellcheck disable=SC1091
	source ".venv/bin/activate"
fi

PYTHON_BIN="${PYTHON:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
	echo "ERROR: Python was not found. Install python3 or set the PYTHON environment variable."
	exit 1
fi

if ! "$PYTHON_BIN" -c "import cv2, numpy" >/dev/null 2>&1; then
	echo "ERROR: Missing Python dependency: cv2 or numpy."
	echo "On Orange Pi, try: sudo apt install -y python3-opencv python3-numpy"
	echo "If using project .venv, install numpy there and make sure OpenCV is available."
	exit 1
fi

export DISPLAY="${DISPLAY:-:0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

exec "$PYTHON_BIN" main.py
