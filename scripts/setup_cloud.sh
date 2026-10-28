#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv-cloud}"
"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install -r requirements-cloud.txt
"$VENV/bin/python" -m pip check
printf '\nReady. Run: %s/bin/python cloud_asr.py /path/to/audio --output-dir 3asr\n' "$VENV"
