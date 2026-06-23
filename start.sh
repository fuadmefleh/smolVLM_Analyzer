#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/.venv"
REQ_FILE="$DIR/requirements.txt"
REQ_STAMP="$VENV/.requirements.sha256"

if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"

current_hash="$(sha256sum "$REQ_FILE" | awk '{print $1}')"
saved_hash=""
if [ -f "$REQ_STAMP" ]; then
    saved_hash="$(cat "$REQ_STAMP")"
fi

if [ "$current_hash" != "$saved_hash" ]; then
    echo "Installing/updating Python dependencies..."
    pip install --upgrade pip
    pip install -r "$REQ_FILE"
    printf '%s' "$current_hash" > "$REQ_STAMP"
else
    echo "Dependencies unchanged; skipping pip install."
fi

python "$DIR/webapp.py"
