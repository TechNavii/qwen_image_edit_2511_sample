#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Installing dependencies..."
    ./venv/bin/pip install -r requirements.txt
fi

echo "Starting Qwen Image Edit GUI..."
./venv/bin/python app.py
