#!/bin/bash

echo "==================================================="
echo "     Manga Translator Pro - Easy Installer"
echo "==================================================="

# Check for Python
if ! command -v python3 &> /dev/null
then
    echo "[ERROR] Python3 could not be found. Please install it."
    exit 1
fi

# Create venv
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install
echo "[INFO] Installing dependencies..."
pip install -r requirements.txt

# Run
echo ""
echo "[INFO] Starting the App..."
export PYTHONPATH=$PWD
python src/ui/gradio_app.py
