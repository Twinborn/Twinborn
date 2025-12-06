
#!/bin/bash

echo "==================================================="
echo "  Manga Translator Pro - Easy Installer"
echo "==================================================="

# Check python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install it first."
    exit 1
fi

# Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install deps
echo "Installing dependencies..."
pip install -r requirements.txt

# API Key
if [ ! -f ".env" ]; then
    read -p "Enter your Google Gemini API Key: " APIKEY
    echo "GOOGLE_API_KEY=$APIKEY" > .env
fi

echo "Starting the application..."
python app.py
