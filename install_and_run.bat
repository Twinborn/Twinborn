@echo off
setlocal

echo ===================================================
echo      Manga Translator Pro - Easy Installer
echo ===================================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10+ from python.org
    pause
    exit /b
)

:: Create Virtual Environment if it doesn't exist
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

:: Activate Virtual Environment
call venv\Scripts\activate

:: Install Dependencies
echo [INFO] Installing dependencies (this may take a while)...
pip install -r requirements.txt

:: Run the App
echo.
echo [INFO] Starting the App...
echo [INFO] Please open your browser at http://localhost:7860 if it doesn't open automatically.
echo.

set PYTHONPATH=%CD%
python src/ui/gradio_app.py

pause
