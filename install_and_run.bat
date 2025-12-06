
@echo off
setlocal

echo ===================================================
echo   Manga Translator Pro - Easy Installer
echo ===================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.10+ and try again.
    pause
    exit /b
)

REM Create Virtual Environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate Virtual Environment
call venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Ask for API Key if not set in .env
if not exist ".env" (
    set /p APIKEY="Enter your Google Gemini API Key: "
    echo GOOGLE_API_KEY=%APIKEY% > .env
)

echo Starting the application...
python app.py

pause
