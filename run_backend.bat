@echo off
title Auto Clip Maker - Backend
echo Starting Backend Server...

REM Activate virtual environment
if exist "%~dp0venv\Scripts\activate.bat" (
    call "%~dp0venv\Scripts\activate.bat"
)

cd /d "%~dp0backend"
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
pause
