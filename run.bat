@echo off
title Auto Clip Maker
echo ========================================
echo    Auto Clip Maker - Starting...
echo ========================================
echo.

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Start backend in background
echo Starting Backend Server...
start /B cmd /c "cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000"

REM Wait for backend to start
timeout /t 3 /nobreak > nul

REM Start frontend
echo Starting Frontend...
cd frontend
call npm run dev

pause
