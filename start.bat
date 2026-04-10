@echo off
title Auto Clip Launcher

echo ========================================
echo        Auto Clip Maker v2.0.0
echo ========================================
echo.

:: Start Backend
echo Starting Backend Server...
start "Auto Clip - Backend" cmd /k "cd /d %~dp0backend && python main.py"

:: Wait a moment for backend to initialize
timeout /t 0 /nobreak >nul

:: Start Frontend
echo Starting Frontend...
start "Auto Clip - Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ========================================
echo   Backend: http://localhost:8000
echo   Frontend: http://localhost:5173
echo ========================================
echo.
