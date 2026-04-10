@echo off
echo Building Auto Clip Maker for Production...

echo.
echo === Building Frontend ===
cd /d "%~dp0frontend"
call npm run build

echo.
echo Build complete!
echo Frontend built to: frontend/dist
echo.
echo To run in production:
echo   cd backend
echo   python -m uvicorn main:app --host 0.0.0.0 --port 9000
echo.
pause
