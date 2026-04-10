@echo off
echo Installing Auto Clip Maker dependencies...

echo.
echo === Installing Backend Dependencies ===
cd /d "%~dp0backend"
pip install -r requirements.txt

echo.
echo === Installing Frontend Dependencies ===
cd /d "%~dp0frontend"
call npm install

echo.
echo Installation complete!
echo.
echo To run the app:
echo   1. Run run_backend.bat in one terminal
echo   2. Run run_frontend.bat in another terminal
echo   3. Open http://localhost:5173 in your browser
echo.
pause
