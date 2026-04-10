@echo off
echo ============================================
echo   YouTube Auto Clip Maker - Setup
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
)

echo [2/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/3] Installing dependencies...
pip install --upgrade pip

REM Install PyTorch with CUDA first
echo.
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Install other requirements
echo.
echo Installing other requirements...
pip install -r requirements.txt

echo.
echo ============================================
echo   Setup Complete!
echo ============================================
echo.
echo Run 'run.bat' to start the application
echo.
pause
