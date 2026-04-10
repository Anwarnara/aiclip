@echo off
title Auto Clip Maker - Frontend
echo Starting Frontend Dev Server...

cd /d "%~dp0frontend"
call npm run dev
pause
