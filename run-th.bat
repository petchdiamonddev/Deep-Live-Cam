@echo off
title Deep-Live-Cam Launcher (Thai)
echo ========================================================
echo   Deep-Live-Cam Launcher
echo ========================================================
echo.

if not exist venv\Scripts\python.exe (
    echo [INFO] Creating Virtual Environment...
    python -m venv venv
)

echo [INFO] Launching Deep-Live-Cam with DirectML GPU...
venv\Scripts\python.exe run.py --execution-provider dml --lang th

pause
