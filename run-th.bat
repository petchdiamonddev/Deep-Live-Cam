@echo off
title Deep-Live-Cam Launcher (DirectML GPU)
echo ========================================================
echo   Deep-Live-Cam (DirectML GPU Acceleration)
echo ========================================================
echo.

if not exist venv\Scripts\activate.bat (
    echo [INFO] Creating Virtual Environment (venv)...
    python -m venv venv
)

echo [INFO] Activating Virtual Environment...
call venv\Scripts\activate.bat

echo [INFO] Checking dependencies...
pip install -r requirements.txt --quiet

echo.
echo [INFO] Starting Deep-Live-Cam with DirectML GPU...
python run.py --execution-provider dml --lang th

pause
