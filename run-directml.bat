@echo off
title Deep-Live-Cam DirectML
echo ========================================================
echo   Deep-Live-Cam DirectML Launcher
echo ========================================================
echo.

venv\Scripts\python.exe run.py --execution-provider dml
pause
