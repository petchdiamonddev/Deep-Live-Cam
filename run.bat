@echo off
title Deep-Live-Cam One-Click Setup & Launcher
echo ========================================================
echo   Deep-Live-Cam Automatic Setup & Launcher
echo ========================================================
echo.

:: 1. Create Virtual Environment if missing
if not exist venv\Scripts\activate.bat (
    echo [INFO] Creating Virtual Environment (venv)...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Python not found or failed to create venv.
        pause
        exit /b 1
    )
)

:: 2. Activate Virtual Environment
echo [INFO] Activating Virtual Environment...
call venv\Scripts\activate.bat

:: 3. Install/Check Dependencies
echo [INFO] Checking & installing required packages...
pip install -r requirements.txt --quiet --default-timeout=1000

:: 4. Create models directory and download required models if missing
if not exist models (
    mkdir models
)

if not exist models\inswapper_128_fp16.onnx (
    echo [INFO] Downloading inswapper_128_fp16.onnx model (approx 264 MB)...
    curl.exe -L -o models\inswapper_128_fp16.onnx "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx"
)

if not exist models\GFPGANv1.4.onnx (
    echo [INFO] Downloading GFPGANv1.4.onnx model (approx 324 MB)...
    curl.exe -L -o models\GFPGANv1.4.onnx "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx"
)

:: 5. Launch Deep-Live-Cam
echo.
echo [INFO] Launching Deep-Live-Cam with DirectML GPU...
echo ========================================================
python run.py --execution-provider dml --lang th

pause
