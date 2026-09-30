@echo off
title Deep-Live-Cam Launcher
echo ========================================================
echo   Deep-Live-Cam Automatic Launcher
echo ========================================================
echo.

if not exist venv\Scripts\python.exe (
    echo [INFO] Creating Virtual Environment...
    python -m venv venv
)

if not exist models (
    mkdir models
)

if not exist models\inswapper_128_fp16.onnx (
    echo [INFO] Downloading inswapper_128_fp16.onnx model...
    curl.exe -L -o models\inswapper_128_fp16.onnx "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx"
)

if not exist models\GFPGANv1.4.onnx (
    echo [INFO] Downloading GFPGANv1.4.onnx model...
    curl.exe -L -o models\GFPGANv1.4.onnx "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx"
)

echo [INFO] Launching Deep-Live-Cam...
venv\Scripts\python.exe run.py --execution-provider dml --lang th

pause
