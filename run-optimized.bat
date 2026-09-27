@echo off
if not exist venv\Scripts\activate (
    echo [INFO] Creating Virtual Environment...
    python -m venv venv
)
call venv\Scripts\activate.bat
python run.py --execution-provider dml --max-memory 4 --execution-threads 1 --lang th
pause
