@echo off
echo Loading sample data...

call .venv\Scripts\activate.bat

if not exist "tw.csv" (
    echo Error: tw.csv not found in project root
    pause
    exit /b 1
)

if not exist "moloco.csv" (
    echo Error: moloco.csv not found in project root
    pause
    exit /b 1
)

echo Loading TW and Moloco data...
python -m src.main --tw tw.csv --moloco moloco.csv

echo Data loaded successfully!
pause