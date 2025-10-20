@echo off
echo Setting up TW Buy Stat project...

REM Create virtual environment
echo Creating virtual environment...
python -m venv .venv

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Initialize database
echo Initializing database...
python -c "from src.db.setup import ensure_schema; ensure_schema()"

echo Setup complete!
echo.
echo Next steps:
echo 1. Copy your data files (tw.csv, moloco.csv) to project root
echo 2. Run: scripts\load_data.bat
echo 3. Run: scripts\run_dev.bat
echo.
pause