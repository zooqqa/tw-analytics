@echo off
echo Starting TW Buy Stat development server...

call .venv\Scripts\activate.bat

echo Installing dependencies if needed...
pip install -r requirements.txt > nul 2>&1

echo Starting FastAPI server...
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

pause