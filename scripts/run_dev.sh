#!/bin/bash
echo "Starting TW Buy Stat development server..."

source .venv/bin/activate

echo "Installing dependencies if needed..."
pip install -r requirements.txt > /dev/null 2>&1

echo "Starting FastAPI server..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload