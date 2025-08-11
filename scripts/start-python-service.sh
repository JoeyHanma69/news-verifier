#!/bin/bash

# Start Python API services for news verification

echo "Starting Python ML Services..."

# Check if Python virtual environment exists
if [ ! -d "python-api/venv" ]; then
    echo "Creating Python virtual environment..."
    cd python-api
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd ..
fi

# Activate virtual environment
source python-api/venv/bin/activate

# Start FastAPI server
echo "Starting FastAPI server on port 8000..."
cd python-api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &

# Store PID for cleanup
FASTAPI_PID=$!
echo $FASTAPI_PID > ../fastapi.pid

echo "Python services started!"
echo "FastAPI server: http://localhost:8000"
echo "API docs: http://localhost:8000/docs"

# Keep script running
wait $FASTAPI_PID
