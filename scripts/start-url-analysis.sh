#!/bin/bash

# Start URL-based news verification services

echo "Starting URL-based News Verification System..."

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

echo "URL-based analysis services started!"
echo "FastAPI server: http://localhost:8000"
echo "API docs: http://localhost:8000/docs"

# Example usage
echo ""
echo "Example usage:"
echo "1. Analyze single URL:"
echo "   curl -X POST http://localhost:8000/analyze/url -H 'Content-Type: application/json' -d '{\"url\": \"https://www.reuters.com/world/\"}'"
echo ""
echo "2. Batch analyze URLs:"
echo "   python scripts/url_batch_analyzer.py --urls_file example_urls.txt --output results.json"
echo ""
echo "3. Check source credibility:"
echo "   python scripts/source_credibility_checker.py --domain reuters.com"

# Keep script running
wait $FASTAPI_PID
