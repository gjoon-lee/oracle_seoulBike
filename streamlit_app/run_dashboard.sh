#!/bin/bash

# Streamlit Dashboard Runner Script

echo "🚴 서울시 따릉이 통합 분석 시스템 시작..."
echo "================================================"

# Check if virtual environment exists
if [ -d "../.venv" ]; then
    echo "✓ Virtual environment found"
    source ../.venv/Scripts/activate 2>/dev/null || source ../.venv/bin/activate
else
    echo "⚠ Virtual environment not found, using system Python"
fi

# Install required packages
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if API is running
echo "🔍 Checking API server..."
curl -s http://localhost:8000/health > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ API server is running"
else
    echo "⚠ API server is not running. Please start it first:"
    echo "  cd ../realtime_prediction && python main.py"
fi

# Run Streamlit
echo "🚀 Starting Streamlit dashboard..."
echo "================================================"
echo "Dashboard will open at: http://localhost:8501"
echo "Press Ctrl+C to stop"
echo "================================================"

streamlit run dashboard.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false \
    --theme.primaryColor "#5B47FB" \
    --theme.backgroundColor "#FFFFFF" \
    --theme.secondaryBackgroundColor "#F5F5F5" \
    --theme.textColor "#262730"