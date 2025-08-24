#!/bin/bash

# Start the FastAPI server in the background
echo "Starting FastAPI server..."
cd realtime_prediction
../.venv/Scripts/python.exe main.py &
FASTAPI_PID=$!

# Wait for the server to start
sleep 5

# Start the Streamlit dashboard
echo "Starting Streamlit dashboard..."
cd ../streamlit_app
../.venv/Scripts/python.exe -m streamlit run dashboard.py --server.port 8501 --server.address localhost

# Kill the FastAPI server when streamlit exits
kill $FASTAPI_PID