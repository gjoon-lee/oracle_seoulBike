@echo off
echo Starting Seoul Bike Dashboard System...

REM Start FastAPI server in a new window on port 8001
echo Starting FastAPI server on port 8001...
start "FastAPI Server" cmd /k "cd realtime_prediction && ..\.venv\Scripts\python.exe main.py"

REM Wait for server to start
echo Waiting for server to initialize...
timeout /t 5 /nobreak > nul

REM Start Streamlit dashboard
echo Starting Streamlit dashboard...
cd streamlit_app
..\.venv\Scripts\python.exe -m streamlit run dashboard.py --server.port 8501 --server.address localhost

echo Dashboard closed. Please close the FastAPI server window manually.