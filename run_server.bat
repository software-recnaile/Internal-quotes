.\.@echo off
cd /d %~dp0
call .venv\Scripts\activate
echo Starting FastAPI server with virtual environment...

uvicorn Backend.main:app --host 0.0.0.0 --port 8000
pause
