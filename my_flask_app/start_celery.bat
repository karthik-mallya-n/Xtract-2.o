@echo off
REM Celery Worker startup script for Windows

echo Starting Celery Worker for ML Platform...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if Redis is running by trying to import redis
python -c "import redis; r=redis.Redis(); r.ping()" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Cannot connect to Redis server.
    echo Please ensure Redis is running on localhost:6379
    echo.
    echo Options to start Redis:
    echo   1. Docker: docker run -d -p 6379:6379 redis:latest
    echo   2. WSL: wsl -e sudo service redis-server start
    echo   3. Windows Redis: Start redis-server.exe
    echo.
    pause
)

echo Starting Celery worker...
echo Press Ctrl+C to stop the worker
echo ============================================================
echo.

celery -A tasks worker --loglevel=info

pause