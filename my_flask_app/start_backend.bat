@echo off
REM Backend startup script for ML Platform (Windows)
REM This script starts Flask and Celery services

echo Starting ML Platform Backend Services...

REM Create necessary directories
if not exist uploads mkdir uploads
if not exist models mkdir models
if not exist logs mkdir logs

REM Check if .env file exists
if not exist .env (
    echo Creating .env file...
    (
        echo # Flask Configuration
        echo FLASK_DEBUG=True
        echo FLASK_PORT=5000
        echo SECRET_KEY=dev-secret-key-change-in-production
        echo MAX_CONTENT_LENGTH=16777216
        echo.
        echo # CORS Configuration
        echo CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
        echo.
        echo # File Storage
        echo UPLOAD_FOLDER=uploads
        echo MODEL_STORAGE_PATH=models
        echo.
        echo # Celery Configuration
        echo CELERY_BROKER_URL=redis://localhost:6379/0
        echo CELERY_RESULT_BACKEND=redis://localhost:6379/0
        echo.
        echo # Redis Configuration
        echo REDIS_HOST=localhost
        echo REDIS_PORT=6379
        echo REDIS_DB=0
        echo.
        echo # OpenRouter API Configuration
        echo OPENROUTER_API_KEY=your_openrouter_api_key_here
        echo OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
    ) > .env
    echo .env file created! Please update OPENROUTER_API_KEY with your actual API key.
)

echo.
echo ============================================================
echo  ML Platform Backend Startup
echo ============================================================
echo.
echo NOTE: This script requires Redis to be running.
echo For development on Windows, you can use one of these options:
echo.
echo 1. Install Redis for Windows:
echo    Download from: https://github.com/microsoftarchive/redis/releases
echo.
echo 2. Use Docker:
echo    docker run -d -p 6379:6379 redis:latest
echo.
echo 3. Use WSL with Redis:
echo    wsl -e sudo service redis-server start
echo.
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Start Flask application
echo Starting Flask application on port 5000...
echo.
echo API Endpoints will be available at:
echo   Health Check: http://localhost:5000/api/health
echo   Upload File:  http://localhost:5000/api/upload
echo   Get Models:   http://localhost:5000/api/recommend-model
echo   Start Train:  http://localhost:5000/api/train
echo   Train Status: http://localhost:5000/api/training-status/^<task_id^>
echo   Make Predict: http://localhost:5000/api/predict
echo   List Models:  http://localhost:5000/api/models
echo.
echo To start Celery worker (in another terminal):
echo   celery -A tasks worker --loglevel=info
echo.
echo Press Ctrl+C to stop the Flask server
echo ============================================================
echo.

python app.py

pause