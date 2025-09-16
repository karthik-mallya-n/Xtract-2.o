#!/bin/bash
# Backend startup script for ML Platform
# This script starts Flask, Redis, and Celery services

echo "Starting ML Platform Backend Services..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1
}

echo -e "${YELLOW}Setting up environment...${NC}"

# Create necessary directories
mkdir -p uploads models logs

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cat > .env << EOL
# Flask Configuration
FLASK_DEBUG=True
FLASK_PORT=5000
SECRET_KEY=dev-secret-key-change-in-production
MAX_CONTENT_LENGTH=16777216

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# File Storage
UPLOAD_FOLDER=uploads
MODEL_STORAGE_PATH=models

# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Redis Configuration  
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# OpenRouter API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
EOL
    echo -e "${GREEN}.env file created! Please update OPENROUTER_API_KEY with your actual API key.${NC}"
fi

# Check if Redis is installed and running
echo -e "${YELLOW}Checking Redis...${NC}"
if command_exists redis-server; then
    if ! port_in_use 6379; then
        echo -e "${YELLOW}Starting Redis server...${NC}"
        redis-server --daemonize yes --port 6379
        sleep 2
        if port_in_use 6379; then
            echo -e "${GREEN}Redis started successfully on port 6379${NC}"
        else
            echo -e "${RED}Failed to start Redis${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}Redis is already running on port 6379${NC}"
    fi
else
    echo -e "${RED}Redis is not installed. Please install Redis:${NC}"
    echo "  Ubuntu/Debian: sudo apt-get install redis-server"
    echo "  macOS: brew install redis"
    echo "  Windows: Download from https://redis.io/download"
    exit 1
fi

# Start Celery worker in background
echo -e "${YELLOW}Starting Celery worker...${NC}"
celery -A tasks worker --loglevel=info --detach --pidfile=logs/celery.pid --logfile=logs/celery.log

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Celery worker started successfully${NC}"
else
    echo -e "${RED}Failed to start Celery worker${NC}"
    exit 1
fi

# Start Flask application
echo -e "${YELLOW}Starting Flask application...${NC}"
python app.py &
FLASK_PID=$!

# Wait a moment for Flask to start
sleep 3

# Check if Flask is running
if port_in_use 5000; then
    echo -e "${GREEN}Flask application started successfully on port 5000${NC}"
    echo -e "${GREEN}Backend services are ready!${NC}"
    echo ""
    echo -e "${YELLOW}API Endpoints:${NC}"
    echo "  Health Check: http://localhost:5000/api/health"
    echo "  Upload File:  http://localhost:5000/api/upload"
    echo "  Get Models:   http://localhost:5000/api/recommend-model"
    echo "  Start Train:  http://localhost:5000/api/train"
    echo "  Train Status: http://localhost:5000/api/training-status/<task_id>"
    echo "  Make Predict: http://localhost:5000/api/predict"
    echo "  List Models:  http://localhost:5000/api/models"
    echo ""
    echo -e "${YELLOW}To stop services:${NC}"
    echo "  1. Press Ctrl+C to stop Flask"
    echo "  2. Kill Celery: pkill -f 'celery.*worker'"
    echo "  3. Stop Redis: redis-cli shutdown"
    echo ""
    echo -e "${GREEN}Waiting for Flask application... (Press Ctrl+C to stop)${NC}"
    wait $FLASK_PID
else
    echo -e "${RED}Flask application failed to start${NC}"
    # Clean up Celery if Flask fails
    pkill -f "celery.*worker"
    exit 1
fi

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    
    # Stop Celery worker
    if [ -f logs/celery.pid ]; then
        kill $(cat logs/celery.pid) 2>/dev/null
        rm -f logs/celery.pid
        echo -e "${GREEN}Celery worker stopped${NC}"
    fi
    
    # Note: We don't stop Redis as it might be used by other applications
    echo -e "${YELLOW}Redis left running (use 'redis-cli shutdown' to stop manually)${NC}"
    
    echo -e "${GREEN}Backend services stopped${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM