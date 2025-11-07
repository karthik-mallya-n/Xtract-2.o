@echo off
echo ================================
echo Advanced Model Training System
echo ================================
echo.

echo 1. Starting Flask server...
start "Flask Server" cmd /k "python app.py"

echo 2. Waiting for server to start...
timeout /t 5 /nobreak > nul

echo 3. Running advanced training test...
echo.
python test_advanced_training.py

echo.
echo ================================
echo Test completed!
echo Check the 'models' folder for trained models
echo ================================
pause