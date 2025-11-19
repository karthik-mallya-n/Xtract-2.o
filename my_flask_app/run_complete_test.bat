@echo off
echo.
echo ==========================================
echo   ML Platform - Complete Test Suite
echo ==========================================
echo.

echo Starting Flask server...
start "Flask Server" cmd /c "cd /d %~dp0 && python app.py"

echo Waiting for server to start...
timeout /t 5 /nobreak > nul

echo.
echo Running complete ML pipeline test...
echo.

python test_complete_pipeline.py

echo.
echo ==========================================
echo   Test Complete!
echo ==========================================
echo.
echo Press any key to close Flask server...
pause > nul

echo Stopping Flask server...
taskkill /f /fi "WindowTitle eq Flask Server*" > nul 2>&1

echo.
echo All done!
pause