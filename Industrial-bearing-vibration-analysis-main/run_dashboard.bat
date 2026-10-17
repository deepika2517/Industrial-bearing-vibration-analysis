@echo off
title Industrial Bearing Vibration Analysis - Dashboard
cd /d "%~dp0"
echo.
echo   Starting the Bearing Vibration Analysis dashboard...
echo   A browser window will open automatically. Keep this window open.
echo   Press Ctrl+C here to stop the dashboard.
echo.
python -m streamlit run "dashboard\app.py"
echo.
echo   Dashboard stopped.
pause
