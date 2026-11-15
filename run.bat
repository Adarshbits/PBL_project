@echo off
title Adarsh AI Clone
echo ===================================================
echo Starting Adarsh AI Clone...
echo ===================================================
py app.py
if %ERRORLEVEL% NEQ 0 (
    python app.py
)
pause
