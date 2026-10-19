@echo off
title AI Spam Email and SMS Classifier
echo ======================================================
echo          AI Spam Email and SMS Classifier
echo ======================================================
echo.

:: Fast dependency check: Only run pip if packages are missing
python -c "import streamlit, sklearn, pandas, numpy, joblib" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [Checking dependencies - first run only...]
    python -m pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install dependencies. Please check your Python environment.
        pause
        exit /b 1
    )
)

echo [Launching application in dedicated window...]
echo ======================================================
echo  * Opening in its own separate window.
echo  * Your other browser tabs will NOT be affected.
echo  * Click 'Stop Session' in the app to close cleanly.
echo ======================================================
echo.

python launch.py
