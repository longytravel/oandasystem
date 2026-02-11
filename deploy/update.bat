@echo off
title OANDA Trading System - Update
cd /d C:\Trading\oandasystem

echo.
echo ============================================================
echo    OANDA TRADING SYSTEM - UPDATE + RESTART
echo ============================================================
echo.

echo Pulling latest changes...
git pull
if errorlevel 1 (
    echo.
    echo WARNING: git pull failed. Check for merge conflicts.
    pause
    exit /b 1
)
echo [OK] Code updated
echo.

echo Restarting all services...
python deploy\install_services.py
if errorlevel 1 (
    echo.
    echo ERROR: Service restart failed. Check output above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo    UPDATE COMPLETE
echo ============================================================
echo.
echo   Dashboard: http://localhost:8080
echo.
echo ============================================================
pause
