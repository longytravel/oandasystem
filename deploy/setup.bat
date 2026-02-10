@echo off
setlocal enabledelayedexpansion
title OANDA Trading System - VPS Setup
color 0A

echo.
echo ============================================================
echo    OANDA TRADING SYSTEM - ONE-CLICK VPS SETUP
echo ============================================================
echo.
echo This will:
echo   1. Install Python dependencies
echo   2. Configure your OANDA API credentials
echo   3. Install all strategies as Windows services
echo   4. Start trading
echo.
echo Re-run this anytime to add/update strategies.
echo Edit deploy\strategies.json to configure what runs.
echo.
echo ============================================================
echo.

:: Check we're in the right directory
if not exist "scripts\run_live.py" (
    echo ERROR: Run this from the oandasystem root directory.
    echo   cd C:\Trading\oandasystem
    echo   deploy\setup.bat
    pause
    exit /b 1
)

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.11+ from python.org
    echo   Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
echo [OK] Python found

:: Install deps
echo.
echo Installing dependencies...
pip install pandas numpy requests loguru pydantic pydantic-settings python-dotenv numba pandas-ta ta pyarrow --quiet
if errorlevel 1 (
    echo WARNING: Some packages may have failed. Continuing...
)
echo [OK] Dependencies installed

:: Setup .env
echo.
if exist ".env" (
    echo [OK] .env file already exists
) else (
    echo --- OANDA API Configuration ---
    echo.
    echo You need your OANDA Practice account API key.
    echo Get it from: https://www.oanda.com/demo-account/tpa/personal_token
    echo.
    set /p "API_KEY=Enter OANDA API Key: "
    set /p "ACCOUNT_ID=Enter OANDA Account ID (e.g. 101-004-12345678-001): "
    echo.
    set /p "TG_TOKEN=Telegram Bot Token (press Enter to skip): "
    set /p "TG_CHAT=Telegram Chat ID (press Enter to skip): "

    (
        echo OANDA_API_KEY=!API_KEY!
        echo OANDA_ACCOUNT_TYPE=practice
        echo OANDA_ACCOUNT_ID=!ACCOUNT_ID!
        echo TELEGRAM_BOT_TOKEN=!TG_TOKEN!
        echo TELEGRAM_CHAT_ID=!TG_CHAT!
    ) > .env

    echo [OK] .env created
)

:: Test OANDA connection
echo.
echo Testing OANDA connection...
python -c "from config.settings import settings; from live.oanda_client import OandaClient; c = OandaClient(); c.account_id = settings.OANDA_ACCOUNT_ID or c.get_accounts()[0]['id']; a = c.get_account_summary(); print(f'[OK] Connected - Balance: {a[\"balance\"]}, Account: {c.account_id}')"
if errorlevel 1 (
    echo ERROR: Could not connect to OANDA. Check your API key in .env
    pause
    exit /b 1
)

:: Download NSSM if not present
echo.
if exist "deploy\nssm\nssm.exe" (
    echo [OK] NSSM already downloaded
) else (
    echo Downloading NSSM (service manager)...
    if not exist "deploy\nssm" mkdir deploy\nssm
    powershell -Command "Invoke-WebRequest -Uri 'https://nssm.cc/release/nssm-2.24.zip' -OutFile 'deploy\nssm\nssm.zip'"
    if errorlevel 1 (
        echo ERROR: Failed to download NSSM.
        echo   Download manually from https://nssm.cc/download
        echo   Put nssm.exe in deploy\nssm\
        pause
        exit /b 1
    )
    powershell -Command "Expand-Archive -Path 'deploy\nssm\nssm.zip' -DestinationPath 'deploy\nssm\temp' -Force"
    copy "deploy\nssm\temp\nssm-2.24\win64\nssm.exe" "deploy\nssm\nssm.exe" >nul
    rmdir /s /q "deploy\nssm\temp"
    del "deploy\nssm\nssm.zip"
    echo [OK] NSSM downloaded
)

:: Hand off to Python to install all strategies from strategies.json
echo.
echo ============================================================
echo    Installing strategies from deploy\strategies.json
echo ============================================================
echo.
python deploy\install_services.py
if errorlevel 1 (
    echo.
    echo ERROR: Service installation failed. Check output above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo    SETUP COMPLETE
echo ============================================================
echo.
echo   Quick commands:
echo     Status:   deploy\nssm\nssm.exe status OandaTrader_STRATEGY_ID
echo     Logs:     type instances\STRATEGY_ID\logs\stderr.log
echo     Stop:     deploy\nssm\nssm.exe stop OandaTrader_STRATEGY_ID
echo     Restart:  deploy\nssm\nssm.exe restart OandaTrader_STRATEGY_ID
echo.
echo   Add a strategy:
echo     1. Edit deploy\strategies.json
echo     2. Put config.json in instances\STRATEGY_ID\
echo     3. Re-run deploy\setup.bat
echo.
echo ============================================================
pause
